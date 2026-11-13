"""Fenced, metadata-only retirement of duplicate persistent TPU pool bindings.

The controller owns the operation. A durable intent quarantines the alias
before either native database changes; replay completes only the same intent.
No provider lifecycle or API request mutation belongs in this module.
"""
import dataclasses
import hashlib
import json
import pathlib
import pickle
from typing import List

from sky import global_user_state
from sky.adaptors import common as adaptors_common
from sky.jobs import state as job_state
from sky.provision.gcp import instance_utils
from sky.provision.gcp import queued_resource_wait as queue
from sky.serve import serve_state
from sky.server.requests import payloads
from sky.utils import common_utils

ROOT = pathlib.Path('~/.sky/serve/alias_retirements')
pool_recovery = adaptors_common.LazyImport('sky.serve.pool_recovery')


@dataclasses.dataclass(frozen=True)
class AliasRetirement:
    """Exact expected bindings, supplied by an independently reviewed audit."""

    pool_hash: str
    owner: str
    workspace: str
    version: int
    alias_id: int
    keeper_id: int
    alias_hash: str
    keeper_hash: str
    alias_request: str
    keeper_request: str
    project: str
    zone: str
    cloud_name: str
    node_name: str
    node_id: str
    node_create_time: str
    queue_name: str
    queue_create_time: str
    reason: str

    def __post_init__(self):
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if field.name in ('version', 'alias_id', 'keeper_id'):
                if (not isinstance(value, int) or isinstance(value, bool) or
                        value < 1):
                    raise ValueError(f'Invalid {field.name}')
            elif not isinstance(value, str) or not value.strip():
                raise ValueError(f'Missing {field.name}')
        if (self.alias_id == self.keeper_id or
                self.alias_hash == self.keeper_hash):
            raise ValueError(
                'Alias and keeper must have distinct native identities')
        parent = f'projects/{self.project}/locations/{self.zone}'
        if (not self.node_name.startswith(parent + '/nodes/') or
                not self.queue_name.startswith(parent + '/queuedResources/')):
            raise ValueError('Provider placement does not match the intent')


def _path(intent: AliasRetirement) -> pathlib.Path:
    key = json.dumps([intent.pool_hash, intent.alias_id, intent.alias_hash])
    return ROOT.expanduser() / (hashlib.sha256(key.encode()).hexdigest() +
                                '.json')


def _receipts(service_name: str):
    service = serve_state.get_service_from_name(service_name)
    if service is None:
        return
    for path in ROOT.expanduser().glob('*.json'):
        receipt = json.loads(path.read_text())
        if (receipt['service_name'] == service_name and
                receipt['intent']['pool_hash'] == service['hash']):
            yield receipt


def retired_replica_ids(service_name: str) -> List[int]:
    """Prepared intents also quarantine scheduling and reserve their IDs."""
    return [r['intent']['alias_id'] for r in _receipts(service_name)]


def pending_intents(service_name: str) -> List[AliasRetirement]:
    return [
        AliasRetirement(**r['intent'])
        for r in _receipts(service_name)
        if r['stage'] == 'prepared'
    ]


def _service_fence(service_name: str, intent: AliasRetirement) -> None:
    service = serve_state.get_service_from_name(service_name)
    if (service is None or not service['pool'] or
            service['hash'] != intent.pool_hash or
            service['user_hash'] != intent.owner or
            common_utils.get_user_hash() != intent.owner or
            service['current_version'] != intent.version):
        raise ValueError('Pool owner, version or incarnation changed')


def _request_fence(name: str, request_id: str, cluster_hash: str,
                   owner: str) -> 'payloads.LaunchBody':
    # Imported lazily to avoid requests -> serve -> requests initialization.
    # pylint: disable=import-outside-toplevel
    from sky.server.requests import requests
    request = requests.get_request(request_id)
    if (request is None or request.cluster_name != name or
            request.user_id != owner or
            request.status != requests.RequestStatus.SUCCEEDED or
            not isinstance(request.request_body, payloads.LaunchBody) or
            not request.request_body.is_launched_by_sky_serve_controller or
            request.request_body.is_launched_by_jobs_controller):
        raise ValueError('Original successful pool launch binding changed')
    attempts = global_user_state.get_launch_attempts_for_cluster(name)
    if not any(a.request_id == request_id and a.cluster_hash == cluster_hash and
               a.outcome == 'succeeded' for a in attempts):
        raise ValueError('Original launch attempt binding changed')
    if requests.get_request_tasks(
            requests.RequestTaskFilter(
                cluster_names=[name],
                status=requests.RequestStatus.active_statuses())):
        raise ValueError('Worker has an active API operation')
    return request.request_body


def _cluster_fence(service_name: str, intent: AliasRetirement,
                   receipt: dict) -> None:
    bodies = []
    for replica_id, expected_hash in ((intent.keeper_id, intent.keeper_hash),
                                      (intent.alias_id, intent.alias_hash)):
        name = f'{service_name}-{replica_id}'
        info = serve_state.get_replica_info_from_id(service_name, replica_id)
        cluster = global_user_state.get_cluster_from_name(name)
        resuming = bool(receipt) and replica_id == intent.alias_id
        if info is None:
            if not resuming:
                raise ValueError('Expected READY replica is missing')
        elif (info.cluster_name != name or info.version != intent.version or
              info.status != serve_state.ReplicaStatus.READY):
            raise ValueError('Replica identity or readiness changed')
        if cluster is None:
            if not resuming:
                raise ValueError('Expected native cluster is missing')
        elif (cluster['cluster_hash'] != expected_hash or
              cluster['user_hash'] != intent.owner or
              cluster['workspace'] != intent.workspace or
              cluster['handle'].cluster_name_on_cloud != intent.cloud_name):
            raise ValueError('Native cluster binding changed')
        request_id = (intent.keeper_request if replica_id == intent.keeper_id
                      else intent.alias_request)
        bodies.append(
            _request_fence(name, request_id, expected_hash, intent.owner))
    if not pool_recovery.equivalent_worker_launches(bodies[0], bodies[1]):
        raise ValueError('Alias and keeper immutable launch inputs differ')


def _provider_fence(intent: AliasRetirement) -> None:
    journals = list(
        queue.matching_journals(intent.project, intent.zone, intent.cloud_name))
    if len(journals) != 1:
        raise ValueError('Expected one original allocation journal')
    _, journal = journals[0]
    if (journal.get('stage') != 'ready' or
            journal.get('native_cluster_hash') != intent.keeper_hash or
            journal['request_id'] != intent.keeper_request or
            set(journal['resources']) != {intent.queue_name}):
        raise ValueError('Original allocation owner changed')
    row = journal['resources'][intent.queue_name]
    if row.get('create_time') != intent.queue_create_time:
        raise ValueError('Original queued generation changed')
    specs = row['body']['tpu']['nodeSpec']
    if isinstance(specs, dict):
        specs = [specs]
    node_suffix = intent.node_name.rsplit('/', 1)[-1]
    if (len(specs) != 1 or specs[0].get('nodeId') != node_suffix or
            specs[0].get('multiNodeParams')):
        raise ValueError('Expected one exact TPU node')
    parent = intent.queue_name.rsplit('/queuedResources/', 1)[0]
    canonical = row.get('canonical_parent', parent)
    queue_names = {
        intent.queue_name,
        intent.queue_name.replace(parent, canonical, 1)
    }
    api = instance_utils.GCPTPUVMInstance.load_resource().projects().locations()
    resource = api.queuedResources().get(name=intent.queue_name).execute(
        num_retries=2)
    if (resource.get('name') not in queue_names or
            resource.get('state', {}).get('state') != 'ACTIVE'):
        raise ValueError('Provider allocation is no longer ACTIVE')
    queue._validate(resource, resource['name'], row)  # pylint: disable=protected-access
    node = api.nodes().get(name=intent.node_name).execute(num_retries=2)
    node_names = {
        intent.node_name,
        intent.node_name.replace(parent, canonical, 1)
    }
    if (node.get('name') not in node_names or
            str(node.get('id')) != intent.node_id or
            node.get('createTime') != intent.node_create_time or
            node.get('state') != 'READY' or
            node.get('queuedResource') not in queue_names or node.get(
                'labels', {}).get('ray-cluster-name') != intent.cloud_name):
        raise ValueError('Provider node generation or binding changed')


def retire(service_name: str, intent: AliasRetirement) -> dict:
    """Caller holds controller, scheduling, cluster and allocation locks."""
    _service_fence(service_name, intent)
    path = _path(intent)
    receipt = json.loads(path.read_text()) if path.exists() else {}
    if receipt:
        if (receipt['service_name'] != service_name or
                receipt['intent'] != dataclasses.asdict(intent)):
            raise ValueError('Retirement intent changed')
        if receipt['stage'] == 'committed':
            return receipt
        if receipt['stage'] != 'prepared':
            raise ValueError('Unknown retirement stage')
    if job_state.get_nonterminal_job_ids_by_pool(service_name):
        raise ValueError('Pool has nonterminal jobs; retirement refused')
    _cluster_fence(service_name, intent, receipt)
    _provider_fence(intent)
    if not receipt:
        alias = serve_state.get_replica_info_from_id(service_name,
                                                     intent.alias_id)
        receipt = dict(service_name=service_name,
                       intent=dataclasses.asdict(intent),
                       stage='prepared',
                       alias_record=pickle.dumps(alias).hex())
        queue.save(path, receipt)
    # All history, launch attempts, API requests and provider journals remain.
    # Replaying a prepared intent can finish either partially applied step.
    name = f'{service_name}-{intent.alias_id}'
    if global_user_state.get_cluster_from_name(name) is not None:
        global_user_state.remove_cluster(name, terminate=True)
    serve_state.remove_replica(service_name, intent.alias_id)
    receipt['stage'] = 'committed'
    queue.save(path, receipt)
    return receipt
