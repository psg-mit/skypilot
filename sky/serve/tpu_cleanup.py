"""Reconcile exact, durably owned terminal TPU allocations of native pools.

The allocation receipt, not a mutable cluster-name registration, owns cleanup.
The controller also fences scheduling, native launch/status and provider
allocation locks. Uncertain creates are parked.
"""
import dataclasses
# pylint: disable=protected-access
import hashlib
import json
import pathlib
import time
from typing import Optional

import filelock

from sky import global_user_state
from sky import sky_logging
from sky import task as task_lib
from sky.adaptors import common as adaptors_common
from sky.adaptors import gcp
from sky.backends import backend_utils
from sky.jobs import state as job_state
from sky.provision.gcp import queued_resource_wait as queue
from sky.serve import constants
from sky.serve import pool_recovery
from sky.serve import serve_state
from sky.serve import serve_utils
from sky.server.requests import payloads
from sky.server.requests import restart_recovery
from sky.utils import common_utils
from sky.utils import dag_utils
from sky.utils import locks

api_requests = adaptors_common.LazyImport('sky.server.requests.requests')
instance_utils = adaptors_common.LazyImport('sky.provision.gcp.instance_utils')
logger = sky_logging.init_logger(__name__)
RECHECK_SECONDS = 300


@dataclasses.dataclass(frozen=True)
class PoolAllocationOwner:
    """Original native owner, persisted before provider creation."""

    service_name: str
    pool_hash: str
    owner: str
    workspace: str
    version: int
    replica_id: int
    native_name: str
    native_hash: str
    request_id: str
    launch_sha256: str


def launch_digest(body: payloads.LaunchBody) -> str:
    return hashlib.sha256(
        json.dumps(pool_recovery._binding(body),
                   sort_keys=True).encode()).hexdigest()


def capture_owner(native_hash: Optional[str]) -> Optional[dict]:
    """Capture only an independently registered pool worker's launch binding."""
    request = api_requests.get_request(common_utils.get_current_request_id())
    if (request is None or not native_hash or
            not isinstance(request.request_body, payloads.LaunchBody)):
        return None
    body = request.request_body
    if (not body.is_launched_by_sky_serve_controller or
            body.is_launched_by_jobs_controller):
        return None
    service_name, _, replica_id = request.cluster_name.rpartition('-')
    if not replica_id.isdecimal():
        raise ValueError('Unknown native TPU pool worker')
    service = serve_state.get_service_from_name(service_name)
    replica = serve_state.get_replica_info_from_id(service_name,
                                                   int(replica_id))
    cluster = global_user_state.get_cluster_from_name(request.cluster_name)
    if (service is None or not service['pool'] or replica is None or
            service['user_hash'] != request.user_id or cluster is None or
            cluster['cluster_hash'] != native_hash or
            cluster['user_hash'] != request.user_id or
            replica.cluster_name != request.cluster_name):
        raise ValueError('Cannot persist an ambiguous TPU cleanup owner')
    return dataclasses.asdict(
        PoolAllocationOwner(service_name, service['hash'], request.user_id,
                            cluster['workspace'], replica.version,
                            int(replica_id), request.cluster_name, native_hash,
                            request.request_id, launch_digest(body)))


def pending_native_allocation(native_name: str,
                              native_hash: Optional[str] = None) -> bool:
    """A missing native row is not proof that its cloud resources are absent."""
    if not restart_recovery.enabled():
        return False
    for _, value in queue.journals():
        if value.get('stage') == 'retired':
            continue
        owner = value.get('pool_owner', {})
        if (owner.get('native_name') == native_name or
            (native_hash and value.get('native_cluster_hash') == native_hash)):
            return True
    return False


def _owner_fence(value: dict) -> PoolAllocationOwner:
    owner = PoolAllocationOwner(**value['pool_owner'])
    if owner.native_name != f'{owner.service_name}-{owner.replica_id}':
        raise ValueError('Pool worker identity tuple is inconsistent')
    service = serve_state.get_service_from_name(owner.service_name)
    request = api_requests.get_request(owner.request_id)
    if (service is None or not service['pool'] or
            service['hash'] != owner.pool_hash or
            service['user_hash'] != owner.owner or
            value['request_id'] != owner.request_id or
            value.get('native_cluster_hash') != owner.native_hash or
            request is None or request.name != 'sky.launch' or
            request.user_id != owner.owner or
            request.cluster_name != owner.native_name or
            not isinstance(request.request_body, payloads.LaunchBody) or
            launch_digest(request.request_body) != owner.launch_sha256 or
            request.status in api_requests.RequestStatus.active_statuses()):
        raise ValueError('Original pool allocation owner is unresolved')
    body = request.request_body
    if (not body.is_launched_by_sky_serve_controller or
            body.is_launched_by_jobs_controller):
        raise ValueError('Original creator is not a native pool launch')
    cluster = global_user_state.get_cluster_from_name(owner.native_name)
    if cluster and (cluster['user_hash'] != owner.owner or
                    cluster['workspace'] != owner.workspace):
        raise ValueError('Native worker owner changed')
    replica = serve_state.get_replica_info_from_id(owner.service_name,
                                                   owner.replica_id)
    if replica and (replica.cluster_name != owner.native_name or
                    replica.status.value in ('READY', 'NOT_READY')):
        raise ValueError('Pool worker is still eligible for work')
    jobs = job_state.get_nonterminal_job_ids_by_pool_grouped(owner.service_name)
    if jobs.get(owner.native_name):
        raise ValueError('Pool worker still has assigned work')
    for job_id in jobs.get(None, []):
        # The service scheduling lock fences assignment while cleanup runs.
        # Jobs still waiting for a worker do not use this terminal allocation;
        # an unbound running job or a retained submit ID remains ambiguous.
        if (job_state.get_status(job_id) not in {
                job_state.ManagedJobStatus.PENDING,
                job_state.ManagedJobStatus.STARTING,
                job_state.ManagedJobStatus.RECOVERING,
        } or job_state.get_pool_submit_info(job_id) != (None, None)):
            raise ValueError('Pool has an unresolved job assignment')
    for other in api_requests.get_request_tasks(
            api_requests.RequestTaskFilter(
                cluster_names=[owner.native_name],
                status=api_requests.RequestStatus.active_statuses())):
        if (other.user_id != owner.owner or other.pid is not None or
                other.status != api_requests.RequestStatus.WAITING or
                not isinstance(other.request_body, payloads.LaunchBody) or
                not pool_recovery.equivalent_worker_launches(
                    body, other.request_body)):
            raise ValueError('Worker has an executing or unrelated operation')
    return owner


def _get(endpoint, name: str) -> Optional[dict]:
    try:
        return endpoint.get(name=name).execute(num_retries=2)
    except gcp.http_error_exception() as error:
        if error.resp.status != 404:
            raise
        return None


def _terminal_resources(value: dict, api) -> tuple:
    """Re-read every exact request and every guest; no SSH inference."""
    resources = {}
    guests = {}
    for name, row in value['resources'].items():
        resource = _get(api.queuedResources(), name)
        if not row.get('create_time'):
            rejected = (row.get('create_error', {}).get('code')
                        in queue.DEFINITIVE_CREATE_ERRORS)
            if (resource is not None or row.get('phase') != 'rejected' or
                    not (rejected or queue._operation_error(row))):
                raise ValueError(
                    'Uncertain TPU create requires operator reconciliation')
        if resource is not None:
            queue._validate(resource, name, row)
            if resource.get('state',
                            {}).get('state') not in ('FAILED', 'SUSPENDED'):
                raise ValueError('TPU request is not terminal')
        parent = 'projects/{}/locations/{}'.format(value['project'],
                                                   value['zone'])
        for spec in queue._specs(row['body']):
            node_id = spec.get('nodeId')
            if not node_id or spec.get('multiNodeParams'):
                raise ValueError('TPU guest identity is incomplete')
            node = _get(api.nodes(), f'{parent}/nodes/{node_id}')
            guest_name = f'{parent}/nodes/{node_id}'
            guests[guest_name] = ([node.get('id'),
                                   node.get('createTime')]
                                  if node is not None else None)
            previous = value.get('cleanup_guests', {})
            if (guest_name in previous and node is not None and
                    previous[guest_name] != guests[guest_name]):
                raise ValueError('TPU guest incarnation changed during cleanup')
            if node is not None:
                desired = spec['node']
                if (resource is None or node.get('state') != 'PREEMPTED' or
                        not node.get('id') or not node.get('createTime') or any(
                            node.get(k) != desired.get(k)
                            for k in ('acceleratorType', 'runtimeVersion',
                                      'labels'))):
                    raise ValueError('TPU guest is live, changed or unresolved')
        resources[name] = resource
    return resources, guests


def reconcile_allocation(path: pathlib.Path) -> PoolAllocationOwner:
    """Finish exact terminal cleanup, preserving original identities."""
    value = queue.load(path)
    owner = _owner_fence(value)
    if value.get('stage') == 'retired':
        return owner
    with filelock.FileLock(
            serve_utils.get_service_filelock_path(owner.service_name),
            timeout=0), locks.get_lock(
                backend_utils.cluster_status_lock_id(owner.native_name),
                timeout=0), queue.allocation_lock(value['project'],
                                                  value['zone'],
                                                  value['cluster'],
                                                  timeout=0):
        value = queue.load(path)
        _owner_fence(value)
        api = instance_utils.GCPTPUVMInstance.load_resource().projects(
        ).locations()
        for name, row in value['resources'].items():
            operation = row.get('cleanup_operation')
            if operation:
                result = api.operations().get(name=operation['name']).execute(
                    num_retries=2)
                if not result.get('done'):
                    return owner
                if result.get('error'):
                    raise ValueError('TPU deletion failed; operation retained')
            elif row.get('cleanup_submitting'):
                if _get(api.queuedResources(), name) is not None:
                    raise ValueError(
                        'Uncertain deletion response; retaining intent')
        resources, guests = _terminal_resources(value, api)
        value.setdefault('cleanup_guests', guests)
        for name, resource in resources.items():
            if resource is None:
                continue
            if value['resources'][name].get('cleanup_operation'):
                raise ValueError(
                    'Deletion finished but request remains; retaining operation'
                )
            # Repeat ownership and provider/guest guards immediately before RPC.
            _owner_fence(value)
            _terminal_resources(value, api)
            row = value['resources'][name]
            row['cleanup_submitting'] = True
            value['cleanup_started_at'] = value.get('cleanup_started_at',
                                                    time.time())
            queue.save(path, value)
            operation = api.queuedResources().delete(
                name=name, force=True).execute(num_retries=0)
            row['cleanup_operation'] = operation
            queue.save(path, value)
            return owner  # Poll the retained operation on a later pass.
        value['stage'] = 'retired'
        value['cleanup_completed_at'] = time.time()
        value.pop('cleanup_error', None)
        queue.save(path, value)
        return owner


def reconcile_pool(service_name: str, excluded_clusters=()) -> list:
    """Bound each pass and retain an actionable reason on failure."""
    if not restart_recovery.enabled():
        return []
    completed = []
    candidates = []
    active_statuses = api_requests.RequestStatus.active_statuses()
    for path, value in queue.journals():
        owner = value.get('pool_owner', {})
        if (owner.get('service_name') != service_name or
                owner.get('native_name') in excluded_clusters or
                value.get('stage') == 'retired' or
                value.get('cleanup_recheck_at', 0) > time.time()):
            continue
        creator = api_requests.get_request(owner['request_id'])
        if creator and creator.status in active_statuses:
            continue
        replica = serve_state.get_replica_info_from_id(service_name,
                                                       owner['replica_id'])
        if replica and replica.status in (serve_state.ReplicaStatus.READY,
                                          serve_state.ReplicaStatus.NOT_READY):
            # A successful creator remains terminal for the lifetime of a
            # healthy worker; it is not a cleanup failure. The deletion fence
            # still rechecks readiness immediately before every provider RPC.
            continue
        candidates.append((value.get('cleanup_recheck_at', 0), path))
    for _, path in sorted(candidates)[:2]:
        value = queue.load(path)
        try:
            owner = reconcile_allocation(path)
            value = queue.load(path)
            if value.get('stage') == 'retired':
                completed.append(owner)
        except Exception as error:  # pylint: disable=broad-except
            value = queue.load(path)
            value['cleanup_error'] = f'{type(error).__name__}: {error}'
            logger.warning('TPU allocation reconciliation parked: %s: %s',
                           value.get('cluster'), value['cleanup_error'])
        value['cleanup_recheck_at'] = time.time() + RECHECK_SECONDS
        queue.save(path, value)
    return completed


def bind_retained_allocation(path: pathlib.Path, owner: PoolAllocationOwner,
                             expected_sha256: str) -> None:
    """Enroll an operator-reviewed old receipt without weakening cleanup guards.

    Existing creator/history records must independently prove its original
    native generation. Missing records require archive reconciliation first.
    """
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError('Reviewed allocation journal changed')
    value = queue.load(path)
    if value.get('pool_owner') or value.get('stage') == 'retired':
        raise ValueError('Allocation is already bound or retired')
    candidate = dict(value, pool_owner=dataclasses.asdict(owner))
    _owner_fence(candidate)
    histories = global_user_state.get_clusters_from_history(
        cluster_hashes=[owner.native_hash])
    if not any(h['cluster_hash'] == owner.native_hash and h['name'] ==
               owner.native_name and h['user_hash'] == owner.owner and
               h['workspace'] == owner.workspace for h in histories):
        raise ValueError('Original native generation history is missing')
    attempts = global_user_state.get_launch_attempts_for_cluster(
        owner.native_name)
    if not any(a.request_id == owner.request_id and a.cluster_name_on_cloud ==
               value['cluster'] and a.workspace == owner.workspace
               for a in attempts):
        raise ValueError('Original provider launch lineage is missing')
    request = api_requests.get_request(owner.request_id)
    dag = dag_utils.load_chain_dag_from_yaml_str(request.request_body.task)
    if len(dag.tasks) != 1:
        raise ValueError('Original pool launch must contain exactly one task')
    original = dag.tasks[0]
    pool_yaml = serve_state.get_yaml_content(owner.service_name, owner.version)
    if pool_yaml is None:
        raise ValueError('Original pool definition is missing')
    expected = task_lib.Task.from_yaml_str(pool_yaml)
    expected.update_envs({constants.REPLICA_ID_ENV_VAR: str(owner.replica_id)})

    def fingerprint(task):
        content = task.to_yaml_config()
        for key in ('workdir', '_user_specified_yaml', 'file_mounts_mapping'):
            content.pop(key, None)
        return content

    if fingerprint(original) != fingerprint(expected):
        raise ValueError(
            'Original worker differs from the retained pool definition')
    with filelock.FileLock(
            serve_utils.get_service_filelock_path(owner.service_name),
            timeout=0), locks.get_lock(
                backend_utils.cluster_status_lock_id(owner.native_name),
                timeout=0), queue.allocation_lock(value['project'],
                                                  value['zone'],
                                                  value['cluster'],
                                                  timeout=0):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError('Allocation changed during operator audit')
        _owner_fence(candidate)
        candidate['cleanup_binding_audited_at'] = time.time()
        queue.save(path, candidate)
