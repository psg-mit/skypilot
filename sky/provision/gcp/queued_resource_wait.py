"""Durable, request-bound Cloud TPU capacity waits.

Only the API executor opts in. In-process callers retain the blocking path.
The journal is written before create, and survives a lost create response or
server restart. An uncertain or displaced resource is never recreated/adopted.
"""
import asyncio
import contextvars
import copy
import hashlib
import json
import os
import pathlib
import time
from typing import Any, Dict, NoReturn, Optional

import filelock

from sky import exceptions
from sky.adaptors import gcp
from sky.provision import common
from sky.utils import common_utils
from sky.utils import context

JOURNAL_ROOT = pathlib.Path('~/.sky/gcp_queue_waits')
POLL_SECONDS = 30
QUEUE_TIMEOUT = 6000
PROVISIONING_GRACE = 900


def journal_path(request_id: str, project: str, zone: str,
                 cluster: str) -> pathlib.Path:
    key = json.dumps([request_id, project, zone, cluster])
    return JOURNAL_ROOT.expanduser() / (
        hashlib.sha256(key.encode()).hexdigest() + '.json')


def save(path: pathlib.Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with filelock.FileLock(str(path) + '.lock'):
        if (path.exists() and
                json.loads(path.read_text()).get('stage') == 'retired' and
                value.get('stage') != 'retired'):
            return  # A late cancelled probe cannot resurrect retired intent.
        common_utils.atomic_write_text(str(path), json.dumps(value), mode=0o600)
    with path.open('rb') as stream:
        os.fsync(stream.fileno())
    directory = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def load(path: pathlib.Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        pause(path, 'TPU journal unreadable; reconciliation required')


def pause(path: pathlib.Path, reason: str) -> NoReturn:
    raise exceptions.ExecutionPausedError(
        reason,
        hint='The original TPU allocation is retained while waiting.',
        retry_wait_seconds=POLL_SECONDS,
        continue_condition=QueuedResourceCondition(str(path)))


def contract(config: dict, labels: dict, count: int) -> dict:
    """The complete immutable input, before provider defaults are expanded."""
    return dict(config=config, labels=labels, count=count)


def allocation_lock(project: str, zone: str, cluster: str) -> filelock.FileLock:
    """Serialize label ownership, including the first durable create intent."""
    root = JOURNAL_ROOT.expanduser() / 'bindings'
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps([project, zone,
                                     cluster]).encode()).hexdigest()
    return filelock.FileLock(str(root / f'{key}.lock'))


def matching_journals(project: str, zone: str, cluster: str):
    """Read original allocation ownership without changing any journal."""
    for path in JOURNAL_ROOT.expanduser().glob('*.json'):
        value = load(path)
        if ((value['project'], value['zone'], value['cluster'])
                == (project, zone, cluster) and
                value.get('stage') != 'retired'):
            yield path, value


def open_journal(project: str, zone: str, cluster: str,
                 expected: dict) -> Optional[pathlib.Path]:
    if not common_utils.is_in_request_context():
        return None
    path = journal_path(common_utils.get_current_request_id(), project, zone,
                        cluster)
    try:
        current_hash = native_cluster_hash()
    except Exception:  # pylint: disable=broad-except
        pause(path, 'Native TPU identity unavailable; retaining allocation')
    reusable = None
    for other, value in matching_journals(project, zone, cluster):
        if (not current_hash or
                value.get('native_cluster_hash') != current_hash):
            # In particular, READY is still owned. Reusing a cloud label is
            # not authority to attach a different native cluster incarnation.
            pause(path, 'Another native TPU incarnation owns this cloud name')
        if other != path and (value.get('stage') != 'ready' or
                              value.get('contract') != expected):
            pause(path, 'Another original TPU request owns this capacity wait')
        if other != path:
            reusable = other
    if not path.exists() and reusable is not None:
        return reusable
    if path.exists():
        try:
            value = load(path)
        except Exception:  # pylint: disable=broad-except
            pause(path, 'TPU journal unreadable; reconciliation required')
        if value.get('legacy_input'):
            # An operator may import an audited pre-journal request using its
            # immutable native YAML plus the exact provider generation. Bind
            # bootstrap defaults only after they match that provider record.
            original = copy.deepcopy(expected)
            for key in ('networkConfig', 'networkInterfaces', 'serviceAccount'):
                if key not in value['contract']['config']:
                    original['config'].pop(key, None)
            if original != value['contract']:
                pause(path, 'Imported TPU launch inputs changed')
            for row in value['resources'].values():
                for spec in _specs(row['body']):
                    for key in ('networkConfig', 'serviceAccount'):
                        desired = expected['config'].get(key, {})
                        actual = spec['node'].get(key, {})
                        if any(actual.get(k) != v for k, v in desired.items()):
                            pause(path,
                                  'Imported TPU bootstrap identity changed')
            value['contract'] = copy.deepcopy(expected)
            value.pop('legacy_input')
            save(path, value)
        if value['contract'] != expected:
            pause(path, 'TPU launch inputs changed; reconciliation required')
        if value.get('stage') == 'retired':
            archive = path.parent / 'retired'
            archive.mkdir(mode=0o700, exist_ok=True)
            path.rename(archive / f'{path.stem}-{time.time_ns()}.json')
    return path


def validate_existing_nodes(path: pathlib.Path, nodes: list) -> None:
    """Existing nodes need a durable, generation-bound allocation receipt."""
    if not nodes:
        return
    if not path.exists():
        pause(path, 'Existing TPU nodes have no original allocation journal')
    value = load(path)
    expected = {
        spec['nodeId']
        for row in value['resources'].values()
        for spec in _specs(row['body'])
    }
    if (value.get('native_cluster_hash') != native_cluster_hash() or any(
            node['name'].rsplit('/', 1)[-1] not in expected for node in nodes)):
        pause(path, 'Existing TPU nodes belong to another allocation')


def retire(project: str, zone: str, cluster: str) -> None:
    """Called only after native teardown verified the entire queue is absent."""
    for path in JOURNAL_ROOT.expanduser().glob('*.json'):
        value = load(path)
        if (value['project'], value['zone'],
                value['cluster']) == (project, zone, cluster):
            for name, row in value['resources'].items():
                if row['phase'] in ('prepared', 'rejected'):
                    continue
                if not row.get('create_time'):
                    raise common.ProvisionerError(
                        f'Unresolved TPU create {name}; retain ownership')
                try:
                    _resources().get(name=name).execute(num_retries=2)
                except gcp.http_error_exception() as error:
                    if error.resp.status != 404:
                        raise
                else:
                    raise common.ProvisionerError(
                        f'TPU request {name} still exists after teardown')
            value['stage'] = 'retired'
            save(path, value)


def validate_teardown(project: str, zone: str, cluster: str,
                      resources: dict) -> None:
    """Fence known generations and uncertain creates before native deletion."""
    current_hash = native_cluster_hash()
    for path in JOURNAL_ROOT.expanduser().glob('*.json'):
        value = load(path)
        if ((value['project'], value['zone'], value['cluster']) !=
            (project, zone, cluster) or value.get('stage') == 'retired'):
            continue
        if (not current_hash or
                value.get('native_cluster_hash') != current_hash):
            raise common.ProvisionerError(
                'TPU teardown ownership mismatch; retaining original allocation'
            )
        for name, row in value['resources'].items():
            if name in resources:
                try:
                    _validate(resources[name], name, row)
                except ValueError as error:
                    raise common.ProvisionerError(
                        f'TPU generation changed before teardown: {name}'
                    ) from error
                row['create_time'] = resources[name]['createTime']
                row['phase'] = 'observed'
            elif (row['phase'] not in ('prepared', 'rejected') and
                  not row.get('create_time')):
                raise common.ProvisionerError(
                    f'Unresolved TPU create {name}; retain ownership')
        save(path, value)


def prepare(path: pathlib.Path, project: str, zone: str, cluster: str,
            expected: dict, bodies: dict) -> dict:
    """Commit the entire allocation group before sending any create RPC."""
    try:
        canonical = canonical_parent(project, zone)
    except Exception:  # pylint: disable=broad-except
        pause(path, 'TPU project identity unavailable; no create sent')
    value = dict(version=1,
                 stage='creating',
                 request_id=common_utils.get_current_request_id(),
                 project=project,
                 zone=zone,
                 cluster=cluster,
                 contract=copy.deepcopy(expected),
                 queue_deadline=time.time() + QUEUE_TIMEOUT,
                 resources={
                     name: dict(body=body,
                                phase='prepared',
                                canonical_parent=canonical)
                     for name, body in bodies.items()
                 })
    try:
        value['native_cluster_hash'] = native_cluster_hash()
    except Exception:  # pylint: disable=broad-except
        pause(path, 'Native cluster identity unavailable; no create sent')
    save(path, value)
    return value


def native_cluster_hash() -> Optional[str]:
    """Authoritative generation for completed-launch controller reattachment."""
    if not common_utils.is_in_request_context():
        return None
    # These imports would form a requests/provision initialization cycle.
    # pylint: disable=import-outside-toplevel
    from sky import global_user_state
    from sky.server.requests import requests
    request = requests.get_request(common_utils.get_current_request_id())
    if request is None or not request.cluster_name:
        return None
    return global_user_state.get_cluster_hash(request.cluster_name)


def _specs(body: dict) -> list:
    specs = body['tpu']['nodeSpec']
    return [specs] if isinstance(specs, dict) else specs


def canonical_parent(project: str, zone: str) -> str:
    """Resolve Google's numeric parent using an authenticated project read."""
    resource = gcp.build('cloudresourcemanager',
                         'v1',
                         credentials=None,
                         cache_discovery=False).projects().get(
                             projectId=project).execute(num_retries=2)
    number = str(resource.get('projectNumber', ''))
    if not number.isdigit() or project not in (resource.get('projectId'),
                                               number):
        raise ValueError('GCP project identity is not verified')
    return f'projects/{number}/locations/{zone}'


def _validate(resource: dict, name: str, row: dict) -> None:
    """Fence the provider generation and exact allocation/route/owner."""
    expected = row['body']
    if resource.get('name') != name or not resource.get('createTime'):
        raise ValueError('missing provider identity')
    if (row.get('create_time') is not None and
            row['create_time'] != resource['createTime']):
        raise ValueError('provider generation changed')
    if _spot_route(expected) != _spot_route(resource):
        raise ValueError('TPU route changed')
    actual_specs, expected_specs = _specs(resource), _specs(expected)
    if len(actual_specs) != len(expected_specs):
        raise ValueError('allocation group changed')
    for actual, desired in zip(actual_specs, expected_specs):
        parents = {desired['parent']}
        if row.get('canonical_parent'):
            parents.add(row['canonical_parent'])
        if actual.get('parent') not in parents:
            raise ValueError('TPU parent identity changed')
        if any(
                actual.get(key) != desired.get(key)
                for key in ('nodeId', 'multiNodeParams')):
            raise ValueError('TPU node identity changed')
        node, desired_node = actual.get('node', {}), desired['node']
        for key in ('acceleratorType', 'runtimeVersion', 'labels'):
            if node.get(key) != desired_node.get(key):
                raise ValueError(f'TPU {key} changed')


def _spot_route(resource: dict) -> bool:
    # v2alpha1 canonicalizes the submitted spot tier to bestEffort + SPOT.
    # bestEffort alone does not establish Spot entitlement or quota usage.
    model = resource.get('provisioningModel')
    if model not in (None, 'PROVISIONING_MODEL_UNSPECIFIED', 'STANDARD',
                     'SPOT'):
        raise ValueError('Unsupported TPU provisioning model')
    if (('spot' in resource and model == 'STANDARD') or
        ('guaranteed' in resource and ('spot' in resource or model == 'SPOT'))):
        raise ValueError('Conflicting TPU route fields')
    return 'spot' in resource or model == 'SPOT'


def _resources():
    # Imported here because instance_utils calls this module during creation.
    # pylint: disable=import-outside-toplevel
    from sky.provision.gcp.instance_utils import GCPTPUVMInstance
    return GCPTPUVMInstance.load_resource().projects().locations(
    ).queuedResources()


def _operation_error(row: dict) -> Optional[dict]:
    if 'operation' not in row or row.get('create_time'):
        return None
    # Imported here for the same instance_utils import cycle as _resources.
    # pylint: disable=import-outside-toplevel
    from sky.provision.gcp.instance_utils import GCPTPUVMInstance
    operation = GCPTPUVMInstance.load_resource().projects().locations(
    ).operations().get(name=row['operation']['name']).execute(num_retries=2)
    return operation.get('error') if operation.get('done') else None


def _raise_terminal(name: str, details: dict) -> None:
    error = common.ProvisionerError(f'TPU queued resource {name}: {details}')
    error.errors = [
        dict(code=details.get('code', 'TIMEOUT'),
             domain='queued_resource',
             message=str(error))
    ]
    raise error


def check(path: pathlib.Path, *, raise_terminal: bool = False) -> bool:
    """Read all original resources; return True only when execution can resume.

    Provider uncertainty parks without teardown. A known FAILED state or a
    bounded timeout returns to the executor for normal native error/cleanup.
    """
    value = load(path)
    api = _resources()
    ready = True
    for name, row in value['resources'].items():
        if row['phase'] == 'prepared':
            return True  # The executor still has an unsent create to perform.
        try:
            operation_error = _operation_error(row)
            if operation_error:
                row['phase'] = 'rejected'
                save(path, value)
                if raise_terminal:
                    _raise_terminal(name, operation_error)
                return True
            resource = api.get(name=name).execute(num_retries=2)
            _validate(resource, name, row)
        except common.ProvisionerError:
            raise
        except Exception as error:  # pylint: disable=broad-except
            value['blocked'] = (f'Uncertain TPU request {name}: '
                                f'{type(error).__name__}; reconciliation '
                                'required if it persists')
            save(path, value)
            return False
        row['create_time'] = resource['createTime']
        row['phase'] = 'observed'
        state = resource.get('state', {}).get('state')
        if state == 'PROVISIONING' and 'provisioning_deadline' not in row:
            row['provisioning_deadline'] = time.time() + PROVISIONING_GRACE
        deadline = max(value['queue_deadline'],
                       row.get('provisioning_deadline', 0))
        if state == 'FAILED' or (state != 'ACTIVE' and time.time() >= deadline):
            save(path, value)
            if raise_terminal:
                details = resource.get('status', {})
                _raise_terminal(name, details or dict(message=state))
            return True
        ready = ready and state == 'ACTIVE'
    value.pop('blocked', None)
    save(path, value)
    return ready


def wait(path: pathlib.Path) -> list:
    try:
        ready = check(path, raise_terminal=True)
    except common.ProvisionerError:
        raise
    except Exception:  # pylint: disable=broad-except
        pause(path, 'TPU status unavailable; retaining original allocation')
    if not ready:
        value = load(path)
        pause(path, value.get('blocked', 'Waiting for Cloud TPU capacity'))
    value = load(path)
    value['stage'] = 'ready'
    save(path, value)
    return [
        spec['nodeId']
        for row in value['resources'].values()
        for spec in _specs(row['body'])
    ]


def create(path: pathlib.Path) -> list:
    """Send only unsent intents. Never repeat an uncertain create RPC."""
    value = load(path)
    try:
        api = _resources()
    except Exception:  # pylint: disable=broad-except
        pause(path, 'TPU API unavailable; retaining original allocation')
    for name, row in value['resources'].items():
        if row['phase'] != 'prepared':
            continue
        # Refuse even a coincidental pre-existing name before mutation.
        try:
            api.get(name=name).execute(num_retries=2)
        except gcp.http_error_exception() as error:
            if error.resp.status != 404:
                pause(path, 'TPU pre-create inventory unavailable')
        else:
            pause(path,
                  'TPU request name already exists; reconciliation required')
        row['phase'] = 'submitting'
        save(path, value)
        try:
            operation = api.create(
                parent=f'projects/{value["project"]}/locations/{value["zone"]}',
                queuedResourceId=name.rsplit('/', 1)[-1],
                body=row['body']).execute(num_retries=0)
        except Exception as error:  # pylint: disable=broad-except
            if (isinstance(error, gcp.http_error_exception()) and
                    error.resp.status in (400, 401, 403)):
                row['phase'] = 'rejected'
                save(path, value)
                raise  # A definitive rejection follows normal native teardown.
            pause(path,
                  'TPU create response uncertain; preserving original intent')
        row['operation'] = operation
        row['phase'] = 'submitted'
        save(path, value)
    value['stage'] = 'waiting'
    save(path, value)
    return wait(path)


class QueuedResourceCondition:
    """Poll off the executor, under the original request's environment.

    Only names/path cross the process boundary. Credentials stay in the server's
    request/config storage. The shared async waiter owns no per-request thread.
    """

    def __init__(self, path: str):
        self.path = path

    def _check(self) -> bool:
        return contextvars.Context().run(self._check_in_context)

    def _check_in_context(self) -> bool:
        context.initialize()
        # These server imports would form a cycle at module import time.
        # pylint: disable=import-outside-toplevel
        from sky.server.requests import executor
        from sky.server.requests import requests
        path = pathlib.Path(self.path)
        if not path.exists():
            # A conflicting request must retry its ownership fence.
            return True
        value = load(path)
        request = requests.get_request(value['request_id'])
        if (request is None or request.status > requests.RequestStatus.RUNNING):
            return False
        with executor.override_request_env_and_config(request.request_body,
                                                      request.request_id,
                                                      request.name):
            return check(path)

    def wait(self,
             *,
             is_cancelled,
             fallback_wait_seconds,
             update_status_msg=None) -> bool:
        del fallback_wait_seconds, update_status_msg
        while not is_cancelled():
            time.sleep(POLL_SECONDS)
            if self._check():
                return True
        return False

    async def wait_async(self,
                         *,
                         is_cancelled,
                         fallback_wait_seconds,
                         update_status_msg=None) -> bool:
        del fallback_wait_seconds, update_status_msg
        loop = asyncio.get_running_loop()
        while not await is_cancelled():
            await asyncio.sleep(POLL_SECONDS)
            if await is_cancelled():
                return False
            if await loop.run_in_executor(None, self._check):
                return not await is_cancelled()
        return False
