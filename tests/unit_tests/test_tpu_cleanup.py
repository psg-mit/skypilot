"""Terminal cleanup survives missing native rows, replies and restarts."""
# pylint: disable=protected-access,redefined-outer-name,unused-argument
import contextlib
import copy
import dataclasses
import hashlib
import pathlib
import threading
from types import SimpleNamespace
from unittest import mock

from googleapiclient import errors
import httplib2
import pytest
import yaml

from sky import backends
from sky import clouds
from sky import task as task_lib
from sky.backends import cloud_vm_ray_backend
from sky.provision import common
from sky.provision.gcp import instance_utils
from sky.serve import constants
from sky.serve import replica_managers
from sky.serve import tpu_cleanup as cleanup
from sky.server.requests import payloads
from sky.server.requests import requests
from sky.utils import common_utils
from sky.utils import locks


@pytest.fixture
def case(monkeypatch, tmp_path):
    monkeypatch.setenv('SKYPILOT_DURABLE_TPU_REQUESTS', '1')
    monkeypatch.setattr(cleanup.queue, 'JOURNAL_ROOT', tmp_path)
    monkeypatch.setattr(cleanup.filelock, 'FileLock',
                        lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(cleanup.locks, 'get_lock',
                        lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(cleanup.queue, 'allocation_lock',
                        lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(cleanup.serve_utils, 'get_service_filelock_path',
                        lambda _: '/unused')
    owner = cleanup.PoolAllocationOwner('logan-pool', 'pool-hash', 'owner',
                                        'default', 1, 3, 'logan-pool-3',
                                        'old-native-hash', 'original', 'digest')
    body = payloads.LaunchBody.model_construct(
        task='name: pool\nresources:\n  cpus: 1\n',
        cluster_name=owner.native_name,
        env_vars={},
        is_launched_by_sky_serve_controller=True,
        is_launched_by_jobs_controller=False)
    request = SimpleNamespace(request_id='original',
                              name='sky.launch',
                              cluster_name=owner.native_name,
                              user_id='owner',
                              status=requests.RequestStatus.SUCCEEDED,
                              request_body=body)
    monkeypatch.setattr(cleanup, 'launch_digest', lambda _: 'digest')
    monkeypatch.setattr(requests, 'get_request', lambda _: request)
    active = []
    monkeypatch.setattr(requests, 'get_request_tasks', lambda _: active)
    service = dict(hash='pool-hash', user_hash='owner', pool=True)
    monkeypatch.setattr(cleanup.serve_state, 'get_service_from_name',
                        lambda _: service)
    monkeypatch.setattr(cleanup.serve_state, 'get_replica_info_from_id',
                        lambda *_: None)
    monkeypatch.setattr(cleanup.global_user_state, 'get_cluster_from_name',
                        lambda _: None)
    jobs = {}
    monkeypatch.setattr(cleanup.job_state,
                        'get_nonterminal_job_ids_by_pool_grouped',
                        lambda _: jobs)
    parent = 'projects/project/locations/zone'
    name = parent + '/queuedResources/request'
    node_name = parent + '/nodes/node'
    spec = dict(parent=parent,
                nodeId='node',
                node=dict(acceleratorType='v6e-8',
                          runtimeVersion='runtime',
                          labels={
                              'ray-cluster-name': 'cloud',
                              'skypilot-user': 'owner'
                          }))
    resource = dict(name=name,
                    createTime='queue-generation',
                    tpu=dict(nodeSpec=[spec]),
                    state=dict(state='SUSPENDED'))
    resources = {name: resource}
    nodes = {}
    api = mock.MagicMock()

    def get(rows, key):

        def execute(**_):
            if key not in rows:
                raise errors.HttpError(httplib2.Response({'status': '404'}),
                                       b'{}')
            return copy.deepcopy(rows[key])

        return SimpleNamespace(execute=execute)

    api.queuedResources().get.side_effect = lambda name: get(resources, name)
    api.nodes().get.side_effect = lambda name: get(nodes, name)
    api.queuedResources().delete.return_value.execute.return_value = {
        'name': parent + '/operations/delete'
    }
    api.operations().get.return_value.execute.return_value = {'done': False}
    monkeypatch.setattr(
        instance_utils.GCPTPUVMInstance, 'load_resource',
        lambda: SimpleNamespace(projects=lambda: SimpleNamespace(locations=
                                                                 lambda: api)))
    path = tmp_path / 'original.json'
    value = dict(project='project',
                 zone='zone',
                 cluster='cloud',
                 native_cluster_hash=owner.native_hash,
                 request_id='original',
                 stage='ready',
                 pool_owner=dataclasses.asdict(owner),
                 resources={
                     name: dict(body=copy.deepcopy(resource),
                                create_time='queue-generation',
                                phase='observed')
                 })
    cleanup.queue.save(path, value)
    return SimpleNamespace(owner=owner,
                           body=body,
                           request=request,
                           active=active,
                           service=service,
                           jobs=jobs,
                           parent=parent,
                           name=name,
                           node_name=node_name,
                           spec=spec,
                           resource=resource,
                           resources=resources,
                           nodes=nodes,
                           api=api,
                           path=path)


def test_missing_native_row_deletes_exact_request_then_retires(case):
    assert cleanup.pending_native_allocation(case.owner.native_name)
    cleanup.reconcile_allocation(case.path)
    case.api.queuedResources().delete.assert_called_once_with(name=case.name,
                                                              force=True)
    first = cleanup.queue.load(case.path)
    assert first['stage'] == 'ready'
    assert first['resources'][case.name]['cleanup_operation']
    # A new process can resume the persisted operation, without another delete.
    cleanup.reconcile_allocation(case.path)
    assert case.api.queuedResources().delete.call_count == 1
    case.api.operations().get.return_value.execute.return_value = {'done': True}
    case.resources.clear()
    cleanup.reconcile_allocation(case.path)
    final = cleanup.queue.load(case.path)
    assert final['stage'] == 'retired'
    assert final['pool_owner'] == first['pool_owner']
    assert final['request_id'] == 'original'
    assert not cleanup.pending_native_allocation(case.owner.native_name)


@pytest.mark.parametrize('state',
                         ['ACTIVE', 'WAITING_FOR_RESOURCES', 'PROVISIONING'])
def test_nonterminal_request_never_deleted(case, state):
    case.resource['state']['state'] = state
    with pytest.raises(ValueError, match='not terminal'):
        cleanup.reconcile_allocation(case.path)
    case.api.queuedResources().delete.assert_not_called()


@pytest.mark.parametrize('change', [
    'queue_generation', 'labels', 'live_guest', 'pool_owner', 'assigned_job',
    'active_launch', 'unknown_create'
])
def test_changed_or_busy_allocation_is_preserved(case, change):
    if change == 'queue_generation':
        case.resource['createTime'] = 'replacement'
    elif change == 'labels':
        case.resource['tpu']['nodeSpec'][0]['node']['labels'][
            'skypilot-user'] = 'other'
    elif change == 'live_guest':
        case.nodes[case.node_name] = dict(state='READY',
                                          id='vm',
                                          createTime='new')
    elif change == 'pool_owner':
        case.service['user_hash'] = 'other'
    elif change == 'assigned_job':
        case.jobs[case.owner.native_name] = [123]
    elif change == 'active_launch':
        case.active.append(
            SimpleNamespace(user_id='owner',
                            pid=123,
                            status=requests.RequestStatus.RUNNING))
    else:
        value = cleanup.queue.load(case.path)
        value['resources'][case.name].pop('create_time')
        cleanup.queue.save(case.path, value)
    with pytest.raises(ValueError):
        cleanup.reconcile_allocation(case.path)
    case.api.queuedResources().delete.assert_not_called()
    assert cleanup.queue.load(case.path)['stage'] != 'retired'


def test_parked_equivalent_successor_is_preserved(case, monkeypatch):
    successor = SimpleNamespace(user_id='owner',
                                pid=None,
                                status=requests.RequestStatus.WAITING,
                                request_body=case.body)
    case.active.append(successor)
    monkeypatch.setattr(cleanup.pool_recovery, 'equivalent_worker_launches',
                        lambda *_: True)
    cleanup.reconcile_allocation(case.path)
    assert case.active == [successor]
    assert successor.status == requests.RequestStatus.WAITING
    assert case.api.queuedResources().delete.call_count == 1


def test_lost_delete_reply_requires_absence_before_retirement(case):
    case.api.queuedResources().delete.return_value.execute.side_effect = (
        TimeoutError('reply lost'))
    with pytest.raises(TimeoutError):
        cleanup.reconcile_allocation(case.path)
    with pytest.raises(ValueError, match='Uncertain deletion'):
        cleanup.reconcile_allocation(case.path)
    assert case.api.queuedResources().delete.call_count == 1
    case.resources.clear()
    cleanup.reconcile_allocation(case.path)
    assert cleanup.queue.load(case.path)['stage'] == 'retired'


def test_completed_operation_with_lagging_request_never_redeleted(case):
    cleanup.reconcile_allocation(case.path)
    case.api.operations().get.return_value.execute.return_value = {'done': True}
    with pytest.raises(ValueError, match='request remains'):
        cleanup.reconcile_allocation(case.path)
    assert case.api.queuedResources().delete.call_count == 1


def test_node_generation_change_between_audits_blocks_delete(case, monkeypatch):
    guest = dict(copy.deepcopy(case.spec['node']),
                 state='PREEMPTED',
                 id='original-vm',
                 createTime='original-boot')
    case.nodes[case.node_name] = guest
    native = cleanup._terminal_resources
    calls = []

    def change_after_first(value, api):
        result = native(value, api)
        calls.append(None)
        if len(calls) == 1:
            guest['id'] = 'new-vm'
        return result

    monkeypatch.setattr(cleanup, '_terminal_resources', change_after_first)
    with pytest.raises(ValueError, match='incarnation changed'):
        cleanup.reconcile_allocation(case.path)
    case.api.queuedResources().delete.assert_not_called()


def test_controller_rechecks_with_durable_error_and_backoff(case):
    case.jobs[case.owner.native_name] = [123]
    assert not cleanup.reconcile_pool('logan-pool')
    value = cleanup.queue.load(case.path)
    assert 'assigned work' in value['cleanup_error']
    assert value['cleanup_recheck_at'] > cleanup.time.time()
    case.api.queuedResources().delete.assert_not_called()


@pytest.mark.parametrize('status', [
    cleanup.job_state.ManagedJobStatus.PENDING,
    cleanup.job_state.ManagedJobStatus.STARTING,
    cleanup.job_state.ManagedJobStatus.RECOVERING,
])
def test_unassigned_waiting_jobs_do_not_block_terminal_cleanup(
        case, monkeypatch, status):
    case.jobs[None] = [123]
    monkeypatch.setattr(cleanup.job_state, 'get_status', lambda _: status)
    monkeypatch.setattr(cleanup.job_state, 'get_pool_submit_info', lambda _:
                        (None, None))
    cleanup.reconcile_allocation(case.path)
    case.api.queuedResources().delete.assert_called_once_with(name=case.name,
                                                              force=True)
    case.api.operations().get.return_value.execute.return_value = {'done': True}
    case.resources.clear()
    cleanup.reconcile_allocation(case.path)
    assert cleanup.queue.load(case.path)['stage'] == 'retired'
    assert case.jobs == {None: [123]}


@pytest.mark.parametrize('status,binding', [
    (cleanup.job_state.ManagedJobStatus.RUNNING, (None, None)),
    (cleanup.job_state.ManagedJobStatus.STARTING, (None, 42)),
    (cleanup.job_state.ManagedJobStatus.STARTING, ('successor', None)),
    (None, (None, None)),
])
def test_ambiguous_unassigned_job_still_blocks_cleanup(case, monkeypatch,
                                                       status, binding):
    case.jobs[None] = [123]
    monkeypatch.setattr(cleanup.job_state, 'get_status', lambda _: status)
    monkeypatch.setattr(cleanup.job_state, 'get_pool_submit_info',
                        lambda _: binding)
    with pytest.raises(ValueError, match='unresolved job assignment'):
        cleanup.reconcile_allocation(case.path)
    case.api.queuedResources().delete.assert_not_called()


@pytest.mark.parametrize('status', [
    cleanup.serve_state.ReplicaStatus.READY,
    cleanup.serve_state.ReplicaStatus.NOT_READY
])
def test_healthy_completed_launch_is_not_a_cleanup_candidate(
        case, monkeypatch, status):
    replica = SimpleNamespace(status=status)
    monkeypatch.setattr(cleanup.serve_state, 'get_replica_info_from_id',
                        lambda *_: replica)
    original = case.path.read_bytes()
    assert not cleanup.reconcile_pool('logan-pool')
    assert case.path.read_bytes() == original
    case.api.queuedResources().delete.assert_not_called()


def test_missing_cluster_row_cannot_complete_pool_teardown(case, monkeypatch):

    manager = object.__new__(replica_managers.SkyPilotReplicaManager)
    manager.lock = threading.RLock()
    manager._service_name = 'logan-pool'
    manager._launch_thread_pool = {}
    manager._down_thread_pool = {}
    manager._handle_sky_down_finish = mock.Mock()
    info = SimpleNamespace(cluster_name=case.owner.native_name,
                           status_property=SimpleNamespace(preempted=True))
    monkeypatch.setattr(cleanup.serve_state, 'get_replica_info_from_id',
                        lambda *_: info)
    monkeypatch.setattr(cleanup.global_user_state, 'cluster_with_name_exists',
                        lambda _: False)
    monkeypatch.setattr(cleanup.serve_utils, 'generate_replica_log_file_name',
                        lambda *_: '/unused')
    manager._terminate_replica(3, False, 0, is_scale_down=True)
    assert 'requires cleanup' in (
        manager._handle_sky_down_finish.call_args.kwargs['format_exc'])


def test_backend_preserves_native_state_until_allocation_retired(
        case, monkeypatch):
    monkeypatch.setattr(cleanup.global_user_state, 'get_cluster_hash',
                        lambda _: case.owner.native_hash)
    handle = SimpleNamespace(
        cluster_name=case.owner.native_name,
        cluster_name_on_cloud='cloud',
        launched_resources=SimpleNamespace(cloud=clouds.GCP()))
    backend = object.__new__(backends.CloudVmRayBackend)
    with pytest.raises(common.ProvisionerError, match='retaining native state'):
        backend.post_teardown_cleanup(handle, terminate=True)


def test_operator_enrollment_rejects_missing_original_history(
        case, monkeypatch):
    value = cleanup.queue.load(case.path)
    value.pop('pool_owner')
    cleanup.queue.save(case.path, value)
    digest = hashlib.sha256(case.path.read_bytes()).hexdigest()
    monkeypatch.setattr(cleanup.global_user_state, 'get_clusters_from_history',
                        lambda **_: [])
    with pytest.raises(ValueError, match='generation history'):
        cleanup.bind_retained_allocation(case.path, case.owner, digest)
    assert 'pool_owner' not in cleanup.queue.load(case.path)
    case.api.queuedResources().delete.assert_not_called()


def test_proven_rejected_create_can_retire_only_after_provider_absence(
        case, monkeypatch):
    case.resources.clear()
    value = cleanup.queue.load(case.path)
    row = value['resources'][case.name]
    row.pop('create_time')
    row['phase'] = 'rejected'
    cleanup.queue.save(case.path, value)
    monkeypatch.setattr(cleanup.queue, '_operation_error',
                        lambda _: {'code': 8})
    cleanup.reconcile_allocation(case.path)
    assert cleanup.queue.load(case.path)['stage'] == 'retired'
    case.api.queuedResources().delete.assert_not_called()


@pytest.mark.parametrize('code', [400, 401, 403, 429, 500])
def test_durable_http_rejection_needs_exact_absence(case, monkeypatch, code):
    case.resources.clear()
    value = cleanup.queue.load(case.path)
    row = value['resources'][case.name]
    row.pop('create_time')
    row['phase'] = 'rejected'
    row['create_error'] = dict(code=code, message='provider response')
    cleanup.queue.save(case.path, value)
    monkeypatch.setattr(cleanup.queue, '_operation_error', lambda _: None)
    if code == 500:
        with pytest.raises(ValueError, match='Uncertain TPU create'):
            cleanup.reconcile_allocation(case.path)
    else:
        cleanup.reconcile_allocation(case.path)
        assert cleanup.queue.load(case.path)['stage'] == 'retired'
    case.api.queuedResources().delete.assert_not_called()


@pytest.mark.parametrize('wire_format', ['task', 'dag', 'multiple_tasks'])
def test_audited_enrollment_preserves_original_receipt(case, monkeypatch,
                                                       wire_format):

    base = 'name: pool\nresources:\n  cpus: 1\n'
    task = task_lib.Task.from_yaml_str(base)
    task.update_envs({constants.REPLICA_ID_ENV_VAR: '3'})
    case.body.task = yaml.safe_dump(task.to_yaml_config())
    if wire_format != 'task':
        documents = [{'name': 'original-pool-launch'}, task.to_yaml_config()]
        if wire_format == 'multiple_tasks':
            documents.append(task.to_yaml_config())
        case.body.task = yaml.safe_dump_all(documents)
    monkeypatch.setattr(cleanup.serve_state, 'get_yaml_content',
                        lambda *_: base)
    monkeypatch.setattr(
        cleanup.global_user_state, 'get_clusters_from_history', lambda **_: [
            dict(cluster_hash=case.owner.native_hash,
                 name=case.owner.native_name,
                 user_hash='owner',
                 workspace='default')
        ])
    monkeypatch.setattr(
        cleanup.global_user_state, 'get_launch_attempts_for_cluster',
        lambda _: [
            SimpleNamespace(request_id='original',
                            cluster_name_on_cloud='cloud',
                            workspace='default')
        ])
    value = cleanup.queue.load(case.path)
    value.pop('pool_owner')
    cleanup.queue.save(case.path, value)
    digest = hashlib.sha256(case.path.read_bytes()).hexdigest()
    if wire_format == 'multiple_tasks':
        with pytest.raises(ValueError, match='exactly one task'):
            cleanup.bind_retained_allocation(case.path, case.owner, digest)
        assert 'pool_owner' not in cleanup.queue.load(case.path)
        case.api.queuedResources().delete.assert_not_called()
        return
    cleanup.bind_retained_allocation(case.path, case.owner, digest)
    enrolled = cleanup.queue.load(case.path)
    assert enrolled['pool_owner'] == dataclasses.asdict(case.owner)
    assert enrolled['resources'] == value['resources']
    assert enrolled['request_id'] == value['request_id']
    case.api.queuedResources().delete.assert_not_called()


def test_new_launch_persists_independent_cleanup_owner(case, monkeypatch):
    monkeypatch.setattr(common_utils, 'get_current_request_id',
                        lambda: 'original')
    monkeypatch.setattr(
        cleanup.global_user_state, 'get_cluster_from_name',
        lambda _: dict(cluster_hash=case.owner.native_hash,
                       user_hash='owner',
                       workspace='default'))
    monkeypatch.setattr(
        cleanup.serve_state, 'get_replica_info_from_id',
        lambda *_: SimpleNamespace(cluster_name=case.owner.native_name,
                                   version=1))
    assert cleanup.capture_owner(case.owner.native_hash) == dataclasses.asdict(
        case.owner)


@pytest.mark.parametrize('busy', [False, True])
def test_native_tpu_teardown_preserves_lock_inode_and_other_requests(
        monkeypatch, tmp_path, busy):
    monkeypatch.setenv('SKYPILOT_DURABLE_TPU_REQUESTS', '1')
    monkeypatch.setattr(locks, 'SKY_LOCKS_DIR', str(tmp_path))
    native_get_lock = locks.get_lock
    monkeypatch.setattr(
        locks, 'get_lock',
        lambda lock_id, **_: native_get_lock(lock_id, timeout=0))
    monkeypatch.setattr(cloud_vm_ray_backend.backend_utils,
                        'check_owner_identity', lambda _: None)
    cancel = mock.Mock()
    monkeypatch.setattr(requests, 'kill_cluster_requests', cancel)
    backend = object.__new__(backends.CloudVmRayBackend)
    backend.teardown_no_lock = mock.Mock()
    handle = SimpleNamespace(cluster_name='logan-pool-3',
                             launched_resources=SimpleNamespace(
                                 cloud=clouds.GCP(),
                                 accelerators={'tpu-v6e-8': 1},
                                 accelerator_args={'api': 'tpu'}))
    holder = native_get_lock('logan-pool-3_status', timeout=0)
    with holder if busy else contextlib.nullcontext():
        if busy:
            inode = pathlib.Path(holder.lock_path).stat().st_ino
            with pytest.raises(RuntimeError, match='lock remains owned'):
                backend._teardown(handle, terminate=True)
            assert pathlib.Path(holder.lock_path).stat().st_ino == inode
            backend.teardown_no_lock.assert_not_called()
        else:
            backend._teardown(handle, terminate=True)
            backend.teardown_no_lock.assert_called_once()
            assert (tmp_path / '.logan-pool-3_status.lock').exists()
    cancel.assert_not_called()
