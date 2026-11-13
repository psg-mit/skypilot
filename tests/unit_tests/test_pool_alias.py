"""Alias retirement preserves physical capacity and every original receipt."""
# pylint: disable=protected-access,redefined-outer-name,unused-argument
import contextlib
import copy
import dataclasses
import pickle
import threading
import types
from unittest import mock

import fastapi
from fastapi.testclient import TestClient
import filelock
import pytest
import sqlalchemy

from sky.serve import controller
from sky.serve import pool_alias
from sky.serve import replica_managers
from sky.serve import serve_state
from sky.serve import serve_utils
from sky.server.requests import payloads
from sky.server.requests import requests
from sky.utils import common_utils
from sky.utils import thread_utils

POOL = 'logan-test-pool'


@pytest.fixture
def case(monkeypatch, tmp_path):
    native_remove_cluster = pool_alias.global_user_state.remove_cluster
    native_remove_replica = serve_state.remove_replica
    monkeypatch.setattr(pool_alias, 'ROOT', tmp_path / 'retirements')
    monkeypatch.setattr(pool_alias.queue, 'JOURNAL_ROOT', tmp_path / 'queues')
    monkeypatch.setattr(common_utils, 'get_user_hash', lambda: 'owner')
    intent = pool_alias.AliasRetirement(
        pool_hash='pool-generation',
        owner='owner',
        workspace='default',
        version=1,
        alias_id=28,
        keeper_id=23,
        alias_hash='alias-generation',
        keeper_hash='keeper-generation',
        alias_request='alias-request',
        keeper_request='keeper-request',
        project='project',
        zone='zone',
        cloud_name='logan-cloud',
        node_name='projects/project/locations/zone/nodes/node',
        node_id='123',
        node_create_time='node-generation',
        queue_name='projects/project/locations/zone/queuedResources/queue',
        queue_create_time='queue-generation',
        reason='Retire exact duplicate binding')
    infos = {}
    clusters = {}
    launches = {}
    attempts = {}
    for rid, h, req in [(23, intent.keeper_hash, intent.keeper_request),
                        (28, intent.alias_hash, intent.alias_request),
                        (27, 'independent-generation', 'independent-request')]:
        name = f'{POOL}-{rid}'
        info = replica_managers.ReplicaInfo(rid, name, '-', True, None, 1, None)
        info.status_property.sky_launch_status = (
            common_utils.ProcessStatus.SUCCEEDED)
        info.status_property.service_ready_now = True
        infos[rid] = info
        clusters[name] = dict(
            cluster_hash=h,
            user_hash='owner',
            workspace='default',
            handle=types.SimpleNamespace(
                cluster_name_on_cloud='independent-cloud' if rid ==
                27 else intent.cloud_name))
        body = payloads.LaunchBody.model_construct(
            task='name: pool\nresources:\n  cpus: 1\n',
            cluster_name=name,
            env_vars={},
            is_launched_by_sky_serve_controller=True,
            is_launched_by_jobs_controller=False)
        launches[req] = types.SimpleNamespace(
            cluster_name=name,
            user_id='owner',
            status=requests.RequestStatus.SUCCEEDED,
            request_body=body)
        attempts[name] = [
            types.SimpleNamespace(request_id=req,
                                  cluster_hash=h,
                                  outcome='succeeded')
        ]
    service = dict(hash=intent.pool_hash,
                   user_hash='owner',
                   pool=True,
                   current_version=1)
    monkeypatch.setattr(serve_state, 'get_service_from_name', lambda _: service)
    monkeypatch.setattr(serve_state, 'get_replica_info_from_id',
                        lambda _, rid: infos.get(rid))
    monkeypatch.setattr(serve_state, 'get_replica_infos',
                        lambda _: list(infos.values()))
    monkeypatch.setattr(serve_state, 'remove_replica',
                        lambda _, rid: infos.pop(rid, None))
    monkeypatch.setattr(pool_alias.global_user_state, 'get_cluster_from_name',
                        clusters.get)
    monkeypatch.setattr(pool_alias.global_user_state, 'remove_cluster',
                        lambda name, terminate: clusters.pop(name))
    monkeypatch.setattr(pool_alias.global_user_state,
                        'get_launch_attempts_for_cluster', attempts.get)
    monkeypatch.setattr(requests, 'get_request', launches.get)
    active = []
    monkeypatch.setattr(requests, 'get_request_tasks', lambda _: active)
    jobs = []
    monkeypatch.setattr(pool_alias.job_state, 'get_nonterminal_job_ids_by_pool',
                        lambda _: jobs)
    body = dict(tpu=dict(nodeSpec=[
        dict(parent='projects/project/locations/zone',
             nodeId='node',
             node=dict(acceleratorType='v5litepod-16',
                       runtimeVersion='runtime',
                       labels={'ray-cluster-name': 'logan-cloud'}))
    ]))
    resource = dict(**copy.deepcopy(body),
                    name=intent.queue_name,
                    createTime=intent.queue_create_time,
                    state=dict(state='ACTIVE'))
    journal = dict(project=intent.project,
                   zone=intent.zone,
                   cluster=intent.cloud_name,
                   stage='ready',
                   request_id=intent.keeper_request,
                   native_cluster_hash=intent.keeper_hash,
                   resources={
                       intent.queue_name: dict(
                           body=body,
                           phase='observed',
                           create_time=intent.queue_create_time)
                   })
    journal_path = tmp_path / 'queues' / 'original.json'
    pool_alias.queue.save(journal_path, journal)
    node = dict(name=intent.node_name,
                id=intent.node_id,
                createTime=intent.node_create_time,
                state='READY',
                queuedResource=intent.queue_name,
                labels={'ray-cluster-name': intent.cloud_name})
    api = mock.MagicMock()
    api.queuedResources().get().execute.side_effect = lambda **_: copy.deepcopy(
        resource)
    api.nodes().get().execute.side_effect = lambda **_: copy.deepcopy(node)
    monkeypatch.setattr(
        pool_alias.instance_utils.GCPTPUVMInstance, 'load_resource',
        lambda: mock.Mock(projects=lambda: mock.Mock(locations=lambda: api)))
    manager = object.__new__(replica_managers.SkyPilotReplicaManager)
    manager.lock = threading.Lock()
    manager._service_name = POOL
    manager._is_pool = True
    manager._next_replica_id = 29
    for attr in ('_launch_thread_pool', '_down_thread_pool',
                 '_replica_to_request_id', '_replica_to_launch_cancelled'):
        setattr(manager, attr, thread_utils.ThreadSafeDict())
    manager._last_termination_check = {28: 1}
    monkeypatch.setattr(serve_utils, 'get_service_filelock_path',
                        lambda _: str(tmp_path / 'pool.lock'))
    monkeypatch.setattr(replica_managers.locks, 'get_lock',
                        lambda *_: contextlib.nullcontext())
    return types.SimpleNamespace(intent=intent,
                                 infos=infos,
                                 clusters=clusters,
                                 launches=launches,
                                 attempts=attempts,
                                 service=service,
                                 active=active,
                                 jobs=jobs,
                                 resource=resource,
                                 journal=journal,
                                 node=node,
                                 api=api,
                                 manager=manager,
                                 tmp_path=tmp_path,
                                 journal_path=journal_path,
                                 native_remove_cluster=native_remove_cluster,
                                 native_remove_replica=native_remove_replica)


def test_exact_alias_retirement_preserves_keeper_independent_and_history(case):
    snapshots = pickle.dumps(
        (case.infos[23], case.infos[27], case.launches, case.attempts))
    journal_before = case.journal_path.read_bytes()
    result = case.manager.retire_replica_alias(case.intent)
    assert result['stage'] == 'committed'
    assert pickle.loads(bytes.fromhex(result['alias_record'])).replica_id == 28
    assert set(case.infos) == {23, 27}
    assert set(case.clusters) == {f'{POOL}-23', f'{POOL}-27'}
    assert pickle.dumps((case.infos[23], case.infos[27], case.launches,
                         case.attempts)) == snapshots
    assert case.journal_path.read_bytes() == journal_before
    assert not case.api.nodes().delete.called
    assert not case.api.queuedResources().delete.called
    assert pool_alias.retired_replica_ids(POOL) == [28]
    assert case.manager._next_replica_id == 29


def test_native_sqlite_removal_keeps_both_allocations_and_cluster_history(
        case, monkeypatch):
    state = pool_alias.global_user_state
    native = sqlalchemy.create_engine('sqlite:///' +
                                      str(case.tmp_path / 'native.db'))
    services = sqlalchemy.create_engine('sqlite:///' +
                                        str(case.tmp_path / 'serve.db'))
    state.cluster_table.create(native)
    state.cluster_history_table.create(native)
    serve_state.replicas_table.create(services)
    with native.begin() as connection:
        for name, cluster in case.clusters.items():
            connection.execute(state.cluster_table.insert().values(
                name=name, cluster_hash=cluster['cluster_hash']))
            connection.execute(state.cluster_history_table.insert().values(
                name=name, cluster_hash=cluster['cluster_hash']))
    with services.begin() as connection:
        for replica_id, info in case.infos.items():
            connection.execute(serve_state.replicas_table.insert().values(
                service_name=POOL,
                replica_id=replica_id,
                replica_info=pickle.dumps(info)))
    monkeypatch.setattr(state, '_db_manager',
                        types.SimpleNamespace(get_engine=lambda: native))
    monkeypatch.setattr(serve_state, '_db_manager',
                        types.SimpleNamespace(get_engine=lambda: services))
    monkeypatch.setattr(state, '_get_cluster_usage_intervals', lambda _: [])
    monkeypatch.setattr(state, 'get_cluster_provision_log_path', lambda _: None)
    monkeypatch.setattr(state, 'remove_cluster', case.native_remove_cluster)
    monkeypatch.setattr(serve_state, 'remove_replica',
                        case.native_remove_replica)
    with native.connect() as connection:
        history_before = connection.execute(
            sqlalchemy.select(state.cluster_history_table)).all()
    case.manager.retire_replica_alias(case.intent)
    with native.connect() as connection:
        assert set(
            connection.execute(sqlalchemy.select(
                state.cluster_table.c.name)).scalars()) == {
                    f'{POOL}-23', f'{POOL}-27'
                }
        assert connection.execute(sqlalchemy.select(
            state.cluster_history_table)).all() == history_before
    with services.connect() as connection:
        assert set(
            connection.execute(
                sqlalchemy.select(
                    serve_state.replicas_table.c.replica_id)).scalars()) == {
                        23, 27
                    }


@pytest.mark.parametrize('field', [
    'node_id', 'node_create_time', 'queue_create_time', 'keeper_request',
    'alias_request', 'keeper_hash', 'alias_hash', 'owner', 'workspace',
    'pool_hash', 'version'
])
def test_changed_expected_binding_refuses_before_any_metadata_change(
        case, field):
    value = 2 if field == 'version' else 'changed'
    intent = dataclasses.replace(case.intent, **{field: value})
    before = pickle.dumps((case.infos, case.clusters))
    with pytest.raises(ValueError):
        case.manager.retire_replica_alias(intent)
    assert pickle.dumps((case.infos, case.clusters)) == before
    assert not list(pool_alias.ROOT.glob('*.json'))


@pytest.mark.parametrize('busy', ['job', 'api', 'launch', 'down'])
def test_busy_binding_is_preserved(case, busy):
    if busy == 'job':
        case.jobs.append(2400)
    elif busy == 'api':
        case.active.append(mock.Mock())
    else:
        getattr(case.manager, '_' + busy + '_thread_pool')[28] = mock.Mock()
    with pytest.raises(ValueError):
        case.manager.retire_replica_alias(case.intent)
    assert set(case.infos) == {23, 27, 28}
    assert not list(pool_alias.ROOT.glob('*.json'))


def test_different_immutable_source_or_launch_options_are_refused(case):
    body = case.launches[case.intent.alias_request].request_body
    body.no_setup = True
    with pytest.raises(ValueError, match='immutable launch inputs'):
        case.manager.retire_replica_alias(case.intent)
    assert set(case.infos) == {23, 27, 28}


def test_expected_replica_environment_difference_is_permitted(case):
    for rid, request_id in ((23, case.intent.keeper_request),
                            (28, case.intent.alias_request)):
        body = case.launches[request_id].request_body
        body.task += (
            'envs:\n  ' +
            pool_alias.pool_recovery.serve_constants.REPLICA_ID_ENV_VAR +
            ': "' + str(rid) + '"\n')
    assert case.manager.retire_replica_alias(
        case.intent)['stage'] == 'committed'


def test_lost_receipt_replay_does_not_touch_new_jobs_on_keeper(case):
    first = case.manager.retire_replica_alias(case.intent)
    calls = list(case.api.mock_calls)
    case.jobs.append(2400)
    assert case.manager.retire_replica_alias(case.intent) == first
    assert case.api.mock_calls == calls
    with pytest.raises(ValueError, match='intent changed'):
        case.manager.retire_replica_alias(
            dataclasses.replace(case.intent, reason='changed'))


@pytest.mark.parametrize('fail_at',
                         ['cluster_remove', 'replica_remove', 'commit'])
def test_partial_operation_stays_quarantined_and_resumes(
        case, monkeypatch, fail_at):
    if fail_at == 'commit':
        original = pool_alias.queue.save

        def fail(path, value):
            if value['stage'] == 'committed':
                raise OSError('lost receipt')
            original(path, value)

        monkeypatch.setattr(pool_alias.queue, 'save', fail)
    else:
        module = (pool_alias.global_user_state
                  if fail_at == 'cluster_remove' else serve_state)
        name = ('remove_cluster'
                if fail_at == 'cluster_remove' else 'remove_replica')
        original = getattr(module, name)
        monkeypatch.setattr(module, name,
                            mock.Mock(side_effect=OSError('interrupted')))
    with pytest.raises(OSError):
        case.manager.retire_replica_alias(case.intent)
    assert pool_alias.pending_intents(POOL) == [case.intent]
    assert {r.replica_id for r in serve_utils.get_ready_replicas(POOL)
           } == {23, 27}
    if fail_at == 'commit':
        monkeypatch.setattr(pool_alias.queue, 'save', original)
    else:
        monkeypatch.setattr(module, name, original)
    assert case.manager.retire_replica_alias(
        case.intent)['stage'] == 'committed'
    assert set(case.infos) == {23, 27}


def test_pool_scheduling_lock_covers_metadata_transaction(case, monkeypatch):
    original = pool_alias.retire

    def checked(*args):
        with pytest.raises(filelock.Timeout):
            filelock.FileLock(str(case.tmp_path /
                                  'pool.lock')).acquire(timeout=0)
        assert case.manager.lock.locked()
        return original(*args)

    monkeypatch.setattr(pool_alias, 'retire', checked)
    case.manager.retire_replica_alias(case.intent)


def test_controller_restart_reserves_retired_high_water_id(case, monkeypatch):
    case.manager.retire_replica_alias(case.intent)
    case.manager._next_replica_id = 1
    monkeypatch.setattr(serve_state, 'get_replicas_at_status', lambda *_: [])
    case.manager._recover_replica_operations()
    assert case.manager._next_replica_id == 29


def test_partial_recovery_refuses_changed_provider_and_keeps_quarantine(
        case, monkeypatch):
    remove = serve_state.remove_replica
    monkeypatch.setattr(serve_state, 'remove_replica',
                        mock.Mock(side_effect=OSError()))
    with pytest.raises(OSError):
        case.manager.retire_replica_alias(case.intent)
    monkeypatch.setattr(serve_state, 'remove_replica', remove)
    case.node['id'] = 'replacement'
    with pytest.raises(ValueError, match='generation'):
        case.manager.retire_replica_alias(case.intent)
    assert pool_alias.pending_intents(POOL) == [case.intent]
    assert 28 not in {
        r.replica_id for r in serve_utils.get_ready_replicas(POOL)
    }


def test_controller_endpoint_rejects_unknown_fields_and_returns_durable_receipt(
        case, monkeypatch):
    app_controller = object.__new__(controller.SkyServeController)
    app_controller._service_name = POOL
    app_controller._replica_manager = case.manager
    app_controller._app = fastapi.FastAPI()
    app_controller._host = '127.0.0.1'
    app_controller._port = 0
    with monkeypatch.context() as patch:
        patch.setattr(controller.threading.Thread, 'start', lambda _: None)
        patch.setattr(controller.uvicorn, 'run', lambda *_a, **_kw: None)
        app_controller.run()
    client = TestClient(app_controller._app)
    payload = dataclasses.asdict(case.intent)
    endpoint = '/controller/retire_replica_alias'
    assert client.post(endpoint, json=dict(**payload,
                                           purge=True)).status_code == 400
    first = client.post(endpoint, json=payload)
    assert first.status_code == 200
    assert first.json()['stage'] == 'committed'
    assert client.post(endpoint, json=payload).json() == first.json()
