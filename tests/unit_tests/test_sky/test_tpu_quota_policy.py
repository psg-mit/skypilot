"""Tests for sky.policies.tpu_quota."""
# pylint: disable=redefined-outer-name
import contextlib
import json
from unittest import mock

import pytest

import sky
from sky import admin_policy
from sky import exceptions
from sky import models
from sky import resources as resources_lib
from sky.backends import backend_utils
from sky.clouds.gcp import GCP
from sky.jobs import state as managed_job_state
from sky.policies import tpu_quota
from sky.server.requests import request_names
from sky.server.requests import requests as api_requests
from sky.utils import admin_policy_utils
from sky.utils import common_utils
from sky.utils import config_utils
from sky.utils import locks
from sky.utils import status_lib

_LIMITS = {
    'unit': 'chips',
    'pool': {
        'v5p': 12,
        'v6e': 8
    },
    'per_user': {
        'default': {
            'v5p': 4
        },
        'alice': {
            'v5p': 8
        },
    },
}


def _cluster(user: str,
             accelerators: dict,
             num_nodes: int = 1,
             status=status_lib.ClusterStatus.UP,
             name: str = 'cluster') -> dict:
    handle = mock.MagicMock()
    handle.launched_resources.accelerators = accelerators
    return {
        'name': name,
        'user_name': user,
        'status': status,
        'num_nodes': num_nodes,
        'handle': handle,
    }


def _job(user: str,
         resources: str,
         status=managed_job_state.ManagedJobStatus.PENDING,
         name: str = 'job',
         pool: str = None,
         cluster_name: str = None) -> dict:
    return {
        'job_name': name,
        'user_name': user,
        'status': status,
        'resources': resources,
        'pool': pool,
        'current_cluster_name': cluster_name,
    }


def _request(user: str,
             accelerators,
             num_nodes: int = 1,
             request_name=request_names.AdminPolicyRequestName.CLUSTER_LAUNCH,
             at_client_side: bool = False,
             cluster_name=None,
             **resources_kwargs) -> admin_policy.UserRequest:
    if isinstance(accelerators, list):
        resources = [
            resources_lib.Resources(cloud=GCP(), accelerators=acc)
            for acc in accelerators
        ]
    else:
        resources = resources_lib.Resources(cloud=GCP(),
                                            accelerators=accelerators,
                                            **resources_kwargs)
    task = sky.Task(num_nodes=num_nodes).set_resources(resources)
    request_options = None
    if cluster_name is not None:
        request_options = admin_policy.RequestOptions(
            cluster_name=cluster_name,
            idle_minutes_to_autostop=None,
            down=False,
            dryrun=False)
    return admin_policy.UserRequest(task=task,
                                    skypilot_config=config_utils.Config(),
                                    request_name=request_name,
                                    request_options=request_options,
                                    at_client_side=at_client_side,
                                    user=models.User(id='h', name=user))


@pytest.fixture
def quota_file(tmp_path, monkeypatch):
    path = tmp_path / 'tpu_quota.yaml'
    monkeypatch.setenv(tpu_quota.QUOTA_FILE_ENV_VAR, str(path))

    def write(limits):
        import yaml  # pylint: disable=import-outside-toplevel
        path.write_text(yaml.safe_dump(limits))
        return path

    return write


class _FakeLock:
    """Records acquisitions; never blocks."""

    def __init__(self):
        self.held = False
        self.acquisitions = 0

    def __enter__(self):
        assert not self.held, 'the admission lock is not reentrant'
        self.held = True
        self.acquisitions += 1
        return self

    def __exit__(self, *exc):
        self.held = False


@pytest.fixture
def usage(monkeypatch):
    """Sets the clusters, jobs and requests the policy sees.

    ``admitted`` collects the reservations the policy records, each tagged
    with whether the admission lock was held at the time.
    """
    state = {
        'clusters': [],
        'jobs': [],
        'parked': [],
        'running': [],
        'admitted': [],
        'lock': _FakeLock(),
    }
    monkeypatch.setattr(tpu_quota, '_cluster_records',
                        lambda: state['clusters'])
    monkeypatch.setattr(tpu_quota, '_job_records', lambda: state['jobs'])
    monkeypatch.setattr(tpu_quota, '_parked_requests', lambda: state['parked'])
    monkeypatch.setattr(tpu_quota, '_running_requests',
                        lambda: state['running'])
    monkeypatch.setattr(tpu_quota, '_admission_lock', lambda: state['lock'])

    def record(user, unit, demand, cluster_name, job_name):
        state['admitted'].append({
            'user': user,
            'unit': unit,
            'demand': demand,
            'cluster': cluster_name,
            'job': job_name,
            'locked': state['lock'].held,
        })

    monkeypatch.setattr(tpu_quota, '_record_admission', record)
    return state


def _parked(user_id: str, status_msg: str):
    request = mock.MagicMock()
    request.request_id = f'req-{user_id}-{len(status_msg)}'
    request.user_id = user_id
    request.status_msg = status_msg
    return request


def _running(request_id: str,
             user: str = None,
             demand: dict = None,
             cluster: str = None,
             job: str = None,
             unit: str = 'chips',
             status_msg: str = None):
    """A RUNNING launch request, admitted unless ``status_msg`` is given."""
    request = mock.MagicMock()
    request.request_id = request_id
    if status_msg is None:
        reservation = {
            'user': user,
            'unit': unit,
            'demand': demand,
            'cluster': cluster,
            'job': job,
        }
        status_msg = f'[tpu-quota admitted] {json.dumps(reservation)}'
    request.status_msg = status_msg
    return request


@pytest.mark.parametrize('acc_name, family, chips', [
    ('tpu-v5p-8', 'v5p', 4),
    ('tpu-v5p-16', 'v5p', 8),
    ('tpu-v4-8', 'v4', 4),
    ('tpu-v2-8', 'v2', 4),
    ('tpu-v5litepod-4', 'v5litepod', 4),
    ('tpu-v6e-1', 'v6e', 1),
    ('tpu-v6e-8', 'v6e', 8),
])
def test_family_and_chips(acc_name, family, chips):
    assert tpu_quota.tpu_family(acc_name) == family
    assert tpu_quota.tpu_amount(acc_name, 'chips') == chips
    assert tpu_quota.tpu_amount(acc_name, 'slices') == 1


def test_family_of_non_tpu_is_none():
    assert tpu_quota.tpu_family('H100') is None
    assert tpu_quota.tpu_family('tpu-v5-lite-podslice') is None


def test_quota_user_limit_fallbacks():
    quota = tpu_quota.Quota.from_dict(_LIMITS)
    assert quota.user_limit('alice', 'v5p') == 8
    assert quota.user_limit('bob', 'v5p') == 4
    # Not in the user's limits nor in default: the pool is the limit.
    assert quota.user_limit('alice', 'v6e') == 8


def test_quota_rejects_unknown_unit():
    with pytest.raises(ValueError, match='unit'):
        tpu_quota.Quota.from_dict({'unit': 'cores', 'pool': {}})


def test_load_quota_fails_open(quota_file):
    assert tpu_quota.load_quota() is None  # missing file
    path = quota_file(_LIMITS)
    assert tpu_quota.load_quota().pool == {'v5p': 12, 'v6e': 8}
    path.write_text('unit: [')
    assert tpu_quota.load_quota() is None  # malformed file


def test_task_demand_counts_nodes_and_largest_candidate():
    req = _request('bob', 'tpu-v5p-8', num_nodes=2)
    assert tpu_quota.task_demand(req.task, 'chips') == {'v5p': 8}
    req = _request('bob', ['tpu-v5p-8', 'tpu-v5p-16', 'tpu-v6e-4'])
    assert tpu_quota.task_demand(req.task, 'chips') == {'v5p': 8, 'v6e': 4}
    req = _request('bob', {'H100': 8})
    assert not tpu_quota.task_demand(req.task, 'chips')


def test_current_usage_from_clusters_and_jobs(usage):
    usage['clusters'] = [
        _cluster('alice', {'tpu-v5p-8': 1}, num_nodes=2),
        _cluster('bob', {'tpu-v6e-4': 1}),
        _cluster('bob', {'tpu-v5p-8': 1},
                 status=status_lib.ClusterStatus.STOPPED),
        _cluster('carol', {'H100': 8}),
        # The Compute Engine API path records the same accelerator name.
        _cluster('carol', {'tpu-v5p-8': 1}),
    ]
    usage['jobs'] = [
        _job('bob', '1x[tpu-v5p-8:1][Spot]'),
        _job('bob',
             '2x[tpu-v6e-1:1]',
             status=managed_job_state.ManagedJobStatus.RUNNING),
        _job('alice',
             '1x[tpu-v5p-8:1]',
             status=managed_job_state.ManagedJobStatus.SUCCEEDED),
        _job('alice', '1x[CPU:2+]'),
    ]
    by_user, total = tpu_quota.current_usage('chips')
    assert by_user == {
        'alice': {
            'v5p': 8
        },
        'bob': {
            'v6e': 6,
            'v5p': 4
        },
        'carol': {
            'v5p': 4
        },
    }
    assert total == {'v5p': 16, 'v6e': 6}


def test_current_usage_reads_accelerators_string(usage):
    usage['clusters'] = [{
        'user_name': 'bob',
        'status': status_lib.ClusterStatus.UP,
        'num_nodes': 1,
        'accelerators': '{\'tpu-v6e-4\': 1}',
    }]
    assert tpu_quota.current_usage('chips')[1] == {'v6e': 4}


def _apply(req):
    return tpu_quota.TPUQuotaPolicy.validate_and_mutate(req)


def test_admits_within_share(quota_file, usage):
    quota_file(_LIMITS)
    usage['clusters'] = [_cluster('alice', {'tpu-v5p-8': 1})]
    result = _apply(_request('alice', 'tpu-v5p-8', cluster_name='c2'))
    assert result.task is not None
    # The admission is recorded under the lock, keyed by the cluster name.
    assert usage['admitted'] == [{
        'user': 'alice',
        'unit': 'chips',
        'demand': {
            'v5p': 4
        },
        'cluster': 'c2',
        'job': None,
        'locked': True,
    }]
    assert usage['lock'].acquisitions == 1


def test_admits_borrowing_while_pool_has_room(quota_file, usage):
    quota_file(_LIMITS)
    # bob is at his share of 4; the pool (12) has 8 left.
    usage['clusters'] = [_cluster('bob', {'tpu-v5p-8': 1})]
    _apply(_request('bob', 'tpu-v5p-8'))


def test_parks_when_over_share_and_pool_full(quota_file, usage):
    quota_file(_LIMITS)
    usage['clusters'] = [
        _cluster('alice', {'tpu-v5p-8': 1}, num_nodes=2),
        _cluster('bob', {'tpu-v5p-8': 1}),
    ]
    with pytest.raises(exceptions.ExecutionPausedError) as exc_info:
        _apply(_request('bob', 'tpu-v5p-8'))
    assert exc_info.value.retry_wait_seconds == tpu_quota.RETRY_WAIT_SECONDS
    message = str(exc_info.value)
    assert message.startswith('[tpu-quota v5p over-share]')
    assert 'bob is at 4/4 v5p chips' in message
    # The pool is a hard cap: carol is within her share (0 of 4) but the pool
    # has no room, so she is parked too, with a marker that says she is
    # within her share.
    with pytest.raises(exceptions.ExecutionPausedError) as exc_info:
        _apply(_request('carol', 'tpu-v5p-8'))
    assert str(exc_info.value).startswith('[tpu-quota v5p under-share]')


def test_within_share_but_pool_full_parks(quota_file, usage):
    quota_file({'unit': 'chips', 'pool': {'v5p': 4}, 'per_user': {}})
    usage['clusters'] = [_cluster('alice', {'tpu-v5p-8': 1})]
    with pytest.raises(exceptions.ExecutionPausedError):
        _apply(_request('bob', 'tpu-v5p-8'))


def test_borrower_yields_to_parked_user_within_share(quota_file, usage):
    quota_file(_LIMITS)
    # bob is at his share; the pool has room, but carol is parked within
    # her share for the same family, so bob yields.
    usage['clusters'] = [_cluster('bob', {'tpu-v5p-8': 1})]
    usage['parked'] = [
        _parked(
            'carol-id', '[tpu-quota v5p under-share] carol is at 0/4 '
            'v5p chips and the pool is full (12/12). (waiting to resume)'),
    ]
    with pytest.raises(exceptions.ExecutionPausedError) as exc_info:
        _apply(_request('bob', 'tpu-v5p-8'))
    assert 'Yielding' in str(exc_info.value)
    # A parked request of another family, of another borrower, or of bob
    # himself does not make bob yield.
    usage['parked'] = [
        _parked('carol-id', '[tpu-quota v6e under-share] ...'),
        _parked('dave-id', '[tpu-quota v5p over-share] ...'),
        _parked('h', '[tpu-quota v5p under-share] ...'),
        _parked('erin-id', 'Waiting for the cluster lock (retrying in 10s)'),
    ]
    _apply(_request('bob', 'tpu-v5p-8'))
    # A user within their share never yields.
    usage['parked'] = [_parked('carol-id', '[tpu-quota v5p under-share] ...')]
    _apply(_request('alice', 'tpu-v5p-8'))


def test_under_share_waiters_parses_markers(usage):
    usage['parked'] = [
        _parked('a', '[tpu-quota v5p under-share] x'),
        _parked('b', '[tpu-quota v5p over-share] x'),
        _parked('c', 'no marker'),
        _parked('me', '[tpu-quota v5p under-share] x'),
    ]
    assert tpu_quota.under_share_waiters('v5p', 'me') == ['req-a-29']
    assert not tpu_quota.under_share_waiters('v6e', 'me')


@pytest.mark.usefixtures('usage')
def test_rejects_request_larger_than_pool(quota_file):
    quota_file(_LIMITS)
    with pytest.raises(ValueError, match='never be admitted'):
        _apply(_request('alice', 'tpu-v5p-32'))


def test_ungoverned_requests_pass(quota_file, usage):
    quota_file(_LIMITS)
    usage['clusters'] = [_cluster('alice', {'tpu-v5p-8': 1}, num_nodes=3)]
    _apply(_request('alice', {'H100': 8}))
    _apply(_request('alice', 'tpu-v5litepod-8'))  # family not in pool
    _apply(_request('alice', 'tpu-v5p-8', at_client_side=True))
    _apply(
        _request(
            'alice',
            'tpu-v5p-8',
            request_name=request_names.AdminPolicyRequestName.CLUSTER_EXEC))


def test_missing_quota_file_admits(usage):
    usage['clusters'] = [_cluster('alice', {'tpu-v5p-8': 1}, num_nodes=10)]
    _apply(_request('alice', 'tpu-v5p-8'))


def test_jobs_launch_is_gated(quota_file, usage):
    quota_file(_LIMITS)
    usage['jobs'] = [
        _job('alice',
             '3x[tpu-v5p-8:1]',
             status=managed_job_state.ManagedJobStatus.RUNNING)
    ]
    with pytest.raises(exceptions.ExecutionPausedError):
        _apply(
            _request(
                'bob',
                'tpu-v5p-8',
                request_name=request_names.AdminPolicyRequestName.JOBS_LAUNCH))


def test_paused_error_propagates_through_admin_policy_utils(
        quota_file, usage, monkeypatch):
    quota_file(_LIMITS)
    usage['clusters'] = [_cluster('alice', {'tpu-v5p-8': 1}, num_nodes=3)]
    config = config_utils.Config.from_dict(
        {'admin_policy': 'sky.policies.tpu_quota.TPUQuotaPolicy'})
    monkeypatch.setattr(sky.skypilot_config, '_get_loaded_config',
                        lambda *a, **k: config)
    monkeypatch.setattr(admin_policy_utils.common_utils, 'get_current_user',
                        lambda: models.User(id='h', name='bob'))
    task = sky.Task().set_resources(
        resources_lib.Resources(cloud=GCP(), accelerators='tpu-v5p-8'))
    with pytest.raises(exceptions.ExecutionPausedError):
        admin_policy_utils.apply(
            task, request_names.AdminPolicyRequestName.CLUSTER_LAUNCH)
    # Other policy failures are still wrapped as rejections.
    task = sky.Task().set_resources(
        resources_lib.Resources(cloud=GCP(), accelerators='tpu-v5p-32'))
    with pytest.raises(exceptions.UserRequestRejectedByPolicy):
        admin_policy_utils.apply(
            task, request_names.AdminPolicyRequestName.CLUSTER_LAUNCH)


def test_parked_request_records_no_admission(quota_file, usage):
    quota_file({'unit': 'chips', 'pool': {'v5p': 4}, 'per_user': {}})
    usage['clusters'] = [_cluster('alice', {'tpu-v5p-8': 1})]
    with pytest.raises(exceptions.ExecutionPausedError):
        _apply(_request('bob', 'tpu-v5p-8'))
    assert usage['admitted'] == []
    assert usage['lock'].acquisitions == 1
    assert not usage['lock'].held


def test_passthrough_skips_the_lock(quota_file, usage):
    quota_file(_LIMITS)
    _apply(_request('alice', {'H100': 8}))
    _apply(_request('alice', 'tpu-v5p-8', at_client_side=True))
    assert usage['lock'].acquisitions == 0


@pytest.mark.usefixtures('usage')
def test_lock_timeout_parks_without_family_marker(quota_file, monkeypatch):
    quota_file(_LIMITS)

    @contextlib.contextmanager
    def timeout():
        raise locks.LockTimeout('busy')
        yield  # pylint: disable=unreachable

    monkeypatch.setattr(tpu_quota, '_admission_lock', timeout)
    with pytest.raises(exceptions.ExecutionPausedError) as exc_info:
        _apply(_request('alice', 'tpu-v5p-8'))
    assert 'admission lock' in str(exc_info.value)
    # No family marker, so borrowers do not yield to this request.
    assert '-share]' not in str(exc_info.value)


def test_admitted_but_invisible_request_counts_as_usage(quota_file, usage):
    """Two requests admitted back to back cannot jointly exceed the pool."""
    quota_file({'unit': 'chips', 'pool': {'v5p': 8}, 'per_user': {}})
    # alice's request was admitted a moment ago; her cluster is not in the
    # cluster table yet.
    usage['running'] = [
        _running('req-a', user='alice', demand={'v5p': 8}, cluster='a')
    ]
    assert tpu_quota.current_usage('chips') == ({
        'alice': {
            'v5p': 8
        }
    }, {
        'v5p': 8
    })
    with pytest.raises(exceptions.ExecutionPausedError) as exc_info:
        _apply(_request('bob', 'tpu-v5p-8'))
    assert 'pool is full (8/8)' in str(exc_info.value)


def test_reservation_dropped_once_cluster_is_visible(quota_file, usage):
    quota_file({'unit': 'chips', 'pool': {'v5p': 8}, 'per_user': {}})
    usage['running'] = [
        _running('req-a', user='alice', demand={'v5p': 4}, cluster='a')
    ]
    usage['clusters'] = [
        _cluster('alice', {'tpu-v5p-8': 1},
                 status=status_lib.ClusterStatus.INIT,
                 name='a')
    ]
    # 4 from the cluster record only, so bob's 4 fit.
    assert tpu_quota.current_usage('chips')[1] == {'v5p': 4}
    _apply(_request('bob', 'tpu-v5p-8'))


def test_reservation_dropped_once_job_is_visible(usage):
    usage['running'] = [
        _running('req-a', user='alice', demand={'v6e': 8}, job='train'),
        _running('req-b', user='alice', demand={'v6e': 8}, job='eval'),
    ]
    usage['jobs'] = [_job('alice', '1x[tpu-v6e-8:1]', name='train')]
    assert tpu_quota.current_usage('chips')[1] == {'v6e': 16}


def test_pending_reservations_skip_self_and_unadmitted(usage, monkeypatch):
    monkeypatch.setattr(common_utils, 'get_current_request_id',
                        lambda: 'req-me')
    usage['running'] = [
        _running('req-me', user='alice', demand={'v5p': 4}),
        # Running but still waiting for the admission lock: nothing recorded.
        _running('req-b', status_msg=''),
        _running('req-c', status_msg='[tpu-quota admitted] not json'),
        _running('req-d', user='bob', demand={'v5p': 4}, unit='slices'),
        _running('req-e', user='carol', demand={'v5p': 4}),
    ]
    assert [r['user'] for r in tpu_quota.pending_reservations(set(), set())
           ] == ['bob', 'carol']
    # The slices reservation does not mix with a chips quota.
    assert tpu_quota.current_usage('chips') == ({
        'carol': {
            'v5p': 4
        }
    }, {
        'v5p': 4
    })


def test_record_admission_writes_marker(monkeypatch):
    request = mock.MagicMock()
    request.status_msg = None

    @contextlib.contextmanager
    def update_request(request_id):
        assert request_id == 'req-me'
        yield request

    monkeypatch.setattr(common_utils, 'is_in_request_context', lambda: True)
    monkeypatch.setattr(common_utils, 'get_current_request_id',
                        lambda: 'req-me')
    monkeypatch.setattr(api_requests, 'update_request', update_request)
    tpu_quota._record_admission(  # pylint: disable=protected-access
        'alice', 'chips', {'v5p': 4}, 'c', None)
    marker = '[tpu-quota admitted] '
    assert request.status_msg.startswith(marker)
    assert json.loads(request.status_msg[len(marker):]) == {
        'user': 'alice',
        'unit': 'chips',
        'demand': {
            'v5p': 4
        },
        'cluster': 'c',
        'job': None,
    }


def test_record_admission_outside_request_context_is_noop(monkeypatch):
    monkeypatch.setattr(common_utils, 'is_in_request_context', lambda: False)
    monkeypatch.setattr(api_requests, 'update_request',
                        mock.MagicMock(side_effect=AssertionError))
    tpu_quota._record_admission(  # pylint: disable=protected-access
        'alice', 'chips', {'v5p': 4}, None, None)


def test_cluster_records_include_controller_launched_clusters(monkeypatch):
    calls = []
    monkeypatch.setattr(backend_utils, 'get_clusters',
                        lambda **kwargs: calls.append(kwargs) or [])
    records = tpu_quota._cluster_records()  # pylint: disable=protected-access
    assert not records
    assert calls[0]['all_users'] is True
    assert calls[0]['_include_is_managed'] is True


def test_job_on_pool_is_not_counted(usage):
    # The pool's workers hold the TPUs and are counted as clusters.
    usage['clusters'] = [
        _cluster('alice', {'tpu-v5p-8': 1}, name='pool-1'),
        _cluster('alice', {'tpu-v5p-8': 1}, name='pool-2'),
    ]
    usage['jobs'] = [
        _job('alice',
             '1x[tpu-v5p-8:1]',
             status=managed_job_state.ManagedJobStatus.RUNNING,
             pool='pool',
             cluster_name='pool-1'),
        _job('alice', '1x[tpu-v5p-8:1]', pool='pool'),
    ]
    assert tpu_quota.current_usage('chips')[1] == {'v5p': 8}


def test_job_with_visible_cluster_is_counted_once(usage):
    usage['clusters'] = [
        _cluster('bob', {'tpu-v6e-8': 1},
                 status=status_lib.ClusterStatus.INIT,
                 name='train-7')
    ]
    usage['jobs'] = [
        _job('bob',
             '1x[tpu-v6e-8:1]',
             status=managed_job_state.ManagedJobStatus.STARTING,
             cluster_name='train-7'),
        # Still waiting in the controller's queue: no cluster yet.
        _job('bob', '1x[tpu-v6e-8:1]', cluster_name=None),
    ]
    assert tpu_quota.current_usage('chips')[1] == {'v6e': 16}


def test_pool_worker_launch_is_gated(quota_file, usage):
    quota_file({'unit': 'chips', 'pool': {'v5p': 8}, 'per_user': {}})
    usage['clusters'] = [_cluster('alice', {'tpu-v5p-8': 1}, name='pool-1')]
    replica = request_names.AdminPolicyRequestName.SERVE_LAUNCH_REPLICA
    _apply(
        _request('alice',
                 'tpu-v5p-8',
                 request_name=replica,
                 cluster_name='pool-2'))
    assert usage['admitted'][0]['cluster'] == 'pool-2'
    usage['clusters'].append(_cluster('alice', {'tpu-v5p-8': 1}, name='pool-2'))
    with pytest.raises(exceptions.ExecutionPausedError):
        _apply(
            _request('alice',
                     'tpu-v5p-8',
                     request_name=replica,
                     cluster_name='pool-3'))


@pytest.mark.parametrize('request_name', [
    request_names.AdminPolicyRequestName.SERVE_UP,
    request_names.AdminPolicyRequestName.SERVE_UPDATE,
])
def test_pool_definition_checks_worker_size_only(quota_file, usage,
                                                 request_name):
    quota_file({'unit': 'chips', 'pool': {'v5p': 8}, 'per_user': {}})
    usage['clusters'] = [
        _cluster('alice', {'tpu-v5p-8': 1}, num_nodes=2, name='pool-1')
    ]
    # The pool is full, but defining a pool takes no quota: its workers are
    # gated when they launch.
    _apply(_request('alice', 'tpu-v5p-8', request_name=request_name))
    assert usage['lock'].acquisitions == 0
    assert usage['admitted'] == []
    with pytest.raises(ValueError, match='worker needs 16 v5p chips'):
        _apply(_request('alice', 'tpu-v5p-32', request_name=request_name))
