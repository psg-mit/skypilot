"""Persistent pool recovery retains request identity without blind replay."""
# pylint: disable=redefined-outer-name
from types import SimpleNamespace
from unittest import mock

import pytest

from sky import global_user_state
from sky.serve import pool_recovery as recovery
from sky.serve import replica_managers
from sky.server.requests import payloads
from sky.server.requests import requests


@pytest.fixture
def pool(monkeypatch, tmp_path):
    monkeypatch.setenv(recovery.restart_recovery.ENV_VAR, '1')
    monkeypatch.setattr(recovery.managed_job_utils, 'is_consolidation_mode',
                        lambda: True)
    monkeypatch.setattr(recovery.common_utils, 'get_user_hash', lambda: 'owner')
    monkeypatch.setattr(recovery.serve_state, 'get_service_from_name',
                        lambda _: dict(pool=True, user_hash='owner'))
    monkeypatch.setattr(recovery.serve_state, 'get_replica_info_from_id',
                        lambda *args: SimpleNamespace(cluster_name='pool-1'))
    monkeypatch.setattr(recovery.queued_resource_wait, 'JOURNAL_ROOT', tmp_path)
    rows = []
    monkeypatch.setattr(requests, 'get_request_tasks', lambda _: rows)
    body = payloads.LaunchBody(task='run: echo setup done\n',
                               cluster_name='pool-1',
                               env_vars={'SKYPILOT_USER_ID': 'owner'},
                               retry_until_up=True,
                               is_launched_by_sky_serve_controller=True)
    return body, rows


def request(body, name='original', created=1, status='WAITING'):
    return SimpleNamespace(request_id=name,
                           created_at=created,
                           user_id='owner',
                           cluster_name='pool-1',
                           status=requests.RequestStatus[status],
                           request_body=body.model_copy(deep=True))


@pytest.mark.parametrize('status', ['PENDING', 'WAITING', 'RUNNING'])
def test_attach_preserves_request_and_inputs(pool, status):
    body, rows = pool
    rows.append(request(body, status=status))
    before = rows[0].request_body.model_dump(mode='json')
    assert recovery.find_existing_launch(body) == 'original'
    assert rows[0].request_body.model_dump(mode='json') == before
    assert rows[0].status == requests.RequestStatus[status]


def test_old_equivalent_duplicates_choose_original_without_mutating_rows(pool):
    body, rows = pool
    rows.extend([request(body, 'new', 2), request(body)])
    assert recovery.find_existing_launch(body) == 'original'
    assert [r.request_id for r in rows] == ['new', 'original']


@pytest.mark.parametrize('field,value', [
    ('task', 'run: echo different\n'),
    ('retry_until_up', False),
    ('no_setup', True),
    ('is_launched_by_jobs_controller', True),
    ('is_launched_by_sky_serve_controller', False),
    ('file_mounts_blob_id', 'different-upload'),
    ('override_skypilot_config', {
        'gcp': {
            'project_id': 'foreign'
        }
    }),
    ('env_vars', {
        'SKYPILOT_USER_ID': 'foreign'
    }),
])
def test_different_inputs_block_new_launch(pool, field, value):
    body, rows = pool
    rows.append(request(body))
    setattr(rows[0].request_body, field, value)
    with pytest.raises(recovery.PoolLaunchRecoveryError,
                       match='ownership or inputs'):
        recovery.find_existing_launch(body)


@pytest.mark.parametrize('field,value', [('user_id', 'foreign'),
                                         ('cluster_name', 'foreign-1')])
def test_foreign_active_request_is_not_ignored(pool, field, value):
    body, rows = pool
    rows.append(request(body))
    setattr(rows[0], field, value)
    with pytest.raises(recovery.PoolLaunchRecoveryError):
        recovery.find_existing_launch(body)


def test_only_controller_sizing_and_telemetry_envs_may_differ(pool):
    body, rows = pool
    rows.append(request(body))
    rows[0].request_body.env_vars.update(SKYPILOT_POD_MEMORY_GB_LIMIT='28',
                                         SKYPILOT_DURABLE_TPU_REQUESTS='1',
                                         SKYPILOT_USAGE_RUN_ID='old')
    assert recovery.find_existing_launch(body) == 'original'
    rows[0].request_body.env_vars[
        'GOOGLE_APPLICATION_CREDENTIALS'] = 'different'
    with pytest.raises(recovery.PoolLaunchRecoveryError):
        recovery.find_existing_launch(body)


@pytest.mark.parametrize('disabled',
                         ['opt-in', 'consolidation', 'ordinary', 'serve'])
def test_other_launch_modes_are_unchanged(pool, monkeypatch, disabled):
    body, _ = pool
    if disabled == 'opt-in':
        monkeypatch.delenv(recovery.restart_recovery.ENV_VAR)
    elif disabled == 'consolidation':
        monkeypatch.setattr(recovery.managed_job_utils, 'is_consolidation_mode',
                            lambda: False)
    elif disabled == 'ordinary':
        body.is_launched_by_sky_serve_controller = False
    else:
        monkeypatch.setattr(recovery.serve_state, 'get_service_from_name',
                            lambda _: dict(pool=False, user_hash='owner'))
    assert recovery.find_existing_launch(body) is None


def test_completed_request_requires_current_provider_incarnation(
        pool, monkeypatch):
    body, rows = pool
    rows.append(request(body, status='SUCCEEDED'))
    monkeypatch.setattr(global_user_state, 'get_cluster_hash',
                        lambda _: 'current')
    assert recovery.find_existing_launch(body) is None
    recovery.queued_resource_wait.save(
        recovery.queued_resource_wait.JOURNAL_ROOT / 'ready.json',
        dict(request_id='original',
             stage='ready',
             native_cluster_hash='current'))
    assert recovery.find_existing_launch(body) == 'original'
    monkeypatch.setattr(global_user_state, 'get_cluster_hash',
                        lambda _: 'replaced')
    assert recovery.find_existing_launch(body) is None


def test_backend_failure_blocks_replay_and_hides_payload(pool, monkeypatch):
    body, _ = pool
    monkeypatch.setattr(requests, 'get_request_tasks',
                        mock.Mock(side_effect=OSError('secret')))
    with pytest.raises(recovery.PoolLaunchRecoveryError) as error:
        recovery.find_existing_launch(body)
    assert 'secret' not in str(error.value)


def test_recovery_mismatch_does_not_clean_up_or_relaunch(
        pool, monkeypatch, tmp_path):
    body, _ = pool
    monkeypatch.setattr(replica_managers.context, 'get', mock.MagicMock)
    launch = mock.Mock(side_effect=recovery.PoolLaunchRecoveryError('mismatch'))
    monkeypatch.setattr(replica_managers.sdk, 'launch', launch)
    cleanup = mock.Mock()
    monkeypatch.setattr(replica_managers, 'terminate_cluster', cleanup)
    cancelled = {}
    monkeypatch.setattr(replica_managers.time, 'sleep',
                        lambda _: cancelled.update({1: True}))
    replica_managers.launch_cluster(1, body.task, body.cluster_name,
                                    str(tmp_path / 'launch.log'), {}, cancelled)
    launch.assert_called_once()
    cleanup.assert_not_called()
