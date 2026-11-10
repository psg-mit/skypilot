"""Persist original requests and replay only fenced, safe launch stages."""
# pylint: disable=protected-access,redefined-outer-name
# pylint: disable=unused-argument
import contextlib
from types import SimpleNamespace
from unittest import mock

import pytest
import uvicorn

from sky.jobs import state
from sky.provision import observation
from sky.provision.gcp import queued_resource_wait as wait
from sky.server import uvicorn as sky_uvicorn
from sky.server.requests import requests
from sky.server.requests import restart_recovery as recovery


@pytest.fixture
def enabled(monkeypatch, tmp_path):
    monkeypatch.setenv(recovery.ENV_VAR, '1')
    monkeypatch.setattr(wait, 'JOURNAL_ROOT', tmp_path / 'waits')


@pytest.mark.parametrize('status,pid,expected', [('PENDING', None, True),
                                                 ('PENDING', 123, False),
                                                 ('RUNNING', 123, False),
                                                 ('SUCCEEDED', None, False),
                                                 ('CANCELLED', None, False)])
def test_pending_pool_apply_only_if_never_executed(enabled, status, pid,
                                                   expected):
    request = SimpleNamespace(name='sky.jobs.pool_apply',
                              request_id='original',
                              status=requests.RequestStatus[status],
                              pid=pid)
    assert recovery.resumable(request) == expected


def test_persistent_reset_keeps_database_logs_and_uploaded_inputs(
        enabled, monkeypatch):
    backend = mock.Mock()
    monkeypatch.setattr(requests.request_storage, 'get_request_backend',
                        lambda: backend)
    delete_database = mock.Mock()
    delete_logs = mock.Mock()
    reset_blobs = mock.Mock()
    monkeypatch.setattr(requests.server_common,
                        'clear_local_api_server_database', delete_database)
    monkeypatch.setattr(requests.shutil, 'rmtree', delete_logs)
    monkeypatch.setattr(requests.bs, 'get_blob_storage', reset_blobs)
    requests.reset_db_and_logs()
    backend.reset_on_startup.assert_called_once()
    assert not delete_database.called
    assert not delete_logs.called
    assert not reset_blobs.called


def test_recovery_requeues_exact_id_and_does_not_replay_unsafe_stages(
        enabled, monkeypatch):
    original = SimpleNamespace(request_id='original',
                               name='sky.launch',
                               pid=None,
                               status=requests.RequestStatus.PENDING)
    unsafe = SimpleNamespace(request_id='unsafe',
                             name='sky.exec',
                             pid=123,
                             status=requests.RequestStatus.RUNNING)
    rows = {r.request_id: r for r in (original, unsafe)}
    monkeypatch.setattr(requests, 'get_request_tasks',
                        lambda _: list(rows.values()))
    monkeypatch.setattr(requests, 'get_request', rows.get)

    @contextlib.contextmanager
    def update(request_id):
        yield rows[request_id]

    monkeypatch.setattr(requests, 'update_request', update)
    failed = mock.Mock()
    monkeypatch.setattr(requests, 'set_request_failed', failed)
    enqueue = mock.Mock()
    recovery.recover(enqueue)
    enqueue.assert_called_once_with(original)
    assert original.request_id == 'original'
    assert original.status == requests.RequestStatus.PENDING
    assert original.pid is None
    assert failed.call_args.args[0] == 'unsafe'


def test_graceful_restart_preserves_wait_instead_of_cancelling(
        enabled, monkeypatch):
    request = SimpleNamespace(request_id='original',
                              name='sky.launch',
                              pid=123,
                              status=requests.RequestStatus.RUNNING)
    wait.save(wait.JOURNAL_ROOT / 'wait.json',
              dict(request_id='original', stage='waiting'))

    @contextlib.contextmanager
    def update(_):
        yield request

    monkeypatch.setattr(requests, 'update_request', update)
    signal = mock.Mock()
    monkeypatch.setattr(sky_uvicorn.os, 'kill', signal)
    server = sky_uvicorn.Server(uvicorn.Config(app='unused'))
    server.interrupt_request_for_retry('original', should_retry=False)
    assert request.status == requests.RequestStatus.WAITING
    assert request.pid is None
    assert request.request_id == 'original'
    signal.assert_called_once()


@pytest.mark.parametrize('status',
                         ['PENDING', 'WAITING', 'RUNNING', 'SUCCEEDED'])
def test_managed_controller_finds_exact_original_request(
        enabled, monkeypatch, status):
    from sky import global_user_state  # pylint: disable=import-outside-toplevel
    monkeypatch.setattr(state, 'get_job_info_row',
                        lambda _: SimpleNamespace(user_hash='owner'))
    body = SimpleNamespace(is_launched_by_jobs_controller=True,
                           extra_launch_context={
                               observation.LAUNCH_CONTEXT_KEY:
                                   state.provisioning_observation_target(10, 0)
                           })
    request = SimpleNamespace(request_id='original',
                              name='sky.launch',
                              status=requests.RequestStatus[status],
                              user_id='owner',
                              cluster_name='exact-cluster',
                              request_body=body)
    monkeypatch.setattr(requests, 'get_request_tasks', lambda _: [request])
    monkeypatch.setattr(global_user_state, 'get_cluster_hash', lambda _: 'hash')
    wait.save(
        wait.JOURNAL_ROOT / 'ready.json',
        dict(request_id='original', stage='ready', native_cluster_hash='hash'))
    assert recovery.managed_launch_request(10, 0, 'exact-cluster') == 'original'
    request.user_id = 'foreign'
    assert recovery.managed_launch_request(10, 0, 'exact-cluster') is None
    request.user_id = 'owner'
    body.extra_launch_context[observation.LAUNCH_CONTEXT_KEY]['task_id'] = 1
    if status == 'SUCCEEDED':
        assert recovery.managed_launch_request(10, 0, 'exact-cluster') is None
    else:
        with pytest.raises(ValueError, match='task binding'):
            recovery.managed_launch_request(10, 0, 'exact-cluster')


def test_ambiguous_managed_requests_never_choose_or_relaunch(
        enabled, monkeypatch):
    monkeypatch.setattr(state, 'get_job_info_row',
                        lambda _: SimpleNamespace(user_hash='owner'))
    body = SimpleNamespace(is_launched_by_jobs_controller=True,
                           extra_launch_context={
                               observation.LAUNCH_CONTEXT_KEY:
                                   state.provisioning_observation_target(10, 0)
                           })
    rows = [
        SimpleNamespace(request_id=str(i),
                        user_id='owner',
                        cluster_name='cluster',
                        request_body=body,
                        status=requests.RequestStatus.WAITING) for i in range(2)
    ]
    monkeypatch.setattr(requests, 'get_request_tasks', lambda _: rows)
    with pytest.raises(ValueError, match='Multiple'):
        recovery.managed_launch_request(10, 0, 'cluster')
