"""Persist original requests and replay only fenced, safe launch stages."""
# pylint: disable=protected-access,redefined-outer-name
# pylint: disable=unused-argument
import contextlib
from types import SimpleNamespace
from unittest import mock

import pytest
import uvicorn

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
