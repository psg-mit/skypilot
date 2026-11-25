"""A failed teardown can retry without discarding cloud ownership."""
from types import SimpleNamespace
from unittest import mock

import fastapi
from fastapi.testclient import TestClient
import pytest

from sky.serve import controller
from sky.serve import serve_state


@pytest.mark.parametrize('status, expected', [
    (serve_state.ReplicaStatus.FAILED_CLEANUP, 200),
    (serve_state.ReplicaStatus.FAILED, 409),
    (serve_state.ReplicaStatus.FAILED_PROVISION, 409),
    (serve_state.ReplicaStatus.SHUTTING_DOWN, 409),
])
def test_cleanup_retry_uses_normal_teardown_without_purge(
        monkeypatch, status, expected):
    instance = object.__new__(controller.SkyServeController)
    instance._service_name = 'logan-pool'  # pylint: disable=protected-access
    instance._app = fastapi.FastAPI()  # pylint: disable=protected-access
    instance._replica_manager = mock.Mock()  # pylint: disable=protected-access
    instance._host = '127.0.0.1'  # pylint: disable=protected-access
    instance._port = 0  # pylint: disable=protected-access
    monkeypatch.setattr(controller.uvicorn, 'run', lambda *a, **k: None)
    monkeypatch.setattr(serve_state, 'get_replica_info_from_id',
                        lambda *_: SimpleNamespace(status=status))
    with mock.patch.object(controller.threading.Thread, 'start'):
        instance.run()
    response = TestClient(instance._app).post(  # pylint: disable=protected-access
        '/controller/terminate_replica',
        json=dict(replica_id=7, purge=False))
    assert response.status_code == expected
    manager = instance._replica_manager  # pylint: disable=protected-access
    if expected == 200:
        manager.scale_down.assert_called_once_with(7, purge=False)
    else:
        manager.scale_down.assert_not_called()
