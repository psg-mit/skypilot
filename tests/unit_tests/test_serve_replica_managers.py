"""Tests for sky/serve/replica_managers.py.

Currently focused on `SkyPilotReplicaManager.__init__` startup ordering:
the daemon threads (especially `_job_status_fetcher`) must NOT race the
main thread for `self.lock` before `_recover_replica_operations` runs.
"""
# pylint: disable=protected-access,redefined-outer-name
import contextlib
import threading
from unittest import mock

import pytest

from sky.serve import replica_managers
from sky.utils import common_utils
from sky.utils import thread_utils


class TestSkyPilotReplicaManagerInitOrdering:
    """`SkyPilotReplicaManager.__init__` must run `_recover_replica_operations`
    BEFORE starting the `_job_status_fetcher` / `_thread_pool_refresher` /
    `_replica_prober` daemon threads.

    If the daemon threads start first, `_job_status_fetcher` will acquire
    `self.lock` (via the `@with_lock` decorator on `_fetch_job_status`)
    and perform a per-replica SSH/gRPC call to query job status. When a
    replica's head node is unreachable (pod / VM gone), each SSH connect
    hangs at the kernel TCP timeout (tens of seconds to minutes). The
    main thread then blocks on `_recover_replica_operations`'s
    `with self.lock:` for the full hang duration, never returns from
    `SkyPilotReplicaManager.__init__`, and `uvicorn.run` is never called.

    With HA recovery changes, `_wait_for_controller_ready`
    then times out (60s) → `_bail_on_boot_failure` → `os._exit(1)` →
    daemon retries → same race → infinite recovery loop.

    The fix: recovery first, daemon threads after.
    """

    def test_recover_called_before_threads_start(self):
        """Verify the call order: `_recover_replica_operations` first,
        then each daemon thread's `.start()`."""
        call_order = []

        def _record(name):

            def _fn(*_args, **_kwargs):
                call_order.append(name)

            return _fn

        # Patch the heavy deps so __init__ doesn't actually do work.
        # We only care about the call order.
        with mock.patch.object(
                replica_managers.ReplicaManager, '__init__',
                return_value=None), \
             mock.patch(
                 'sky.serve.replica_managers.serve_state.get_yaml_content',
                 return_value='dummy: yaml'), \
             mock.patch(
                 'sky.serve.replica_managers.task_lib.Task.from_yaml_str',
                 return_value=mock.MagicMock()), \
             mock.patch(
                 'sky.serve.replica_managers.spot_placer.SpotPlacer.from_task',
                 return_value=None), \
             mock.patch.object(
                 replica_managers.SkyPilotReplicaManager,
                 '_recover_replica_operations',
                 _record('recover')), \
             mock.patch(
                 'sky.serve.replica_managers.threading.Thread') as mock_thread:
            # Each Thread(target=...).start() records the target's name
            # via our side_effect on .start().
            def thread_factory(*_args, **kwargs):
                target = kwargs.get('target')
                t = mock.Mock()
                target_name = getattr(target, '__name__', repr(target))
                t.start.side_effect = _record(f'thread_start:{target_name}')
                return t

            mock_thread.side_effect = thread_factory

            spec = mock.MagicMock()
            replica_managers.SkyPilotReplicaManager(service_name='svc',
                                                    spec=spec,
                                                    version=1)

        # `recover` must come before any `thread_start:*` entry. The
        # daemon threads themselves may be created in any order relative
        # to each other (we don't constrain that), but ALL of them must
        # appear after `recover`.
        assert 'recover' in call_order, (
            f'_recover_replica_operations was never called; '
            f'call_order={call_order}')
        recover_idx = call_order.index('recover')
        for i, name in enumerate(call_order):
            if name.startswith('thread_start:'):
                assert i > recover_idx, (
                    f'{name} happened at index {i} before recover at '
                    f'index {recover_idx}; call_order={call_order}. '
                    f'Daemon threads must NOT start until '
                    f'_recover_replica_operations has finished — '
                    f'see the docstring of '
                    f'TestSkyPilotReplicaManagerInitOrdering.')

    def test_all_three_daemon_threads_are_started(self):
        """Sanity: regardless of ordering, the three daemon threads
        (_thread_pool_refresher / _job_status_fetcher / _replica_prober)
        still all start. The fix is purely a reorder, not a removal."""
        started_targets = []

        with mock.patch.object(
                replica_managers.ReplicaManager, '__init__',
                return_value=None), \
             mock.patch(
                 'sky.serve.replica_managers.serve_state.get_yaml_content',
                 return_value='dummy: yaml'), \
             mock.patch(
                 'sky.serve.replica_managers.task_lib.Task.from_yaml_str',
                 return_value=mock.MagicMock()), \
             mock.patch(
                 'sky.serve.replica_managers.spot_placer.SpotPlacer.from_task',
                 return_value=None), \
             mock.patch.object(
                 replica_managers.SkyPilotReplicaManager,
                 '_recover_replica_operations'), \
             mock.patch(
                 'sky.serve.replica_managers.threading.Thread') as mock_thread:

            def thread_factory(*_args, **kwargs):
                target = kwargs.get('target')
                started_targets.append(getattr(target, '__name__', None))
                t = mock.Mock()
                return t

            mock_thread.side_effect = thread_factory

            spec = mock.MagicMock()
            replica_managers.SkyPilotReplicaManager(service_name='svc',
                                                    spec=spec,
                                                    version=1)

        # Bound methods on the instance — verify by name.
        assert '_thread_pool_refresher' in started_targets
        assert '_job_status_fetcher' in started_targets
        assert '_replica_prober' in started_targets


@pytest.fixture
def thread_pool_manager(monkeypatch):
    manager = object.__new__(replica_managers.SkyPilotReplicaManager)
    manager.lock = threading.RLock()
    manager._service_name = 'pool'
    manager._is_pool = True
    manager._spot_placer = None
    manager.least_recent_version = 1
    manager._launch_thread_pool = thread_utils.ThreadSafeDict()
    manager._down_thread_pool = thread_utils.ThreadSafeDict()
    manager._replica_to_request_id = thread_utils.ThreadSafeDict()
    manager._terminate_replica = mock.Mock()
    manager._handle_sky_down_finish = mock.Mock()
    info = replica_managers.ReplicaInfo(1, 'pool-1', '-', False, None, 1, None)
    monkeypatch.setattr(replica_managers.serve_state,
                        'get_replica_info_from_id', lambda *_: info)
    monkeypatch.setattr(replica_managers.serve_state, 'get_replica_infos',
                        lambda *_: [info])
    monkeypatch.setattr(replica_managers.serve_state, 'add_or_update_replica',
                        mock.Mock())
    monkeypatch.setattr(replica_managers.filelock, 'FileLock',
                        lambda *_: contextlib.nullcontext())
    monkeypatch.setattr(replica_managers.controller_utils, 'can_provision',
                        lambda *_: True)
    monkeypatch.setattr(replica_managers.controller_utils, 'can_terminate',
                        lambda *_: True)
    return manager, info


@pytest.mark.parametrize('failed', [False, True])
def test_completed_launch_with_pending_record_is_collected(
        thread_pool_manager, failed):
    manager, info = thread_pool_manager

    def launch():
        if failed:
            raise RuntimeError('launch failed')

    thread = thread_utils.SafeThread(target=launch)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    manager._launch_thread_pool[1] = thread
    manager._replica_to_request_id[1] = 'original-request'
    # Simulate a persisted status reset while the launch thread was running.
    assert info.status == replica_managers.serve_state.ReplicaStatus.PENDING
    manager._refresh_thread_pool()
    manager._refresh_thread_pool()

    assert not manager._launch_thread_pool
    assert not manager._replica_to_request_id
    expected = (common_utils.ProcessStatus.FAILED
                if failed else common_utils.ProcessStatus.SUCCEEDED)
    assert info.status_property.sky_launch_status == expected
    assert manager._terminate_replica.call_count == int(failed)


def test_unstarted_launch_with_provisioning_record_starts_once(
        thread_pool_manager):
    manager, info = thread_pool_manager
    launched = mock.Mock()
    thread = thread_utils.SafeThread(target=launched)
    manager._launch_thread_pool[1] = thread
    info.status_property.sky_launch_status = common_utils.ProcessStatus.RUNNING

    manager._refresh_thread_pool()
    thread.join(timeout=5)
    manager._refresh_thread_pool()
    launched.assert_called_once_with()
    assert not manager._launch_thread_pool
    assert (info.status_property.sky_launch_status ==
            common_utils.ProcessStatus.SUCCEEDED)


def test_completed_down_with_scheduled_record_is_collected(thread_pool_manager):
    manager, info = thread_pool_manager
    thread = thread_utils.SafeThread(target=lambda: None)
    thread.start()
    thread.join(timeout=5)
    manager._down_thread_pool[1] = thread
    info.status_property.sky_down_status = common_utils.ProcessStatus.SCHEDULED

    manager._refresh_thread_pool()
    manager._refresh_thread_pool()
    assert not manager._down_thread_pool
    manager._handle_sky_down_finish.assert_called_once_with(info,
                                                            format_exc=None)


@pytest.mark.parametrize('disposition',
                         ['scale_down', 'preempted', 'purged', 'old_version'])
def test_failed_teardown_retains_worker_identity(monkeypatch, disposition):
    manager = object.__new__(replica_managers.SkyPilotReplicaManager)
    manager._service_name = 'pool'
    manager.latest_version = 2 if disposition == 'old_version' else 1
    manager._last_termination_check = {1: 10}
    info = replica_managers.ReplicaInfo(1, 'pool-1', '-', False, None, 1, None)
    info.status_property.sky_launch_status = (
        common_utils.ProcessStatus.SUCCEEDED)
    info.status_property.is_scale_down = disposition in ('scale_down',
                                                         'preempted')
    info.status_property.preempted = disposition == 'preempted'
    info.status_property.purged = disposition == 'purged'
    save = mock.Mock()
    remove = mock.Mock()
    monkeypatch.setattr(replica_managers.serve_state, 'add_or_update_replica',
                        save)
    monkeypatch.setattr(replica_managers.serve_state, 'remove_replica', remove)

    manager._handle_sky_down_finish(info, format_exc='provider teardown failed')
    save.assert_called_once_with('pool', 1, info)
    remove.assert_not_called()
    assert info.cluster_name == 'pool-1'
    assert (info.status_property.sky_down_status ==
            common_utils.ProcessStatus.FAILED)
