"""Opt-in single-server recovery of original queued TPU request identities.

Enable only on a persistent, single API supervisor, never multiple replicas
sharing one request backend. Completed requests, logs, uploads and blobs stay
in their original stores; the normal reference-aware collectors own retention.
"""
import os

from sky import exceptions
from sky.provision.gcp import queued_resource_wait

ENV_VAR = 'SKYPILOT_DURABLE_TPU_REQUESTS'
RECOVERABLE_PENDING = {'sky.launch', 'sky.jobs.pool_apply'}


def enabled() -> bool:
    return os.environ.get(ENV_VAR) == '1'


def resumable(request) -> bool:
    # Lazy import: requests imports this module for startup handling.
    # pylint: disable=import-outside-toplevel
    from sky.server.requests import requests
    if request.status > requests.RequestStatus.RUNNING:
        return False
    if (request.name in RECOVERABLE_PENDING and
            request.status == requests.RequestStatus.PENDING and
            request.pid is None):
        return True
    if request.name != 'sky.launch':
        return False
    for path in queued_resource_wait.JOURNAL_ROOT.expanduser().glob('*.json'):
        value = queued_resource_wait.load(path)
        if (value['request_id'] == request.request_id and
                value.get('stage') in ('creating', 'waiting')):
            return True
    return False


def recover(enqueue) -> None:
    """Run once before accepting requests or starting controller recovery."""
    # Lazy imports avoid a requests/executor initialization cycle.
    # pylint: disable=import-outside-toplevel
    from sky.server import daemons
    from sky.server.requests import requests
    if not enabled():
        return
    internal = {daemon.id for daemon in daemons.INTERNAL_REQUEST_DAEMONS}
    # The in-memory queues disappeared with the old supervisor. Re-register
    # existing singleton daemon IDs as executable before lifespan's dedup runs.
    for request_id in internal:
        request = requests.get_request(request_id)
        if request is not None:
            with requests.update_request(request_id) as original:
                original.status = requests.RequestStatus.PENDING
                original.pid = None
                original.error = None
                original.finished_at = None
                original.should_retry = False
            enqueue(request)
    tasks = requests.get_request_tasks(
        requests.RequestTaskFilter(status=[
            requests.RequestStatus.PENDING, requests.RequestStatus.WAITING,
            requests.RequestStatus.RUNNING
        ]))
    for task in tasks:
        if task.request_id in internal:
            continue
        request = requests.get_request(task.request_id)
        if request is None:
            continue
        if resumable(request):
            with requests.update_request(request.request_id) as original:
                original.pid = None
                if original.status != requests.RequestStatus.PENDING:
                    original.status = requests.RequestStatus.WAITING
                original.status_msg = 'Resuming original request after restart'
            enqueue(request)
        else:
            requests.set_request_failed(
                request.request_id,
                exceptions.RequestInterruptedError(
                    'Interrupted outside a durable TPU wait; request retained '
                    'for reconciliation, without automatic replay.'))
