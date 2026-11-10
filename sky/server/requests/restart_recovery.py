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
    if (request.name == 'sky.launch' and
            request.status == requests.RequestStatus.WAITING and
            request.pid is None):
        # WAITING already means replay of this same request is safe; preserve
        # quota/lock/backoff pauses as well as provider-capacity conditions.
        return True
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


def managed_launch_request(job_id: int, task_id: int, cluster_name: str):
    """Find the original request of an exact managed task, without launching.

    The controller and API share a persistent backend only in opt-in single
    supervisor mode. Ownership, exact cluster name and the structured native
    observation target must all agree. Ambiguity never triggers cleanup/replay.
    """
    # Lazy imports avoid requests/jobs initialization cycles.
    # pylint: disable=import-outside-toplevel
    from sky import global_user_state
    from sky.jobs import state
    from sky.provision import observation
    from sky.server.requests import requests
    if not enabled():
        return None
    owner = state.get_job_info_row(job_id)
    if owner is None or owner.user_hash is None:
        raise ValueError('Managed launch owner is unknown')
    target = state.provisioning_observation_target(job_id, task_id)
    candidates = requests.get_request_tasks(
        requests.RequestTaskFilter(
            cluster_names=[cluster_name],
            user_id=owner.user_hash,
            include_request_names=['sky.launch'],
            status=requests.RequestStatus.active_statuses() +
            [requests.RequestStatus.SUCCEEDED]))
    matched = []
    for request in candidates:
        body = request.request_body
        if (request.user_id != owner.user_hash or
                request.cluster_name != cluster_name):
            continue
        if (not getattr(body, 'is_launched_by_jobs_controller', False) or
                body.extra_launch_context.get(
                    observation.LAUNCH_CONTEXT_KEY) != target):
            if request.status != requests.RequestStatus.SUCCEEDED:
                raise ValueError('Existing launch has a different task binding')
            continue
        if request.status == requests.RequestStatus.SUCCEEDED:
            # A finished launch can be attached only in its current native
            # generation; an old successful attempt must never be reused.
            journals = [
                queued_resource_wait.load(p) for p in
                queued_resource_wait.JOURNAL_ROOT.expanduser().glob('*.json')
            ]
            current = global_user_state.get_cluster_hash(cluster_name)
            if not any(v['request_id'] == request.request_id and
                       v.get('stage') == 'ready' and
                       v.get('native_cluster_hash') == current and current
                       for v in journals):
                continue
        matched.append(request.request_id)
    if len(matched) > 1:
        raise ValueError('Multiple live requests for the managed task')
    return matched[0] if matched else None


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
