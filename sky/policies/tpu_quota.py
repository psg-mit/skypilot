"""Per-user TPU quota with borrowing, enforced as an admin policy.

Every user gets a share of each TPU family. A launch that would exceed the
user's share is still admitted while the lab-wide pool has room, so idle
shares are borrowed instead of wasted. A launch that fits neither the share
nor the pool is parked: the request stays queued on the API server and is
re-checked every ``RETRY_WAIT_SECONDS`` until it fits or is cancelled.

The limits live in a YAML file that is re-read on every request, so editing
the file changes the quota without restarting the API server:

.. code-block:: yaml

    # ~/.sky/tpu_quota.yaml (or $SKYPILOT_TPU_QUOTA_FILE)
    unit: chips          # or slices
    pool:                # lab-wide limits per TPU family
      v5p: 48
      v6e: 32
    per_user:
      default: {v5p: 16, v6e: 8}
      alice: {v5p: 32}   # overrides for one user, by SkyPilot user name

Families are the version token of the accelerator name: ``tpu-v5p-8`` is
``v5p``, ``tpu-v5litepod-4`` is ``v5litepod``, ``tpu-v6e-8`` is ``v6e``.
A family missing from ``pool`` is not governed. A family in ``pool`` but
missing from a user's limits and from ``default`` is limited by the pool only.

Usage is counted from the clusters of all users that are UP or INIT and from
the managed jobs of all users that are not finished, so both ``sky launch``
and ``sky jobs launch`` must go through the same API server. The accounting
is by accelerator name, so TPUs of the TPU API and of the Compute Engine API
count the same way.

The pool is a hard cap. Shares decide who goes first when the pool is full:
a parked request from a user within their share is admitted before a request
that would take its user over their share, so borrowed capacity flows back as
soon as it frees up. Only admission is enforced: nobody is preempted to
reclaim a share.

Admission is serialized across the executor workers with a distributed lock,
and an admitted request records its demand on its own request row until its
cluster or managed job shows up in the state above. Concurrent requests
therefore see each other's demand and cannot jointly exceed the pool.
"""
import ast
import dataclasses
import json
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from sky import admin_policy
from sky import core
from sky import exceptions
from sky import sky_logging
from sky.jobs import state as managed_job_state
from sky.jobs.server import core as managed_jobs_core
from sky.server.requests import request_names
from sky.server.requests import requests as api_requests
from sky.utils import common
from sky.utils import common_utils
from sky.utils import locks
from sky.utils import status_lib
from sky.utils import ux_utils
from sky.utils import yaml_utils

logger = sky_logging.init_logger(__name__)

DEFAULT_QUOTA_FILE = '~/.sky/tpu_quota.yaml'
QUOTA_FILE_ENV_VAR = 'SKYPILOT_TPU_QUOTA_FILE'
RETRY_WAIT_SECONDS = 60
ADMISSION_LOCK_ID = 'tpu_quota_admission'
ADMISSION_LOCK_TIMEOUT_SECONDS = 30

UNIT_CHIPS = 'chips'
UNIT_SLICES = 'slices'

_GATED_REQUESTS = (
    request_names.AdminPolicyRequestName.CLUSTER_LAUNCH,
    request_names.AdminPolicyRequestName.JOBS_LAUNCH,
)
# TPU families whose accelerator name counts TensorCores, two per chip.
_CORE_COUNTED_FAMILIES = ('v2', 'v3', 'v4', 'v5p')
_TPU_NAME_RE = re.compile(r'^tpu-([a-z0-9]+)-(\d+)$')
# Accelerator demands inside a managed job's resources string, which looks
# like '1x[tpu-v5p-8:1][Spot]'.
_JOB_RESOURCES_RE = re.compile(r'^(\d+)x\[(.*?)\]')
_ACC_DEMAND_RE = re.compile(r'(tpu-[a-z0-9]+-\d+):(\d+)')
# A parked request's reason starts with this marker, so that other requests
# can see which family it waits for and whether it is within its share. The
# executor keeps the first 200 characters of the reason in the status message.
_WAITING_MARKER_RE = re.compile(r'\[tpu-quota (\S+) (under|over)-share\]')
# An admitted request's status message starts with this marker, followed by
# the JSON reservation (see ``_record_admission``). The executor clears the
# status message when a request starts running, so a request that is still
# waiting for the admission lock carries no reservation.
_ADMITTED_MARKER = '[tpu-quota admitted]'

Usage = Dict[str, int]


@dataclasses.dataclass
class Quota:
    """The limits read from the quota file."""
    unit: str
    pool: Dict[str, int]
    per_user: Dict[str, Dict[str, int]]

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'Quota':
        unit = config.get('unit', UNIT_CHIPS)
        if unit not in (UNIT_CHIPS, UNIT_SLICES):
            raise ValueError(f'unit must be {UNIT_CHIPS!r} or {UNIT_SLICES!r}, '
                             f'got {unit!r}.')
        pool = {str(k): int(v) for k, v in (config.get('pool') or {}).items()}
        per_user = {
            str(user): {str(k): int(v) for k, v in (limits or {}).items()}
            for user, limits in (config.get('per_user') or {}).items()
        }
        return cls(unit=unit, pool=pool, per_user=per_user)

    def user_limit(self, user: str, family: str) -> int:
        """The user's share of a family, falling back to default and pool."""
        for limits in (self.per_user.get(user), self.per_user.get('default')):
            if limits is not None and family in limits:
                return limits[family]
        return self.pool[family]


def quota_file_path() -> str:
    return os.path.expanduser(
        os.environ.get(QUOTA_FILE_ENV_VAR, DEFAULT_QUOTA_FILE))


def load_quota(path: Optional[str] = None) -> Optional[Quota]:
    """Reads the quota file. Returns None when it is missing or malformed.

    A missing or malformed file disables the quota rather than blocking every
    launch, and logs a warning on the API server.
    """
    path = path or quota_file_path()
    if not os.path.exists(path):
        logger.warning(f'TPU quota file {path} does not exist; '
                       'admitting all requests.')
        return None
    try:
        config = yaml_utils.read_yaml(path)
        return Quota.from_dict(config or {})
    except Exception as e:  # pylint: disable=broad-except
        logger.warning(f'Failed to read TPU quota file {path}: {e}; '
                       'admitting all requests.')
        return None


def tpu_family(acc_name: str) -> Optional[str]:
    """'tpu-v5p-8' -> 'v5p'. None for anything that is not a TPU slice."""
    match = _TPU_NAME_RE.match(acc_name)
    if match is None:
        return None
    return match.group(1)


def tpu_amount(acc_name: str, unit: str) -> int:
    """The number of chips (or slices) in one accelerator of this name."""
    if unit == UNIT_SLICES:
        return 1
    match = _TPU_NAME_RE.match(acc_name)
    assert match is not None, acc_name
    family, size = match.group(1), int(match.group(2))
    if family in _CORE_COUNTED_FAMILIES:
        return max(1, size // 2)
    return size


def _add_accelerators(usage: Usage, accelerators: Dict[str, Any],
                      num_nodes: int, unit: str) -> None:
    for acc_name, count in accelerators.items():
        family = tpu_family(str(acc_name))
        if family is None:
            continue
        amount = tpu_amount(str(acc_name), unit) * int(count) * num_nodes
        usage[family] = usage.get(family, 0) + amount


def task_demand(task: Any, unit: str) -> Usage:
    """The TPU demand of a task, per family.

    With several candidate resources (``any_of`` or ``ordered``) the policy
    runs before the optimizer picks one, so the largest candidate counts.
    """
    demand: Usage = {}
    for resources in task.resources:
        candidate: Usage = {}
        if resources.accelerators:
            _add_accelerators(candidate, resources.accelerators, task.num_nodes,
                              unit)
        for family, amount in candidate.items():
            demand[family] = max(demand.get(family, 0), amount)
    return demand


def _cluster_records() -> List[Any]:
    # Managed job clusters are excluded by default; the jobs queue counts them.
    return core.status(refresh=common.StatusRefreshMode.NONE, all_users=True)


def _job_records() -> List[Dict[str, Any]]:
    return managed_jobs_core.queue(refresh=False,
                                   skip_finished=True,
                                   all_users=True)


def _cluster_accelerators(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    handle = record.get('handle')
    launched = getattr(handle, 'launched_resources', None)
    if launched is not None:
        return launched.accelerators
    accelerators = record.get('accelerators')
    if isinstance(accelerators, str):
        try:
            return ast.literal_eval(accelerators)
        except (ValueError, SyntaxError):
            return None
    return accelerators


def current_usage(unit: str) -> Tuple[Dict[str, Usage], Usage]:
    """TPU usage per user and in total.

    Counts the clusters and managed jobs of all users, plus the reservations
    of admitted launch requests whose cluster or job is not visible yet.
    """
    by_user: Dict[str, Usage] = {}
    total: Usage = {}
    visible_clusters: Set[str] = set()
    visible_jobs: Set[str] = set()

    def _account(user: Optional[str], accelerators: Dict[str, Any],
                 num_nodes: int) -> None:
        user_usage = by_user.setdefault(user or '', {})
        _add_accelerators(user_usage, accelerators, num_nodes, unit)
        _add_accelerators(total, accelerators, num_nodes, unit)

    for record in _cluster_records():
        status = record.get('status')
        if status not in (status_lib.ClusterStatus.UP,
                          status_lib.ClusterStatus.INIT):
            continue
        visible_clusters.add(str(record.get('name')))
        accelerators = _cluster_accelerators(record)
        if not accelerators:
            continue
        _account(record.get('user_name'), accelerators,
                 int(record.get('num_nodes') or 1))

    for record in _job_records():
        status = record.get('status')
        if isinstance(
                status,
                managed_job_state.ManagedJobStatus) and status.is_terminal():
            continue
        visible_jobs.add(str(record.get('job_name')))
        accelerators = record.get('accelerators')
        num_nodes = 1
        if not accelerators:
            match = _JOB_RESOURCES_RE.match(str(record.get('resources') or ''))
            if match is None:
                continue
            num_nodes = int(match.group(1))
            accelerators = {
                name: int(count)
                for name, count in _ACC_DEMAND_RE.findall(match.group(2))
            }
        if not accelerators:
            continue
        _account(record.get('user_name'), accelerators, num_nodes)

    for reservation in pending_reservations(visible_clusters, visible_jobs):
        if reservation.get('unit') != unit:
            logger.warning(f'Ignoring a TPU reservation in {reservation!r}: '
                           f'the quota unit is now {unit}.')
            continue
        user_usage = by_user.setdefault(str(reservation.get('user') or ''), {})
        for family, amount in reservation.get('demand', {}).items():
            user_usage[family] = user_usage.get(family, 0) + int(amount)
            total[family] = total.get(family, 0) + int(amount)
    return by_user, total


def _gated_requests_with_status(
        status: api_requests.RequestStatus) -> List[api_requests.Request]:
    return api_requests.get_request_tasks(
        api_requests.RequestTaskFilter(
            status=[status],
            include_request_names=[name.value for name in _GATED_REQUESTS]))


def _parked_requests() -> List[api_requests.Request]:
    return _gated_requests_with_status(api_requests.RequestStatus.WAITING)


def _running_requests() -> List[api_requests.Request]:
    return _gated_requests_with_status(api_requests.RequestStatus.RUNNING)


def _record_admission(user: str, unit: str, demand: Usage,
                      cluster_name: Optional[str],
                      job_name: Optional[str]) -> None:
    """Stores the admitted demand on the current request.

    The reservation is counted by ``current_usage`` until the launched
    cluster (or managed job, matched by name) is visible, or until the
    request leaves the RUNNING status. Outside of a request context (unit
    tests, in-process launches) there is no request row to write to.
    """
    if not common_utils.is_in_request_context():
        return
    reservation = {
        'user': user,
        'unit': unit,
        'demand': demand,
        'cluster': cluster_name,
        'job': job_name,
    }
    with api_requests.update_request(
            common_utils.get_current_request_id()) as request:
        if request is None:
            return
        request.status_msg = f'{_ADMITTED_MARKER} {json.dumps(reservation)}'


def pending_reservations(visible_clusters: Set[str],
                         visible_jobs: Set[str]) -> List[Dict[str, Any]]:
    """Reservations of admitted requests that are not yet visible as usage.

    A reservation whose cluster or job name is visible is dropped: the
    cluster or job records already count it.
    """
    current_id = common_utils.get_current_request_id()
    reservations = []
    for request in _running_requests():
        if request.request_id == current_id:
            continue
        status_msg = request.status_msg or ''
        if not status_msg.startswith(_ADMITTED_MARKER):
            continue
        try:
            reservation = json.loads(status_msg[len(_ADMITTED_MARKER):])
        except ValueError:
            logger.warning(f'Request {request.request_id} carries a '
                           f'malformed TPU reservation: {status_msg!r}')
            continue
        if reservation.get('cluster') in visible_clusters:
            continue
        if reservation.get('job') in visible_jobs:
            continue
        reservations.append(reservation)
    return reservations


def _admission_lock() -> locks.DistributedLock:
    return locks.get_lock(ADMISSION_LOCK_ID,
                          timeout=ADMISSION_LOCK_TIMEOUT_SECONDS)


def under_share_waiters(family: str, user_id: str) -> List[str]:
    """IDs of parked requests of other users within their share of family."""
    waiters = []
    for request in _parked_requests():
        if request.user_id == user_id:
            continue
        match = _WAITING_MARKER_RE.search(request.status_msg or '')
        if match is None:
            continue
        if match.group(1) == family and match.group(2) == 'under':
            waiters.append(request.request_id)
    return waiters


def _waiting_marker(family: str, over_share: bool) -> str:
    return f'[tpu-quota {family} {"over" if over_share else "under"}-share]'


class TPUQuotaPolicy(admin_policy.AdminPolicy):
    """Parks TPU launches that exceed the user's share while the pool is full.

    See the module docstring for the quota file format and the semantics.
    """

    @classmethod
    def validate_and_mutate(
        cls, user_request: admin_policy.UserRequest
    ) -> admin_policy.MutatedUserRequest:
        passthrough = admin_policy.MutatedUserRequest(
            task=user_request.task,
            skypilot_config=user_request.skypilot_config)
        if (user_request.at_client_side or
                user_request.request_name not in _GATED_REQUESTS):
            return passthrough
        quota = load_quota()
        if quota is None:
            return passthrough
        demand = {
            family: amount for family, amount in task_demand(
                user_request.task, quota.unit).items() if family in quota.pool
        }
        if not demand:
            return passthrough

        assert user_request.user is not None, (
            'Failed to get the user initiating the request.')
        try:
            with _admission_lock():
                cls._admit(user_request, quota, demand)
        except locks.LockTimeout:
            cls._park('Waiting for the TPU quota admission lock.')
        return passthrough

    @classmethod
    def _admit(cls, user_request: admin_policy.UserRequest, quota: Quota,
               demand: Usage) -> None:
        """Parks the request unless it fits; records the admission otherwise.

        Runs under the admission lock, so the usage read here cannot change
        before the reservation is recorded.
        """
        assert user_request.user is not None
        user = user_request.user.name or ''
        by_user, total = current_usage(quota.unit)
        for family, amount in demand.items():
            limit = quota.user_limit(user, family)
            pool = quota.pool[family]
            mine = by_user.get(user, {}).get(family, 0)
            used = total.get(family, 0)
            if amount > pool:
                with ux_utils.print_exception_no_traceback():
                    raise ValueError(
                        f'The request needs {amount} {family} {quota.unit}, '
                        f'more than the pool of {pool}. It can never be '
                        'admitted.')
            over_share = mine + amount > limit
            marker = _waiting_marker(family, over_share)
            if used + amount > pool:
                cls._park(f'{marker} {user} is at {mine}/{limit} {family} '
                          f'{quota.unit} and the pool is full ({used}/{pool}). '
                          f'Waiting for {amount} {quota.unit} to free up.')
            if over_share:
                waiters = under_share_waiters(family, user_request.user.id)
                if waiters:
                    cls._park(f'{marker} {user} is at {mine}/{limit} {family} '
                              f'{quota.unit}; {len(waiters)} request(s) of '
                              'users within their share wait for the same '
                              'family. Yielding to them.')
                logger.info(f'User {user} borrows {amount} {family} '
                            f'{quota.unit}: {mine} + {amount} > share {limit}, '
                            f'pool {used} + {amount} <= {pool}.')
        cluster_name = None
        if user_request.request_options is not None:
            cluster_name = user_request.request_options.cluster_name
        _record_admission(user, quota.unit, demand, cluster_name,
                          user_request.task.name)

    @staticmethod
    def _park(message: str) -> None:
        logger.info(message)
        raise exceptions.ExecutionPausedError(
            message,
            hint='Cancel the request with `sky api cancel` to stop waiting.',
            retry_wait_seconds=RETRY_WAIT_SECONDS)
