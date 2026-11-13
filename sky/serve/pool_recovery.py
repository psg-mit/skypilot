"""Reconnect persistent pool controllers to their existing launch requests."""
from typing import Optional

from sky import global_user_state
from sky.adaptors import common as adaptors_common
from sky.jobs import utils as managed_job_utils
from sky.provision.gcp import queued_resource_wait
from sky.serve import constants as serve_constants
from sky.serve import serve_state
from sky.server.requests import payloads
from sky.server.requests import restart_recovery
from sky.usage import constants as usage_constants
from sky.utils import common_utils
from sky.utils import dag_utils

api_requests = adaptors_common.LazyImport('sky.server.requests.requests')


class PoolLaunchRecoveryError(ValueError):
    """An existing pool launch cannot safely be matched to this worker."""


def _binding(body: payloads.LaunchBody):
    value = body.model_dump(mode='json')
    # Server annotations and CLI presentation are not worker inputs.
    for key in ('client_api_version', 'workspace_access', 'entrypoint',
                'entrypoint_command', 'override_skypilot_config_path'):
        value.pop(key, None)
    value['task'] = [
        task.to_yaml_config()
        for task in dag_utils.load_chain_dag_from_yaml_str(body.task).tasks
    ]
    value['env_vars'] = dict(value['env_vars'])
    # These describe the API/controller process, not the worker. All worker
    # resources, credentials, workspace, config content and other envs match.
    for key in (usage_constants.USAGE_RUN_ID_ENV_VAR, restart_recovery.ENV_VAR,
                'SKYPILOT_POD_MEMORY_GB_LIMIT'):
        value['env_vars'].pop(key, None)
    return value


def equivalent_worker_launches(first: payloads.LaunchBody,
                               second: payloads.LaunchBody) -> bool:
    """Compare immutable inputs, allowing native names and replica env IDs."""
    bindings = [_binding(first), _binding(second)]
    for binding in bindings:
        binding.pop('cluster_name', None)
        for task in binding['task']:
            # This duplicates the original YAML for display; the parsed task
            # below retains its effective source, setup, resources and envs.
            task.pop('_user_specified_yaml', None)
            (task.get('envs') or {}).pop(serve_constants.REPLICA_ID_ENV_VAR,
                                         None)
    return bindings[0] == bindings[1]


def _find_existing_launch(body: payloads.LaunchBody) -> Optional[str]:
    """Find the oldest equivalent request, without creating or changing one.

    Enabled only where the pool controller and API share one persistent native
    backend. Older controllers may already have submitted equivalent requests;
    retain every identity and attach deterministically to the original. An
    active request with different ownership or inputs prevents a new launch.
    """
    if (not restart_recovery.enabled() or
            not managed_job_utils.is_consolidation_mode() or
            not body.is_launched_by_sky_serve_controller or
            body.is_launched_by_jobs_controller):
        return None
    service_name, separator, replica_id = body.cluster_name.rpartition('-')
    if not separator or not replica_id.isdecimal():
        return None
    service = serve_state.get_service_from_name(service_name)
    if service is None or not service['pool']:
        return None
    owner = service['user_hash']
    replica = serve_state.get_replica_info_from_id(service_name,
                                                   int(replica_id))
    if (not owner or owner != common_utils.get_user_hash() or replica is None or
            replica.cluster_name != body.cluster_name):
        raise PoolLaunchRecoveryError('Pool worker ownership is unknown')
    candidates = api_requests.get_request_tasks(
        api_requests.RequestTaskFilter(
            cluster_names=[body.cluster_name],
            include_request_names=['sky.launch'],
            status=api_requests.RequestStatus.active_statuses() +
            [api_requests.RequestStatus.SUCCEEDED]))
    expected = _binding(body)
    matched = []
    for request in candidates:
        succeeded = request.status == api_requests.RequestStatus.SUCCEEDED
        if succeeded:
            # Never attach to an earlier successful incarnation of this name.
            current = global_user_state.get_cluster_hash(body.cluster_name)
            journals = (queued_resource_wait.load(p)
                        for p in queued_resource_wait.JOURNAL_ROOT.expanduser().
                        glob('*.json'))
            if not current or not any(
                    v['request_id'] == request.request_id and v.get('stage') ==
                    'ready' and v.get('native_cluster_hash') == current
                    for v in journals):
                continue
        if (request.user_id != owner or
                request.cluster_name != body.cluster_name or
                not isinstance(request.request_body, payloads.LaunchBody) or
                _binding(request.request_body) != expected):
            if succeeded:
                continue
            raise PoolLaunchRecoveryError(
                'Existing pool launch has different ownership or inputs')
        matched.append(request)
    if not matched:
        return None
    original = min(matched, key=lambda r: (r.created_at, r.request_id))
    return original.request_id


def find_existing_launch(body: payloads.LaunchBody) -> Optional[str]:
    """Block launch and cleanup when the native observation fails."""
    try:
        return _find_existing_launch(body)
    except PoolLaunchRecoveryError:
        raise
    except Exception as error:
        raise PoolLaunchRecoveryError(
            'Pool launch recovery observation failed: ' +
            type(error).__name__) from error
