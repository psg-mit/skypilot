"""Managed Instance Group Utils"""
import re
import subprocess
from typing import Any, Dict, Optional

from sky import sky_logging
from sky.adaptors import gcp
from sky.provision.gcp import constants

logger = sky_logging.init_logger(__name__)

MIG_RESOURCE_NOT_FOUND_PATTERN = re.compile(
    r'The resource \'projects/.*/zones/.*/instanceGroupManagers/.*\' was not '
    r'found')

IT_RESOURCE_NOT_FOUND_PATTERN = re.compile(
    r'The resource \'projects/.*/regions/.*/instanceTemplates/.*\' was not '
    'found')


def get_instance_template_name(cluster_name: str) -> str:
    return f'{constants.INSTANCE_TEMPLATE_NAME_PREFIX}{cluster_name}'


def get_managed_instance_group_name(cluster_name: str) -> str:
    return f'{constants.MIG_NAME_PREFIX}{cluster_name}'


def get_workload_policy_name(cluster_name: str) -> str:
    return f'{constants.WORKLOAD_POLICY_NAME_PREFIX}{cluster_name}'


def _is_not_found(e: Exception) -> bool:
    resp = getattr(e, 'resp', None)
    return resp is not None and getattr(resp, 'status', None) == 404


def get_workload_policy(project_id: str, region: str,
                        policy_name: str) -> Optional[Dict[str, Any]]:
    """Returns the workload policy, or None when it does not exist."""
    compute = gcp.build('compute',
                        'v1',
                        credentials=None,
                        cache_discovery=False)
    try:
        return compute.resourcePolicies().get(
            project=project_id, region=region,
            resourcePolicy=policy_name).execute()
    except gcp.http_error_exception() as e:
        if _is_not_found(e):
            return None
        raise


def create_workload_policy(project_id: str, region: str, policy_name: str,
                           topology: str) -> dict:
    """Creates a workload policy that forms a TPU slice of this topology."""
    logger.debug(f'Creating workload policy {policy_name!r} with accelerator '
                 f'topology {topology}.')
    compute = gcp.build('compute',
                        'v1',
                        credentials=None,
                        cache_discovery=False)
    return compute.resourcePolicies().insert(
        project=project_id,
        region=region,
        body={
            'name': policy_name,
            'description': 'SkyPilot workload policy for a multi-host TPU '
                           'slice.',
            'workloadPolicy': {
                'type': 'HIGH_THROUGHPUT',
                'acceleratorTopology': topology,
            },
        }).execute()


def delete_workload_policy(project_id: str, region: str,
                           policy_name: str) -> Optional[dict]:
    """Deletes the workload policy. Returns None when it does not exist."""
    logger.debug(f'Deleting workload policy {policy_name!r}.')
    compute = gcp.build('compute',
                        'v1',
                        credentials=None,
                        cache_discovery=False)
    try:
        return compute.resourcePolicies().delete(
            project=project_id, region=region,
            resourcePolicy=policy_name).execute()
    except gcp.http_error_exception() as e:
        if _is_not_found(e):
            return None
        raise


def check_instance_template_exits(project_id: str, region: str,
                                  template_name: str) -> bool:
    compute = gcp.build('compute',
                        'v1',
                        credentials=None,
                        cache_discovery=False)
    try:
        compute.regionInstanceTemplates().get(
            project=project_id, region=region,
            instanceTemplate=template_name).execute()
    except gcp.http_error_exception() as e:
        if IT_RESOURCE_NOT_FOUND_PATTERN.search(str(e)) is not None:
            # Instance template does not exist.
            return False
        raise
    return True


def create_region_instance_template(cluster_name_on_cloud: str,
                                    project_id: str,
                                    region: str,
                                    template_name: str,
                                    node_config: Dict[str, Any],
                                    for_tpu_slice: bool = False) -> dict:
    """Create a regional instance template.

    The template of a DWS group drops spot scheduling and reservations. The
    template of a multi-host TPU slice keeps the node config as it is, so
    spot slices and reservations work.
    """
    logger.debug(f'Creating regional instance template {template_name!r}.')
    compute = gcp.build('compute',
                        'v1',
                        credentials=None,
                        cache_discovery=False)
    config = node_config.copy()
    config.pop(constants.MANAGED_INSTANCE_GROUP_CONFIG, None)
    config.pop(constants.TPU_SLICE_CONFIG, None)
    if for_tpu_slice:
        operation = compute.regionInstanceTemplates().insert(
            project=project_id,
            region=region,
            body={
                'name': template_name,
                'properties': dict(
                    description=(
                        'SkyPilot instance template for '
                        f'{cluster_name_on_cloud!r}, a multi-host TPU slice.'),
                    **config,
                )
            }).execute()
        return operation

    # We have to ignore user defined scheduling for DWS.
    # TODO: Add a warning log for this behvaiour.
    scheduling = config.get('scheduling', {})
    assert scheduling.get('provisioningModel') != 'SPOT', (
        'DWS does not support spot VMs.')

    reservations_affinity = config.pop('reservation_affinity', None)
    if reservations_affinity is not None:
        logger.warning(
            f'Ignoring reservations_affinity {reservations_affinity} '
            'for DWS.')

    # Create the regional instance template request
    operation = compute.regionInstanceTemplates().insert(
        project=project_id,
        region=region,
        body={
            'name': template_name,
            'properties': dict(
                description=(
                    'SkyPilot instance template for '
                    f'{cluster_name_on_cloud!r} to support DWS requests.'),
                reservationAffinity=dict(
                    consumeReservationType='NO_RESERVATION'),
                **config,
            )
        }).execute()
    return operation


def create_managed_instance_group(
        project_id: str,
        zone: str,
        group_name: str,
        instance_template_url: str,
        size: int,
        workload_policy_url: Optional[str] = None) -> dict:
    """Creates a managed instance group.

    With a workload policy, the group is created in bulk: Compute Engine
    creates all of its VMs at once or none of them, as a multi-host TPU slice
    requires.
    """
    logger.debug(f'Creating managed instance group {group_name!r}.')
    compute = gcp.build('compute',
                        'v1',
                        credentials=None,
                        cache_discovery=False)
    body: Dict[str, Any] = {
        'name': group_name,
        'instanceTemplate': instance_template_url,
        'target_size': size,
        'instanceLifecyclePolicy': {
            'defaultActionOnFailure': 'DO_NOTHING',
        },
        'updatePolicy': {
            'type': 'OPPORTUNISTIC',
        },
    }
    if workload_policy_url is not None:
        body['targetSizePolicy'] = {'mode': 'BULK'}
        body['resourcePolicies'] = {'workloadPolicy': workload_policy_url}
    operation = compute.instanceGroupManagers().insert(project=project_id,
                                                       zone=zone,
                                                       body=body).execute()
    return operation


def resize_managed_instance_group(project_id: str, zone: str, group_name: str,
                                  resize_by: int, run_duration: int) -> dict:
    logger.debug(f'Resizing managed instance group {group_name!r} by '
                 f'{resize_by} with run duration {run_duration}.')
    compute = gcp.build('compute',
                        'beta',
                        credentials=None,
                        cache_discovery=False)
    operation = compute.instanceGroupManagerResizeRequests().insert(
        project=project_id,
        zone=zone,
        instanceGroupManager=group_name,
        body={
            'name': group_name,
            'resizeBy': resize_by,
            'requestedRunDuration': {
                'seconds': run_duration,
            }
        }).execute()
    return operation


def cancel_all_resize_request_for_mig(project_id: str, zone: str,
                                      group_name: str) -> None:
    logger.debug(f'Cancelling all resize requests for MIG {group_name!r}.')
    try:
        compute = gcp.build('compute',
                            'beta',
                            credentials=None,
                            cache_discovery=False)
        operation = compute.instanceGroupManagerResizeRequests().list(
            project=project_id,
            zone=zone,
            instanceGroupManager=group_name,
            filter='state eq ACCEPTED').execute()
        for request in operation.get('items', []):
            try:
                compute.instanceGroupManagerResizeRequests().cancel(
                    project=project_id,
                    zone=zone,
                    instanceGroupManager=group_name,
                    resizeRequest=request['name']).execute()
            except gcp.http_error_exception() as e:
                logger.warning('Failed to cancel resize request '
                               f'{request["id"]!r}: {e}')
    except gcp.http_error_exception() as e:
        if re.search(MIG_RESOURCE_NOT_FOUND_PATTERN, str(e)) is None:
            raise
        logger.warning(f'MIG {group_name!r} does not exist. Skip '
                       'resize request cancellation.')
        logger.debug(f'Error: {e}')


def check_managed_instance_group_exists(project_id: str, zone: str,
                                        group_name: str) -> bool:
    compute = gcp.build('compute',
                        'v1',
                        credentials=None,
                        cache_discovery=False)
    try:
        compute.instanceGroupManagers().get(
            project=project_id, zone=zone,
            instanceGroupManager=group_name).execute()
    except gcp.http_error_exception() as e:
        if MIG_RESOURCE_NOT_FOUND_PATTERN.search(str(e)) is not None:
            return False
        raise
    return True


def wait_for_managed_group_to_be_stable(project_id: str, zone: str,
                                        group_name: str, timeout: int) -> None:
    """Wait until the managed instance group is stable."""
    logger.debug(f'Waiting for MIG {group_name} to be stable with timeout '
                 f'{timeout}.')
    try:
        cmd = ('gcloud compute instance-groups managed wait-until '
               f'{group_name} '
               '--stable '
               f'--zone={zone} '
               f'--project={project_id} '
               f'--timeout={timeout}')
        logger.info(
            f'Waiting for MIG {group_name} to be stable with command:\n{cmd}')
        proc = subprocess.run(
            f'yes | {cmd}',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            check=True,
        )
        stdout = proc.stdout.decode('ascii')
        logger.info(stdout)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode('ascii')
        logger.info(stderr)
        raise
