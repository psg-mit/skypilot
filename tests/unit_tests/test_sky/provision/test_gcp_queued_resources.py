"""Cloud TPU queued resources must not outlive cluster termination."""
# pylint: disable=protected-access,redefined-outer-name
import copy
import json
from unittest import mock

from googleapiclient import errors
import httplib2
import pytest

from sky.provision import common
from sky.provision import constants as provision_constants
from sky.provision.gcp import instance
from sky.provision.gcp import instance_utils

PROJECT = 'test-project'
ZONE = 'us-central2-b'
PARENT = f'projects/{PROJECT}/locations/{ZONE}'
CLUSTER = 'test-cluster'
LABELS = {provision_constants.TAG_RAY_CLUSTER_NAME: CLUSTER}
PROVIDER_CONFIG = {
    'project_id': PROJECT,
    'availability_zone': ZONE,
    '_has_tpus': True,
}


def _queued_resource(name='request',
                     cluster=CLUSTER,
                     kind='head',
                     state='ACTIVE'):
    return {
        'name': f'{PARENT}/queuedResources/{name}',
        'state': {
            'state': state
        },
        'tpu': {
            'nodeSpec': [{
                'parent': PARENT,
                'nodeId': f'{cluster}-{kind}-tpu',
                'node': {
                    'labels': {
                        provision_constants.TAG_RAY_CLUSTER_NAME: cluster,
                        provision_constants.TAG_RAY_NODE_KIND: kind,
                    },
                },
            }],
        },
    }


def _http_error(status):
    return errors.HttpError(
        httplib2.Response({'status': str(status)}),
        json.dumps({
            'error': {
                'message': 'test error'
            }
        }).encode())


@pytest.fixture
def tpu_api(monkeypatch):
    api = mock.MagicMock()
    locations = api.projects.return_value.locations.return_value
    queued = locations.queuedResources.return_value
    locations.nodes.return_value.list.return_value.execute.return_value = {
        'nodes': [],
    }
    queued.list.return_value.execute.return_value = {'queuedResources': []}
    queued.list_next.return_value = None
    queued.delete.return_value.execute.return_value = {
        'name': f'{PARENT}/operations/delete-request',
    }
    monkeypatch.setattr(instance_utils.GCPTPUVMInstance, 'load_resource',
                        lambda: api)
    return api


@pytest.fixture
def wait_for_operation(monkeypatch):
    wait = mock.MagicMock()
    monkeypatch.setattr(instance_utils.GCPTPUVMInstance, 'wait_for_operation',
                        wait)
    return wait


def test_filter_queued_resources_matches_all_node_labels_and_all_pages(tpu_api):
    queued = tpu_api.projects().locations().queuedResources()
    owned = _queued_resource()
    foreign = _queued_resource('foreign', cluster=f'{CLUSTER}-other')
    unlabeled = _queued_resource('unlabeled')
    unlabeled['tpu']['nodeSpec'][0]['node'].pop('labels')
    mixed = _queued_resource('mixed')
    mixed['tpu']['nodeSpec'].extend(foreign['tpu']['nodeSpec'])
    empty = {'name': f'{PARENT}/queuedResources/empty', 'tpu': {}}
    first_page = {
        'queuedResources': [foreign, unlabeled, mixed, empty],
        'nextPageToken': 'next',
    }
    second_page = {'queuedResources': [owned]}
    queued.list.return_value.execute.return_value = first_page
    next_request = mock.MagicMock()
    next_request.execute.return_value = second_page
    queued.list_next.side_effect = [next_request, None]

    result = instance_utils.GCPTPUVMInstance.filter_queued_resources(
        PROJECT, ZONE, LABELS)

    assert result == {owned['name']: owned}
    queued.list.assert_called_once_with(parent=PARENT)
    queued.list_next.assert_has_calls([
        mock.call(queued.list.return_value, first_page),
        mock.call(next_request, second_page),
    ])


def test_filter_queued_resources_requires_cluster_scope(tpu_api):
    with pytest.raises(ValueError, match='cluster'):
        instance_utils.GCPTPUVMInstance.filter_queued_resources(
            PROJECT, ZONE, {})
    tpu_api.projects.assert_not_called()


def test_filter_queued_resources_rejects_incomplete_inventory(tpu_api):
    queued = tpu_api.projects().locations().queuedResources()
    queued.list.return_value.execute.return_value = {
        'queuedResources': [_queued_resource()],
        'unreachable': [ZONE],
    }
    with pytest.raises(common.ProvisionerError, match='unreachable'):
        instance_utils.GCPTPUVMInstance.terminate_queued_resources(
            PROJECT, ZONE, LABELS)
    queued.delete.assert_not_called()


@pytest.mark.parametrize('state', ['ACTIVE', 'ACCEPTED', 'FAILED', 'SUSPENDED'])
def test_terminate_queued_resource_and_backing_nodes(tpu_api,
                                                     wait_for_operation, state):
    queued = tpu_api.projects().locations().queuedResources()
    resource = _queued_resource(state=state)
    queued.list.return_value.execute.return_value = {
        'queuedResources': [resource],
    }

    instance_utils.GCPTPUVMInstance.terminate_queued_resources(
        PROJECT, ZONE, LABELS)

    queued.delete.assert_called_once_with(name=resource['name'], force=True)
    queued.delete.return_value.execute.assert_called_once_with(
        num_retries=instance_utils.GCP_MAX_RETRIES)
    wait_for_operation.assert_called_once_with(
        queued.delete.return_value.execute.return_value, PROJECT, zone=ZONE)
    tpu_api.projects().locations().nodes().delete.assert_not_called()


def test_repeated_queued_resource_cleanup(tpu_api, wait_for_operation):
    queued = tpu_api.projects().locations().queuedResources()
    resource = _queued_resource(state='SUSPENDED')
    queued.list.return_value.execute.side_effect = [{
        'queuedResources': [resource]
    }, {
        'queuedResources': []
    }]

    for _ in range(2):
        instance_utils.GCPTPUVMInstance.terminate_queued_resources(
            PROJECT, ZONE, LABELS)

    queued.delete.assert_called_once_with(name=resource['name'], force=True)
    assert wait_for_operation.call_count == 1


def test_terminate_queued_resources_handles_concurrent_deletion(
        tpu_api, wait_for_operation):
    queued = tpu_api.projects().locations().queuedResources()
    queued.list.return_value.execute.return_value = {
        'queuedResources': [_queued_resource()],
    }
    queued.delete.return_value.execute.side_effect = _http_error(404)

    instance_utils.GCPTPUVMInstance.terminate_queued_resources(
        PROJECT, ZONE, LABELS)

    wait_for_operation.assert_not_called()


@pytest.mark.parametrize('status', [403, 409, 500])
def test_terminate_queued_resources_propagates_delete_errors(
        tpu_api, wait_for_operation, status):
    queued = tpu_api.projects().locations().queuedResources()
    queued.list.return_value.execute.return_value = {
        'queuedResources': [_queued_resource()],
    }
    queued.delete.return_value.execute.side_effect = _http_error(status)

    with pytest.raises(errors.HttpError):
        instance_utils.GCPTPUVMInstance.terminate_queued_resources(
            PROJECT, ZONE, LABELS)
    wait_for_operation.assert_not_called()


def test_terminate_queued_resources_propagates_list_errors(tpu_api):
    queued = tpu_api.projects().locations().queuedResources()
    queued.list.return_value.execute.side_effect = _http_error(403)

    with pytest.raises(errors.HttpError):
        instance_utils.GCPTPUVMInstance.terminate_queued_resources(
            PROJECT, ZONE, LABELS)
    queued.delete.assert_not_called()


def test_terminate_queued_resources_propagates_operation_errors(
        tpu_api, wait_for_operation):
    queued = tpu_api.projects().locations().queuedResources()
    queued.list.return_value.execute.return_value = {
        'queuedResources': [_queued_resource()],
    }
    wait_for_operation.side_effect = common.ProvisionerError('operation failed')

    with pytest.raises(common.ProvisionerError, match='operation failed'):
        instance_utils.GCPTPUVMInstance.terminate_queued_resources(
            PROJECT, ZONE, LABELS)


@pytest.mark.parametrize('state', ['ACCEPTED', 'FAILED', 'SUSPENDED'])
def test_terminate_cluster_cleans_requests_without_nodes(
        tpu_api, wait_for_operation, monkeypatch, state):
    queued = tpu_api.projects().locations().queuedResources()
    resource = _queued_resource(state=state)
    queued.list.return_value.execute.return_value = {
        'queuedResources': [resource],
    }
    filter_instances = mock.MagicMock(return_value={})
    monkeypatch.setattr(instance, '_filter_instances', filter_instances)

    instance.terminate_instances(CLUSTER, PROVIDER_CONFIG)

    queued.delete.assert_called_once_with(name=resource['name'], force=True)
    assert wait_for_operation.call_count == 1
    filter_instances.assert_called_once()


def test_worker_only_termination_preserves_head_and_foreign_requests(
        tpu_api, wait_for_operation, monkeypatch):
    queued = tpu_api.projects().locations().queuedResources()
    worker = _queued_resource('worker', kind='worker')
    head = _queued_resource('head')
    foreign = _queued_resource('foreign', cluster='other', kind='worker')
    mixed = copy.deepcopy(worker)
    mixed['name'] = f'{PARENT}/queuedResources/mixed'
    mixed['tpu']['nodeSpec'].extend(head['tpu']['nodeSpec'])
    queued.list.return_value.execute.return_value = {
        'queuedResources': [worker, head, foreign, mixed],
    }
    monkeypatch.setattr(instance, '_filter_instances',
                        mock.MagicMock(return_value={}))

    instance.terminate_instances(CLUSTER, PROVIDER_CONFIG, worker_only=True)

    queued.delete.assert_called_once_with(name=worker['name'], force=True)
    assert wait_for_operation.call_count == 1


def test_worker_only_termination_preserves_promoted_head(
        tpu_api, wait_for_operation, monkeypatch):
    locations = tpu_api.projects().locations()
    queued = locations.queuedResources()
    promoted = _queued_resource('promoted', kind='worker')
    queued.list.return_value.execute.return_value = {
        'queuedResources': [promoted],
    }
    spec = promoted['tpu']['nodeSpec'][0]
    locations.nodes().list.return_value.execute.return_value = {
        'nodes': [{
            'name': f'{PARENT}/nodes/{spec["nodeId"]}',
            'labels': dict(LABELS,
                           **{provision_constants.TAG_RAY_NODE_KIND: 'head'}),
        }],
    }
    monkeypatch.setattr(instance, '_filter_instances',
                        mock.MagicMock(return_value={}))

    instance.terminate_instances(CLUSTER, PROVIDER_CONFIG, worker_only=True)

    queued.delete.assert_not_called()
    wait_for_operation.assert_not_called()


def test_worker_only_termination_preserves_multi_node_requests(
        tpu_api, wait_for_operation, monkeypatch):
    queued = tpu_api.projects().locations().queuedResources()
    resource = _queued_resource(kind='worker')
    spec = resource['tpu']['nodeSpec'][0]
    spec.pop('nodeId')
    spec['multiNodeParams'] = {'nodeCount': 2, 'nodeIdPrefix': 'worker'}
    queued.list.return_value.execute.return_value = {
        'queuedResources': [resource],
    }
    monkeypatch.setattr(instance, '_filter_instances',
                        mock.MagicMock(return_value={}))

    instance.terminate_instances(CLUSTER, PROVIDER_CONFIG, worker_only=True)

    queued.delete.assert_not_called()
    wait_for_operation.assert_not_called()


def test_terminate_waits_for_parent_cleanup_before_finding_remaining_nodes(
        tpu_api, wait_for_operation, monkeypatch):
    queued = tpu_api.projects().locations().queuedResources()
    queued.list.return_value.execute.return_value = {
        'queuedResources': [_queued_resource()],
    }
    events = []
    wait_for_operation.side_effect = lambda *a, **k: events.append('wait')
    legacy_node = f'{PARENT}/nodes/legacy-tpu'
    node_terminate = mock.MagicMock(return_value={'name': 'delete-legacy'})
    monkeypatch.setattr(instance_utils.GCPTPUVMInstance, 'terminate',
                        node_terminate)

    def filter_instances(*_args, **_kwargs):
        events.append('filter')
        return {instance_utils.GCPTPUVMInstance: [legacy_node]}

    monkeypatch.setattr(instance, '_filter_instances', filter_instances)

    instance.terminate_instances(CLUSTER, PROVIDER_CONFIG)

    assert events == ['wait', 'filter', 'wait']
    node_terminate.assert_called_once_with(PROJECT, ZONE, legacy_node)


def test_compute_termination_skips_tpu_queue(monkeypatch):
    terminate_queued = mock.MagicMock()
    monkeypatch.setattr(instance_utils.GCPTPUVMInstance,
                        'terminate_queued_resources', terminate_queued)
    monkeypatch.setattr(instance, '_filter_instances',
                        mock.MagicMock(return_value={}))

    instance.terminate_instances(CLUSTER, dict(PROVIDER_CONFIG,
                                               _has_tpus=False))

    terminate_queued.assert_not_called()


def test_stop_preserves_queued_resources(monkeypatch):
    terminate_queued = mock.MagicMock()
    monkeypatch.setattr(instance_utils.GCPTPUVMInstance,
                        'terminate_queued_resources', terminate_queued)
    monkeypatch.setattr(instance, '_filter_instances',
                        mock.MagicMock(return_value={}))

    instance.stop_instances(CLUSTER, PROVIDER_CONFIG)

    terminate_queued.assert_not_called()


def test_standard_tpu_vm_termination_is_unchanged(tpu_api):
    node = f'{PARENT}/nodes/standard-tpu'
    nodes = tpu_api.projects().locations().nodes()
    nodes.delete.return_value.execute.return_value = {'name': 'delete-node'}

    result = instance_utils.GCPTPUVMInstance.terminate(PROJECT, ZONE, node)

    assert result == {'name': 'delete-node'}
    nodes.delete.assert_called_once_with(name=node)
    tpu_api.projects().locations().queuedResources().delete.assert_not_called()


@pytest.mark.parametrize('states,should_terminate', [
    ([], False),
    (['CREATING'], False),
    (['READY', 'PREEMPTED'], False),
    (['PREEMPTED'], True),
])
def test_status_query_only_terminates_observed_preempted_nodes(
        monkeypatch, states, should_terminate):
    nodes = {
        f'{PARENT}/nodes/node-{i}': {
            'state': state
        } for i, state in enumerate(states)
    }
    monkeypatch.setattr(instance_utils.GCPTPUVMInstance, 'filter',
                        mock.MagicMock(return_value=nodes))
    terminate = mock.MagicMock()
    monkeypatch.setattr(instance, 'terminate_instances', terminate)

    instance.query_instances(CLUSTER, CLUSTER, PROVIDER_CONFIG)

    assert terminate.called == should_terminate
