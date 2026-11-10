"""Capacity waits release API workers without changing allocation identities."""
# pylint: disable=protected-access,redefined-outer-name
# pylint: disable=unused-argument,invalid-name
import asyncio
import copy
import json
import pickle
from unittest import mock

from googleapiclient import errors
import httplib2
import pytest

from sky import exceptions
from sky.provision import common
from sky.provision.gcp import instance
from sky.provision.gcp import instance_utils
from sky.provision.gcp import queued_resource_wait as wait
from sky.server.requests import requests
from sky.server.requests import restart_recovery
from sky.utils import common_utils

PROJECT, ZONE, CLUSTER = 'project', 'us-central2-b', 'cluster'
PARENT = f'projects/{PROJECT}/locations/{ZONE}'
CONFIG = dict(acceleratorType='v4-8',
              labels={'skypilot-user': 'test-owner'},
              runtimeVersion='runtime',
              gcp_queued_resource=True,
              schedulingConfig=dict(preemptible=True))


def _http_error(status):
    return errors.HttpError(httplib2.Response({'status': str(status)}),
                            json.dumps({
                                'error': {
                                    'message': 'test'
                                }
                            }).encode())


@pytest.fixture
def queue(monkeypatch, tmp_path):
    monkeypatch.setattr(wait, 'JOURNAL_ROOT', tmp_path / 'waits')
    monkeypatch.setattr(wait, 'native_cluster_hash',
                        lambda: 'native-generation')
    monkeypatch.setattr(wait, 'canonical_parent',
                        lambda project, zone: f'projects/123/locations/{zone}')
    monkeypatch.setattr(common_utils, 'is_in_request_context', lambda: True)
    monkeypatch.setattr(common_utils, 'get_current_request_id',
                        lambda: 'original')
    api = mock.MagicMock()
    resources = {}

    def get(name):
        response = mock.Mock()

        def execute(**_):
            if name not in resources:
                raise _http_error(404)
            return copy.deepcopy(resources[name])

        response.execute.side_effect = execute
        return response

    def create(parent, queuedResourceId, body):
        name = f'{parent}/queuedResources/{queuedResourceId}'

        def execute(**_):
            resources[name] = copy.deepcopy(body)
            resources[name].update(name=name,
                                   createTime='generation-1',
                                   state=dict(state='ACCEPTED'))
            specs = resources[name]['tpu']['nodeSpec']
            if isinstance(specs, dict):
                resources[name]['tpu']['nodeSpec'] = [specs]
            return dict(name=f'{PARENT}/operations/{queuedResourceId}')

        response = mock.Mock()
        response.execute.side_effect = execute
        return response

    api.get.side_effect = get
    api.create.side_effect = create
    monkeypatch.setattr(wait, '_resources', lambda: api)
    monkeypatch.setattr(wait, '_operation_error', lambda row: None)
    return api, resources


def _create(count=1):
    return instance_utils.GCPTPUVMInstance.create_instances(
        CLUSTER, PROJECT, ZONE, copy.deepcopy(CONFIG), {}, count, count, True)


def _path():
    return wait.journal_path('original', PROJECT, ZONE, CLUSTER)


def test_park_resume_preserves_entire_group_and_pickle(queue):
    api, resources = queue
    with pytest.raises(exceptions.ExecutionPausedError) as error:
        _create(4)
    assert api.create.call_count == 4
    condition = pickle.loads(pickle.dumps(error.value)).continue_condition
    assert condition.path == str(_path())
    original = wait.load(_path())
    for name in resources:
        resources[name]['state']['state'] = 'ACTIVE'
    names = wait.create(_path())
    assert len(names) == 4
    assert api.create.call_count == 4
    assert wait.load(_path())['request_id'] == 'original'
    assert wait.load(
        _path())['resources'].keys() == original['resources'].keys()
    assert wait.load(_path())['stage'] == 'ready'
    kinds = [
        r['tpu']['nodeSpec'][0]['node']['labels']['ray-node-type']
        for r in resources.values()
    ]
    assert kinds == ['head', 'worker', 'worker', 'worker']


@pytest.mark.parametrize('field', [
    'createTime', 'nodeId', 'acceleratorType', 'runtimeVersion', 'labels',
    'spot', 'missing'
])
def test_changed_or_missing_resource_never_recreated_or_torn_down(queue, field):
    api, resources = queue
    with pytest.raises(exceptions.ExecutionPausedError):
        _create()
    name, resource = next(iter(resources.items()))
    if field == 'missing':
        resources.clear()
    elif field == 'spot':
        resource.pop('spot')
    elif field == 'createTime':
        resource[field] = 'replacement'
    elif field == 'nodeId':
        resource['tpu']['nodeSpec'][0][field] = 'replacement'
    else:
        resource['tpu']['nodeSpec'][0]['node'][field] = 'replacement'
    with pytest.raises(exceptions.ExecutionPausedError):
        wait.create(_path())
    assert api.create.call_count == 1
    assert not api.delete.called
    assert name in wait.load(_path())['resources']


@pytest.mark.parametrize('accepted', [False, True])
def test_lost_create_response_never_repeated(queue, accepted):
    api, resources = queue
    real_create = api.create.side_effect

    def uncertain(**kwargs):
        request = real_create(**kwargs)
        execute = request.execute.side_effect

        def lose_response(**kw):
            if accepted:
                execute(**kw)
            raise TimeoutError('response lost')

        request.execute.side_effect = lose_response
        return request

    api.create.side_effect = uncertain
    with pytest.raises(exceptions.ExecutionPausedError):
        _create()
    for resource in resources.values():
        resource['state']['state'] = 'ACTIVE'
    if accepted:
        assert wait.create(_path())
    else:
        with pytest.raises(exceptions.ExecutionPausedError):
            wait.create(_path())
    assert api.create.call_count == 1


def test_intent_committed_before_mutation_and_unsent_group_resumes(queue):
    api, _ = queue
    real_create = api.create.side_effect

    def assert_durable(**kwargs):
        value = wait.load(_path())
        assert len(value['resources']) == 2
        name = f'{PARENT}/queuedResources/{kwargs["queuedResourceId"]}'
        assert value['resources'][name]['phase'] == 'submitting'
        return real_create(**kwargs)

    api.create.side_effect = assert_durable
    with pytest.raises(exceptions.ExecutionPausedError):
        _create(2)
    assert api.create.call_count == 2


def test_deadline_and_one_provisioning_grace_survive_reentry(
        queue, monkeypatch):
    _, resources = queue
    clock = [1000.0]
    monkeypatch.setattr(wait.time, 'time', lambda: clock[0])
    with pytest.raises(exceptions.ExecutionPausedError):
        _create()
    row = next(iter(resources.values()))
    clock[0] = 6999
    row['state']['state'] = 'PROVISIONING'
    assert not wait.check(_path())
    clock[0] = 7200
    row['state']['state'] = 'ACCEPTED'
    assert not wait.check(_path())
    clock[0] = 7899
    row['state']['state'] = 'PROVISIONING'
    with pytest.raises(common.ProvisionerError):
        wait.wait(_path())
    value = wait.load(_path())
    assert value['queue_deadline'] == 7000
    assert next(iter(
        value['resources'].values()))['provisioning_deadline'] == 7899


def test_another_request_cannot_adopt_wait_or_change_contract(
        queue, monkeypatch):
    api, _ = queue
    with pytest.raises(exceptions.ExecutionPausedError):
        _create()
    monkeypatch.setattr(common_utils, 'get_current_request_id',
                        lambda: 'different')
    with pytest.raises(exceptions.ExecutionPausedError):
        _create()
    monkeypatch.setattr(common_utils, 'get_current_request_id',
                        lambda: 'original')
    with pytest.raises(exceptions.ExecutionPausedError):
        _create(2)
    assert api.create.call_count == 1


@pytest.mark.parametrize('node_count', [0, 1])
def test_run_instances_resumes_before_counting_partial_nodes(
        queue, monkeypatch, node_count):
    api, resources = queue
    with pytest.raises(exceptions.ExecutionPausedError):
        _create(2)
    # Regardless of zero/partial nodes, capacity is still pending. No node
    # counting, new name generation, or new provider create may happen.
    node_filter = mock.Mock(
        return_value={str(i): {} for i in range(node_count)})
    monkeypatch.setattr(instance_utils.GCPTPUVMInstance, 'filter', node_filter)
    config = common.ProvisionConfig(provider_config=dict(
        project_id=PROJECT, availability_zone=ZONE),
                                    authentication_config={},
                                    docker_config={},
                                    node_config=CONFIG,
                                    count=2,
                                    tags={},
                                    resume_stopped_nodes=True,
                                    ports_to_open_on_launch=None)
    with pytest.raises(exceptions.ExecutionPausedError):
        instance._run_instances('us-central2', CLUSTER, config)
    assert api.create.call_count == 2
    assert len(resources) == 2
    assert not node_filter.called


def test_cancelled_condition_does_not_probe_or_reschedule(queue):
    condition = wait.QueuedResourceCondition('unused')
    condition._check = mock.Mock()

    async def cancelled():
        return True

    assert not asyncio.run(
        condition.wait_async(is_cancelled=cancelled, fallback_wait_seconds=0))
    assert not condition._check.called


def test_restart_resumes_original_only_while_waiting(queue):
    with pytest.raises(exceptions.ExecutionPausedError):
        _create()
    request = mock.Mock(request_id='original',
                        name='sky.launch',
                        pid=123,
                        status=requests.RequestStatus.WAITING)
    # Mock(name=...) names the mock itself; use an explicit attribute.
    request.name = 'sky.launch'
    assert restart_recovery.resumable(request)
    value = wait.load(_path())
    value['stage'] = 'ready'
    wait.save(_path(), value)
    assert not restart_recovery.resumable(request)
    request.status = requests.RequestStatus.CANCELLED
    assert not restart_recovery.resumable(request)


def test_imported_wait_binds_only_matching_bootstrap_defaults(queue):
    with pytest.raises(exceptions.ExecutionPausedError):
        _create()
    value = wait.load(_path())
    value['legacy_input'] = True
    desired_network = dict(subnetwork='original-subnet', enableExternalIps=True)
    for row in value['resources'].values():
        row['body']['tpu']['nodeSpec']['node'][
            'networkConfig'] = desired_network
    wait.save(_path(), value)
    config = dict(CONFIG,
                  networkConfig=desired_network,
                  networkInterfaces=[dict(subnetwork='original-subnet')])
    assert wait.open_journal(PROJECT, ZONE, CLUSTER,
                             wait.contract(config, {}, 1)) == _path()
    assert 'legacy_input' not in wait.load(_path())
    value['legacy_input'] = True
    wait.save(_path(), value)
    config['networkConfig'] = dict(subnetwork='replacement')
    with pytest.raises(exceptions.ExecutionPausedError):
        wait.open_journal(PROJECT, ZONE, CLUSTER, wait.contract(config, {}, 1))


def test_api_unavailable_parks_without_native_teardown(queue, monkeypatch):
    with pytest.raises(exceptions.ExecutionPausedError):
        _create()
    monkeypatch.setattr(wait, '_resources',
                        mock.Mock(side_effect=RuntimeError()))
    with pytest.raises(exceptions.ExecutionPausedError):
        wait.create(_path())


def test_known_provider_operation_failure_resumes_native_cleanup(
        queue, monkeypatch):
    with pytest.raises(exceptions.ExecutionPausedError):
        _create()
    monkeypatch.setattr(wait, '_operation_error',
                        lambda _: dict(code=7, message='denied'))
    assert wait.check(_path())
    with pytest.raises(common.ProvisionerError):
        wait.wait(_path())


def test_teardown_fences_generation_and_retires_only_confirmed_absence(queue):
    _, resources = queue
    with pytest.raises(exceptions.ExecutionPausedError):
        _create()
    name = next(iter(resources))
    resources[name]['createTime'] = 'replacement'
    with pytest.raises(common.ProvisionerError):
        wait.validate_teardown(PROJECT, ZONE, CLUSTER, resources)
    resources[name]['createTime'] = 'generation-1'
    wait.validate_teardown(PROJECT, ZONE, CLUSTER, resources)
    with pytest.raises(common.ProvisionerError):
        wait.retire(PROJECT, ZONE, CLUSTER)
    stale = wait.load(_path())
    resources.clear()
    wait.retire(PROJECT, ZONE, CLUSTER)
    wait.save(_path(), stale)
    assert wait.load(_path())['stage'] == 'retired'
    assert wait.open_journal(PROJECT, ZONE, CLUSTER,
                             wait.contract(CONFIG, {}, 1)) == _path()
    assert not _path().exists()
    assert list((wait.JOURNAL_ROOT / 'retired').glob('*.json'))


def test_lost_create_without_resource_cannot_be_declared_clean(queue):
    api, _ = queue
    api.create.return_value.execute.side_effect = TimeoutError()
    api.create.side_effect = None
    with pytest.raises(exceptions.ExecutionPausedError):
        _create()
    with pytest.raises(common.ProvisionerError):
        wait.validate_teardown(PROJECT, ZONE, CLUSTER, {})
    with pytest.raises(common.ProvisionerError):
        wait.retire(PROJECT, ZONE, CLUSTER)


@pytest.mark.parametrize('model,guaranteed,valid',
                         [('SPOT', False, True), ('STANDARD', False, False),
                          ('SPOT', True, False),
                          ('RESERVATION_BOUND', False, False),
                          (None, False, False)])
def test_provider_canonical_spot_route_is_explicitly_fenced(
        queue, model, guaranteed, valid):
    _, resources = queue
    with pytest.raises(exceptions.ExecutionPausedError):
        _create()
    resource = next(iter(resources.values()))
    resource.pop('spot')
    resource['bestEffort'] = {}
    if model is not None:
        resource['provisioningModel'] = model
    if guaranteed:
        resource['guaranteed'] = {}
    resource['state']['state'] = 'ACTIVE'
    assert wait.check(_path()) == valid


@pytest.mark.parametrize('parent,valid',
                         [(f'projects/123/locations/{ZONE}', True),
                          (f'projects/456/locations/{ZONE}', False),
                          ('projects/123/locations/other-zone', False),
                          (None, False)])
def test_numeric_project_parent_requires_verified_alias(queue, parent, valid):
    _, resources = queue
    with pytest.raises(exceptions.ExecutionPausedError):
        _create()
    resource = next(iter(resources.values()))
    resource['tpu']['nodeSpec'][0]['parent'] = parent
    resource['state']['state'] = 'ACTIVE'
    assert wait.check(_path()) == valid


@pytest.mark.parametrize('project_id,number,valid', [(PROJECT, '123', True),
                                                     ('foreign', '456', False),
                                                     (PROJECT, '', False)])
def test_project_alias_is_resolved_from_authenticated_project_response(
        monkeypatch, project_id, number, valid):
    from sky.adaptors import gcp  # pylint: disable=import-outside-toplevel
    api = mock.Mock()
    api.projects.return_value.get.return_value.execute.return_value = dict(
        projectId=project_id, projectNumber=number)
    monkeypatch.setattr(gcp, 'build', lambda *args, **kwargs: api)
    if valid:
        assert wait.canonical_parent(PROJECT,
                                     ZONE) == f'projects/123/locations/{ZONE}'
    else:
        with pytest.raises(ValueError):
            wait.canonical_parent(PROJECT, ZONE)
