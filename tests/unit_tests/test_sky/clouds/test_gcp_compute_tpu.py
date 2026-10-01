"""Tests for TPUs provisioned through the Compute Engine API.

A TPU slice requested with `accelerator_args: {api: compute}` maps to a TPU
machine type of the accelerator-optimized family (e.g. `ct5p-hightpu-4t`)
instead of a TPU API resource.
"""
# pylint: disable=protected-access,redefined-outer-name
from unittest import mock

import jinja2
import jsonschema
import pandas as pd
import pytest

from sky import clouds
from sky import exceptions
from sky import resources as resources_lib
from sky import skypilot_config
from sky.catalog import gcp_catalog
from sky.clouds import Region
from sky.clouds import Zone
from sky.clouds.gcp import GCP
from sky.clouds.utils import gcp_utils
from sky.provision.gcp import instance_utils
from sky.utils import resources_utils
from sky.utils import schemas

_CATALOG_COLUMNS = [
    'InstanceType', 'vCPUs', 'MemoryGiB', 'AcceleratorName', 'AcceleratorCount',
    'GpuInfo', 'Region', 'AvailabilityZone', 'Price', 'SpotPrice'
]


def _tpu_row(acc: str, region: str, zone: str, price: float, spot: float):
    return [None, None, None, acc, 1, acc, region, zone, price, spot]


def _vm_row(instance_type: str, vcpus: float, mem: float, region: str,
            zone: str, price: float, spot: float):
    return [
        instance_type, vcpus, mem, None, None, None, region, zone, price, spot
    ]


@pytest.fixture
def small_catalog():
    """A catalog with two TPU slices and one existing TPU machine type row."""
    rows = [
        _vm_row('n1-standard-8', 8, 30, 'us-east5', 'us-east5-b', 0.38, 0.08),
        # The fuzzy-candidate lookup of the TPU API path needs an N1 host.
        _vm_row('n1-standard-16', 16, 60, 'us-east5', 'us-east5-b', 0.76, 0.16),
        _tpu_row('tpu-v5p-8', 'us-east5', 'us-east5-b', 16.8, 3.81),
        _tpu_row('tpu-v5p-8', 'us-east5', 'us-east5-c', 16.8, 3.81),
        # Duplicated zone, as in the hosted catalog for europe-west4-b.
        _tpu_row('tpu-v5p-8', 'us-east5', 'us-east5-c', 16.8, 3.81),
        _tpu_row('tpu-v5p-16', 'us-east5', 'us-east5-b', 33.6, 7.62),
        _tpu_row('tpu-v6e-4', 'us-east4', 'us-east4-a', 10.8, 0.97),
        # A row fetched from the machineTypes API is kept as it is.
        _vm_row('ct6e-standard-4t', 180, 720, 'us-east4', 'us-east4-a', 0.0,
                0.0),
    ]
    return pd.DataFrame(rows, columns=_CATALOG_COLUMNS)


def test_tpu_machine_type_tables_are_consistent():
    for acc, machine_type in gcp_utils.TPU_MACHINE_TYPES.items():
        assert acc.startswith('tpu-')
        assert gcp_utils.is_tpu_machine_type(machine_type)
        assert gcp_utils.TPU_MACHINE_TYPE_TO_ACC[machine_type] == acc
        assert machine_type in gcp_utils.TPU_MACHINE_TYPE_SPECS
    assert gcp_utils.get_tpu_machine_type('tpu-v5p-8') == 'ct5p-hightpu-4t'
    assert gcp_utils.get_tpu_machine_type('tpu-v5litepod-4') == (
        'ct5lp-hightpu-4t')
    assert gcp_utils.get_tpu_machine_type('tpu-v6e-8') == 'ct6e-standard-8t'
    # Multi-host slices have no single-host machine type.
    assert gcp_utils.get_tpu_machine_type('tpu-v5p-16') is None
    assert not gcp_utils.is_tpu_machine_type('n1-standard-8')
    assert not gcp_utils.is_tpu_machine_type('c3-standard-4')
    assert not gcp_utils.is_tpu_machine_type(None)


def test_add_tpu_machine_type_rows(small_catalog):
    df = gcp_catalog.add_tpu_machine_type_rows(small_catalog)
    assert list(df.columns) == _CATALOG_COLUMNS
    ct5p = df[df['InstanceType'] == 'ct5p-hightpu-4t']
    # One host row per zone, with the duplicated zone collapsed.
    assert sorted(ct5p['AvailabilityZone']) == ['us-east5-b', 'us-east5-c']
    assert (ct5p['vCPUs'] == 208).all()
    assert (ct5p['MemoryGiB'] == 448).all()
    assert (ct5p['Price'] == 0.0).all()
    assert (ct5p['SpotPrice'] == 0.0).all()
    assert ct5p['AcceleratorName'].isna().all()
    # The multi-host slice gets no host row.
    added = df.iloc[len(small_catalog):]
    assert set(added['InstanceType']) == {'ct5p-hightpu-4t'}
    # The existing machine type row is not duplicated.
    ct6e = df[df['InstanceType'] == 'ct6e-standard-4t']
    assert len(ct6e) == 1
    # The original rows are untouched.
    assert len(df) == len(small_catalog) + 2


def test_add_tpu_machine_type_rows_without_tpus():
    df = pd.DataFrame(
        [_vm_row('n1-standard-8', 8, 30, 'us-east5', 'us-east5-b', 0.38, 0.08)],
        columns=_CATALOG_COLUMNS)
    assert gcp_catalog.add_tpu_machine_type_rows(df) is df


def test_catalog_wrapper_recomputes_only_on_reload(small_catalog):
    hosted = mock.MagicMock()
    hosted._load_df.return_value = small_catalog
    wrapper = gcp_catalog._CatalogWithTPUMachineTypes(hosted)
    first = wrapper._load_df()
    assert first is wrapper._load_df()
    assert 'ct5p-hightpu-4t' in set(wrapper['InstanceType'])
    hosted._load_df.return_value = small_catalog.copy()
    second = wrapper._load_df()
    assert second is not first
    assert 'ct5p-hightpu-4t' in set(second['InstanceType'])


def test_get_tpu_machine_type_for_accelerator(small_catalog):
    df = gcp_catalog.add_tpu_machine_type_rows(small_catalog)
    with mock.patch.object(gcp_catalog, '_df', df):
        get = gcp_catalog.get_tpu_machine_type_for_accelerator
        assert get('tpu-v5p-8', 1) == 'ct5p-hightpu-4t'
        assert get('tpu-v5p-8', 1, cpus='200+') == 'ct5p-hightpu-4t'
        assert get('tpu-v5p-8', 1, cpus='208',
                   memory='448') == ('ct5p-hightpu-4t')
        assert get('tpu-v5p-8', 1, cpus='300+') is None
        assert get('tpu-v5p-8', 1, memory='500+') is None
        assert get('tpu-v5p-8', 2) is None
        assert get('tpu-v5p-16', 1) is None
        # No catalog row in any zone.
        assert get('tpu-v5litepod-1', 1) is None


def test_get_accelerators_from_instance_type():
    assert gcp_catalog.get_accelerators_from_instance_type(
        'ct5p-hightpu-4t') == {
            'tpu-v5p-8': 1
        }
    assert gcp_catalog.get_accelerators_from_instance_type(
        'ct6e-standard-1t') == {
            'tpu-v6e-1': 1
        }
    assert gcp_catalog.get_accelerators_from_instance_type(
        'n1-standard-8') is None


def test_check_accelerator_attachable_to_host(small_catalog):
    df = gcp_catalog.add_tpu_machine_type_rows(small_catalog)
    with mock.patch.object(gcp_catalog, '_df', df), \
            mock.patch.object(gcp_catalog, 'list_accelerators',
                              return_value={'tpu-v5p-8': []}):
        check = gcp_catalog.check_accelerator_attachable_to_host
        check('ct5p-hightpu-4t', {'tpu-v5p-8': 1}, 'us-east5-b')
        # Inferred from the machine type.
        check('ct5p-hightpu-4t', None, 'us-east5-b')
        with pytest.raises(exceptions.ResourcesMismatchError):
            check('ct6e-standard-4t', {'tpu-v5p-8': 1}, 'us-east5-b')
        with pytest.raises(exceptions.ResourcesMismatchError):
            check('ct5p-hightpu-4t', {'tpu-v5p-8': 2}, 'us-east5-b')
        # The TPU API paths are unchanged.
        check('TPU-VM', {'tpu-v5p-8': 1}, 'us-east5-b')
        with pytest.raises(exceptions.ResourcesMismatchError):
            check('c3-standard-4', {'tpu-v5p-8': 1}, 'us-east5-b')


def test_resources_classification():
    compute = resources_lib.Resources(cloud=GCP(),
                                      accelerators='tpu-v5p-8',
                                      accelerator_args={'api': 'compute'})
    assert gcp_utils.is_tpu(compute)
    assert gcp_utils.is_compute_tpu(compute)
    assert not gcp_utils.is_tpu_vm(compute)
    assert not gcp_utils.is_tpu_vm_pod(compute)
    assert not gcp_utils.is_tpu_node(compute)
    # No TPU API runtime version is added.
    assert compute.accelerator_args == {'api': 'compute'}

    tpu_vm = resources_lib.Resources(cloud=GCP(), accelerators='tpu-v5p-8')
    assert gcp_utils.is_tpu_vm(tpu_vm)
    assert not gcp_utils.is_compute_tpu(tpu_vm)
    assert not gcp_utils.is_tpu_node(tpu_vm)
    assert tpu_vm.accelerator_args['runtime_version'] == 'v2-alpha-tpuv5'

    tpu_node = resources_lib.Resources(cloud=GCP(),
                                       accelerators='tpu-v2-8',
                                       accelerator_args={'tpu_vm': False})
    assert gcp_utils.is_tpu_node(tpu_node)
    assert not gcp_utils.is_compute_tpu(tpu_node)

    assert not gcp_utils.is_compute_tpu(None)
    assert not gcp_utils.is_compute_tpu(
        resources_lib.Resources(cloud=GCP(), instance_type='n1-standard-8'))


def test_resources_tpu_machine_type_implies_compute_api():
    by_instance_type = resources_lib.Resources(cloud=GCP(),
                                               instance_type='ct6e-standard-1t')
    assert by_instance_type.accelerators == {'tpu-v6e-1': 1}
    assert gcp_utils.is_compute_tpu(by_instance_type)

    both = resources_lib.Resources(cloud=GCP(),
                                   instance_type='ct6e-standard-1t',
                                   accelerators='tpu-v6e-1')
    assert both.accelerator_args == {'api': 'compute'}
    assert gcp_utils.is_compute_tpu(both)


@pytest.mark.parametrize('kwargs, message', [
    (dict(accelerators='tpu-v5p-8',
          accelerator_args={
              'api': 'compute',
              'tpu_vm': True
          }), 'tpu_vm'),
    (dict(accelerators='tpu-v5p-8',
          accelerator_args={
              'api': 'compute',
              'runtime_version': 'v2-alpha-tpuv5'
          }), 'runtime_version'),
    (dict(accelerators='tpu-v5p-16', accelerator_args={'api': 'compute'
                                                      }), 'no single-host'),
    (dict(accelerators='tpu-v5p-8',
          accelerator_args={'api': 'compute'},
          instance_type='n1-standard-8'), 'ct5p-hightpu-4t'),
])
def test_resources_rejects_invalid_compute_tpu_args(kwargs, message):
    with pytest.raises(ValueError, match=message):
        resources_lib.Resources(cloud=GCP(), **kwargs)


def test_schema_accepts_api_key():
    schema = schemas.get_resources_schema()
    jsonschema.validate(
        {
            'accelerators': 'tpu-v5p-8',
            'accelerator_args': {
                'api': 'compute'
            }
        }, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            {
                'accelerators': 'tpu-v5p-8',
                'accelerator_args': {
                    'api': 'gke'
                }
            }, schema)


def test_feasible_resources_use_tpu_machine_type(small_catalog):
    df = gcp_catalog.add_tpu_machine_type_rows(small_catalog)
    with mock.patch.object(gcp_catalog, '_df', df):
        requested = resources_lib.Resources(cloud=GCP(),
                                            accelerators='tpu-v5p-8',
                                            accelerator_args={'api': 'compute'},
                                            use_spot=True)
        feasible = GCP()._get_feasible_launchable_resources(requested)
        assert len(feasible.resources_list) == 1
        launchable = feasible.resources_list[0]
        assert launchable.instance_type == 'ct5p-hightpu-4t'
        assert launchable.accelerators == {'tpu-v5p-8': 1}
        assert launchable.accelerator_args == {'api': 'compute'}
        unsupported = GCP._unsupported_features_for_resources(launchable)
        assert clouds.CloudImplementationFeatures.STOP not in unsupported
        assert (clouds.CloudImplementationFeatures.MULTI_NODE
                not in unsupported)

        too_many_cpus = resources_lib.Resources(
            cloud=GCP(),
            accelerators='tpu-v5p-8',
            accelerator_args={'api': 'compute'},
            cpus='300+')
        assert not GCP()._get_feasible_launchable_resources(
            too_many_cpus).resources_list

        # The TPU API path still resolves to TPU-VM.
        tpu_vm = resources_lib.Resources(cloud=GCP(), accelerators='tpu-v5p-8')
        assert GCP()._get_feasible_launchable_resources(
            tpu_vm).resources_list[0].instance_type == 'TPU-VM'


def _deploy_variables(launchable):
    with mock.patch.object(GCP, 'get_project_id', return_value='project'):
        return GCP().make_deploy_resources_variables(
            launchable,
            resources_utils.ClusterName('tpu', 'tpu-abcd'),
            Region('us-east5'), [Zone('us-east5-b')],
            num_nodes=1,
            dryrun=True)


def _render_node_config(variables):
    with open('sky/templates/gcp-ray.yml.j2', encoding='utf-8') as f:
        template = f.read()
    start = template.index('      machineType: {{instance_type}}')
    end = template.index('{%- endif %}\n\nhead_node_type')
    return jinja2.Template(template[start:end]).render(disk_size=256,
                                                       **variables)


@pytest.mark.parametrize('use_spot', [True, False])
def test_deploy_variables_and_template(use_spot):
    launchable = resources_lib.Resources(cloud=GCP(),
                                         instance_type='ct5p-hightpu-4t',
                                         accelerators='tpu-v5p-8',
                                         accelerator_args={'api': 'compute'},
                                         use_spot=use_spot)
    variables = _deploy_variables(launchable)
    assert variables['instance_type'] == 'ct5p-hightpu-4t'
    assert variables['compute_tpu'] is True
    assert variables['tpu_vm'] is False
    assert variables['tpu_node_name'] is None
    assert variables['gpu'] is None
    assert variables['docker_run_options'] == ['--privileged']
    assert variables['image_id'].endswith(
        'images/family/ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e')
    assert variables['disk_tier'] == 'pd-balanced'

    rendered = _render_node_config(variables)
    assert 'machineType: ct5p-hightpu-4t' in rendered
    assert 'acceleratorType' not in rendered
    assert rendered.count('onHostMaintenance: TERMINATE') == 1
    assert ('provisioningModel: SPOT' in rendered) == use_spot


def test_deploy_variables_tpu_vm_unchanged():
    launchable = resources_lib.Resources(cloud=GCP(),
                                         instance_type='TPU-VM',
                                         accelerators='tpu-v5p-8')
    variables = _deploy_variables(launchable)
    assert variables['tpu_vm'] is True
    assert variables['compute_tpu'] is False
    assert variables['runtime_version'] == 'v2-alpha-tpuv5'


@pytest.mark.parametrize('instance_type, expected', [
    ('ct5p-hightpu-4t', {
        resources_utils.DiskTier.LOW: 'pd-balanced',
        resources_utils.DiskTier.MEDIUM: 'pd-balanced',
        resources_utils.DiskTier.HIGH: 'pd-balanced',
        resources_utils.DiskTier.ULTRA: 'pd-balanced',
    }),
    ('ct6e-standard-8t', {
        resources_utils.DiskTier.LOW: 'hyperdisk-balanced',
        resources_utils.DiskTier.ULTRA: 'hyperdisk-balanced',
    }),
    ('ct5lp-hightpu-4t', {
        resources_utils.DiskTier.LOW: 'pd-balanced',
        resources_utils.DiskTier.ULTRA: 'pd-balanced',
    }),
])
def test_boot_disk_types(instance_type, expected):
    for tier, disk_type in expected.items():
        assert GCP._get_disk_type(instance_type, tier) == disk_type


def test_bulk_insert_disabled_for_tpu_machine_types():
    use_bulk_insert = instance_utils.GCPComputeInstance._use_bulk_insert
    assert not use_bulk_insert({'machineType': 'ct5p-hightpu-4t'})
    assert not use_bulk_insert({'machineType': 'ct6e-standard-1t'})
    assert use_bulk_insert({'machineType': 'n1-standard-8'})
    assert use_bulk_insert({})


def test_is_image_family():
    assert GCP._is_image_family(
        'projects/ubuntu-os-accelerator-images/global/images/family/'
        'ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e')
    assert not GCP._is_image_family(
        'projects/ubuntu-os-cloud/global/images/ubuntu-2204-jammy-v20240101')
    assert not GCP._is_image_family(
        'projects/p/global/machineImages/my-machine-image')


def test_image_size_resolves_image_family():
    compute = mock.MagicMock()
    get_from_family = compute.images.return_value.getFromFamily
    get_from_family.return_value.execute.return_value = {'diskSizeGb': '20'}
    with mock.patch('sky.adaptors.gcp.build', return_value=compute):
        size = GCP._get_image_size(
            'projects/ubuntu-os-accelerator-images/global/images/family/'
            'ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e')
    assert size == 20.0
    get_from_family.assert_called_once_with(
        project='ubuntu-os-accelerator-images',
        family='ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e')
    compute.images.return_value.get.assert_not_called()


# Multi-host slices: tpu-v5p-8 per host, with a topology that spans hosts.
_SLICE_ARGS = {'api': 'compute', 'topology': '2x2x2'}


@pytest.mark.parametrize('machine_type, topology, hosts', [
    ('ct5p-hightpu-4t', '2x2x1', 1),
    ('ct5p-hightpu-4t', '2x2x2', 2),
    ('ct5p-hightpu-4t', '2x4x4', 8),
])
def test_tpu_slice_hosts(machine_type, topology, hosts):
    assert gcp_utils.get_tpu_slice_hosts(machine_type, topology) == hosts


@pytest.mark.parametrize('machine_type, topology, message', [
    ('ct5p-hightpu-4t', '2x2', 'must have 3 dimensions'),
    ('ct5p-hightpu-4t', '2x3x1', 'does not fill whole'),
    ('ct5p-hightpu-4t', '2x0x2', 'Invalid TPU topology'),
    ('ct5p-hightpu-4t', '2-2-2', 'Invalid TPU topology'),
    ('ct6e-standard-4t', '2x4', 'does not form multi-host slices'),
])
def test_tpu_slice_hosts_rejects(machine_type, topology, message):
    with pytest.raises(ValueError, match=message):
        gcp_utils.get_tpu_slice_hosts(machine_type, topology)


def test_resources_slice_topology():
    multi_host = resources_lib.Resources(cloud=GCP(),
                                         accelerators='tpu-v5p-8',
                                         accelerator_args=dict(_SLICE_ARGS))
    assert gcp_utils.get_compute_tpu_slice_topology(multi_host) == '2x2x2'
    single_host = resources_lib.Resources(cloud=GCP(),
                                          accelerators='tpu-v5p-8',
                                          accelerator_args={
                                              'api': 'compute',
                                              'topology': '2x2x1'
                                          })
    assert gcp_utils.get_compute_tpu_slice_topology(single_host) is None
    unsupported = GCP._unsupported_features_for_resources(multi_host)
    assert clouds.CloudImplementationFeatures.STOP in unsupported
    assert (clouds.CloudImplementationFeatures.STOP
            not in GCP._unsupported_features_for_resources(single_host))


@pytest.mark.parametrize('kwargs, message', [
    (dict(accelerators='tpu-v5p-8',
          accelerator_args={
              'api': 'compute',
              'topology': '2x3x1'
          }), 'does not fill whole'),
    (dict(accelerators='tpu-v6e-4',
          accelerator_args={
              'api': 'compute',
              'topology': '2x2'
          }), 'does not form multi-host slices'),
    (dict(accelerators='tpu-v5p-8', accelerator_args={'topology': '2x2x2'
                                                     }), 'requires'),
    (dict(accelerators='tpu-v5p-16', accelerator_args={'api': 'compute'
                                                      }), 'topology: 2x2x2'),
])
def test_resources_rejects_invalid_slice_args(kwargs, message):
    with pytest.raises(ValueError, match=message):
        resources_lib.Resources(cloud=GCP(), **kwargs)


def test_schema_accepts_topology_key():
    jsonschema.validate(
        {
            'accelerators': 'tpu-v5p-8',
            'accelerator_args': dict(_SLICE_ARGS)
        }, schemas.get_resources_schema())


def _slice_launchable(**kwargs):
    return resources_lib.Resources(cloud=GCP(),
                                   instance_type='ct5p-hightpu-4t',
                                   accelerators='tpu-v5p-8',
                                   accelerator_args=dict(_SLICE_ARGS),
                                   **kwargs)


def _slice_deploy_variables(launchable, num_nodes):
    with mock.patch.object(GCP, 'get_project_id', return_value='project'):
        return GCP().make_deploy_resources_variables(
            launchable,
            resources_utils.ClusterName('tpu', 'tpu-abcd'),
            Region('us-east5'), [Zone('us-east5-b')],
            num_nodes=num_nodes,
            dryrun=True)


def _render_slice_config(variables):
    with open('sky/templates/gcp-ray.yml.j2', encoding='utf-8') as f:
        template = f.read()
    provider_start = template.index('  use_managed_instance_group:')
    provider_end = template.index('{%- if enable_gvnic %}')
    node_start = template.index('      {%- if tpu_slice_topology is not none')
    node_end = template.index('      {%- if specific_reservations %}')
    snippet = (template[provider_start:provider_end] +
               template[node_start:node_end])
    return jinja2.Template(snippet).render(**variables)


def test_deploy_variables_and_template_for_slice():
    variables = _slice_deploy_variables(_slice_launchable(use_spot=True), 2)
    assert variables['tpu_slice_topology'] == '2x2x2'
    rendered = _render_slice_config(variables)
    assert 'use_managed_instance_group: True' in rendered
    assert 'tpu_slice_topology: 2x2x2' in rendered
    assert 'tpu-slice:\n        topology: 2x2x2' in rendered

    single = _slice_deploy_variables(
        resources_lib.Resources(cloud=GCP(),
                                instance_type='ct5p-hightpu-4t',
                                accelerators='tpu-v5p-8',
                                accelerator_args={'api': 'compute'}), 1)
    assert single['tpu_slice_topology'] is None
    rendered = _render_slice_config(single)
    assert 'use_managed_instance_group: False' in rendered
    assert 'tpu-slice' not in rendered


def test_deploy_variables_reject_wrong_num_nodes():
    with pytest.raises(ValueError,
                       match='needs num_nodes: 2; got num_nodes: 4'):
        _slice_deploy_variables(_slice_launchable(), 4)


def test_deploy_variables_reject_dws_with_slice():
    real = skypilot_config.get_effective_region_config

    def config(cloud, region, keys, default_value=None, **kwargs):
        if keys == ('managed_instance_group',):
            return {'run_duration': 3600}
        return real(cloud=cloud,
                    region=region,
                    keys=keys,
                    default_value=default_value,
                    **kwargs)

    with mock.patch.object(skypilot_config,
                           'get_effective_region_config',
                           side_effect=config):
        with pytest.raises(ValueError, match='cannot be combined'):
            _slice_deploy_variables(_slice_launchable(), 2)


def test_node_type_of_slice():
    assert instance_utils.get_node_type({
        'machineType': 'ct5p-hightpu-4t',
        'tpu-slice': {
            'topology': '2x2x2'
        },
    }) == instance_utils.GCPNodeType.TPU_SLICE
    assert instance_utils.get_node_type({'machineType': 'ct5p-hightpu-4t'
                                        }) == instance_utils.GCPNodeType.COMPUTE


@pytest.fixture
def slice_apis(monkeypatch):
    """Records the Compute Engine calls of the slice handler."""
    calls = []
    mig = instance_utils.mig_utils
    handler = instance_utils.GCPTPUSliceInstanceGroup

    def record(name, result=None):

        def fn(*args, **kwargs):
            calls.append((name, args, kwargs))
            return result

        return fn

    state = {'group_exists': False, 'hosts': ['h1', 'h2']}
    monkeypatch.setattr(mig, 'check_managed_instance_group_exists',
                        lambda *a: state['group_exists'])
    monkeypatch.setattr(mig, 'create_region_instance_template',
                        record('template', {'name': 'op-t'}))
    monkeypatch.setattr(mig, 'create_workload_policy',
                        record('policy', {'name': 'op-p'}))
    monkeypatch.setattr(mig, 'delete_workload_policy', record('delete_policy'))
    monkeypatch.setattr(mig, 'create_managed_instance_group',
                        record('group', {'name': 'op-g'}))
    monkeypatch.setattr(mig, 'wait_for_managed_group_to_be_stable',
                        record('wait'))
    monkeypatch.setattr(handler, 'wait_for_operation', record('op'))
    monkeypatch.setattr(handler, '_delete_instance_template',
                        record('delete_template'))
    monkeypatch.setattr(handler, 'delete_mig', record('delete_mig'))
    monkeypatch.setattr(handler, '_add_labels_and_find_head',
                        lambda *a: list(state['hosts']))
    monkeypatch.setattr(handler, 'create_node_tag', record('head'))
    return calls, state


def _create_slice(count=2, total_count=2):
    return instance_utils.GCPTPUSliceInstanceGroup.create_instances(
        'tpu-abcd',
        'project',
        'us-east5-b', {
            'machineType': 'zones/us-east5-b/machineTypes/ct5p-hightpu-4t',
            'labels': {
                'skypilot-user': 'Alice'
            },
            'tpu-slice': {
                'topology': '2x2x2'
            },
            'networkInterfaces': [{
                'subnetwork': 'subnet'
            }],
            'networkConfig': {
                'subnetwork': 'subnet'
            },
        }, {},
        count=count,
        total_count=total_count,
        include_head_node=True)


def test_create_slice(slice_apis):
    calls, _ = slice_apis
    assert _create_slice() == (None, ['h1', 'h2'])
    by_name = {name: (args, kwargs) for name, args, kwargs in calls}
    args, kwargs = by_name['template']
    assert kwargs == {'for_tpu_slice': True}
    template_config = args[4]
    assert 'tpu-slice' not in template_config
    assert 'networkConfig' not in template_config
    assert template_config['networkInterfaces'] == [{'subnetwork': 'subnet'}]
    assert template_config['machineType'] == 'ct5p-hightpu-4t'
    assert template_config['labels']['skypilot-user'] == 'alice'
    assert by_name['policy'][0] == ('project', 'us-east5', 'sky-wp-tpu-abcd',
                                    '2x2x2')
    args, kwargs = by_name['group']
    assert args[:3] == ('project', 'us-east5-b', 'sky-mig-tpu-abcd')
    assert kwargs['size'] == 2
    assert kwargs['workload_policy_url'] == (
        'projects/project/regions/us-east5/resourcePolicies/sky-wp-tpu-abcd')
    assert by_name['head'][0][2] == 'h1'
    assert 'delete_mig' not in by_name
    order = [
        name for name, _, _ in calls
        if name in ('template', 'policy', 'group', 'wait')
    ]
    assert order == ['template', 'policy', 'group', 'wait']


def test_create_slice_replaces_leftover_group(slice_apis):
    calls, state = slice_apis
    state['group_exists'] = True
    _create_slice()
    names = [name for name, _, _ in calls]
    assert names.index('delete_mig') < names.index('group')


@pytest.mark.usefixtures('slice_apis')
def test_create_slice_rejects_partial_slice():
    with pytest.raises(RuntimeError, match='1 of 2 hosts left'):
        _create_slice(count=1)


def test_create_slice_checks_host_count(slice_apis):
    _, state = slice_apis
    state['hosts'] = ['h1']
    with pytest.raises(RuntimeError, match='expected 2'):
        _create_slice()


def test_bulk_group_body(monkeypatch):
    compute = mock.MagicMock()
    monkeypatch.setattr(instance_utils.mig_utils.gcp, 'build',
                        lambda *a, **k: compute)
    instance_utils.mig_utils.create_managed_instance_group(
        'project',
        'us-east5-b',
        'sky-mig-x',
        'template-url',
        size=2,
        workload_policy_url='policy-url')
    body = compute.instanceGroupManagers().insert.call_args.kwargs['body']
    assert body['targetSizePolicy'] == {'mode': 'BULK'}
    assert body['resourcePolicies'] == {'workloadPolicy': 'policy-url'}
    assert body['target_size'] == 2

    instance_utils.mig_utils.create_managed_instance_group('project',
                                                           'us-east5-b',
                                                           'sky-mig-x',
                                                           'template-url',
                                                           size=0)
    body = compute.instanceGroupManagers().insert.call_args.kwargs['body']
    assert 'targetSizePolicy' not in body
    assert 'resourcePolicies' not in body


def test_workload_policy_body(monkeypatch):
    compute = mock.MagicMock()
    monkeypatch.setattr(instance_utils.mig_utils.gcp, 'build',
                        lambda *a, **k: compute)
    instance_utils.mig_utils.create_workload_policy('project', 'us-east5',
                                                    'sky-wp-x', '2x2x4')
    kwargs = compute.resourcePolicies().insert.call_args.kwargs
    assert kwargs['region'] == 'us-east5'
    assert kwargs['body']['workloadPolicy'] == {
        'type': 'HIGH_THROUGHPUT',
        'acceleratorTopology': '2x2x4',
    }


def test_delete_workload_policy_ignores_missing(monkeypatch):
    error = type('HttpError', (Exception,), {})
    missing = error('not found')
    missing.resp = mock.MagicMock(status=404)
    compute = mock.MagicMock()
    compute.resourcePolicies().delete().execute.side_effect = missing
    monkeypatch.setattr(instance_utils.mig_utils.gcp, 'build',
                        lambda *a, **k: compute)
    monkeypatch.setattr(instance_utils.mig_utils.gcp, 'http_error_exception',
                        lambda: error)
    assert instance_utils.mig_utils.delete_workload_policy(
        'project', 'us-east5', 'sky-wp-x') is None


def test_stop_slice_is_refused():
    # pylint: disable=import-outside-toplevel
    from sky.provision.gcp import instance as gcp_instance
    with pytest.raises(NotImplementedError, match='sky down'):
        gcp_instance.stop_instances(
            'tpu-abcd', {
                'availability_zone': 'us-east5-b',
                'project_id': 'project',
                'tpu_slice_topology': '2x2x2',
            })


def test_feasible_resources_keep_slice_topology(small_catalog):
    df = gcp_catalog.add_tpu_machine_type_rows(small_catalog)
    with mock.patch.object(gcp_catalog, '_df', df):
        requested = resources_lib.Resources(cloud=GCP(),
                                            accelerators='tpu-v5p-8',
                                            accelerator_args=dict(_SLICE_ARGS),
                                            use_spot=True)
        launchable = GCP()._get_feasible_launchable_resources(
            requested).resources_list[0]
    assert launchable.instance_type == 'ct5p-hightpu-4t'
    assert launchable.accelerator_args == _SLICE_ARGS
    assert gcp_utils.get_compute_tpu_slice_topology(launchable) == '2x2x2'
