"""New GCP names resist collisions without renaming retained allocations."""
# pylint: disable=protected-access
from unittest import mock

import pytest

from sky import clouds
from sky.backends import backend_utils
from sky.utils import common_utils


@pytest.fixture(autouse=True)
def names(monkeypatch):
    monkeypatch.setattr(common_utils, 'get_user_hash', lambda: '65d5779e')
    monkeypatch.setattr(backend_utils.global_user_state,
                        'get_handle_from_cluster_name', lambda _: None)
    monkeypatch.setattr(backend_utils.global_user_state, 'get_cluster_yaml_str',
                        lambda _: None)


def test_new_gcp_workers_do_not_reproduce_known_collision():
    prefix = 'logan-perm-v5litepod16-c1a-spot-extra-'
    assert common_utils.make_cluster_name_on_cloud(
        prefix + '23',
        35) == (common_utils.make_cluster_name_on_cloud(prefix + '28', 35))
    names_on_cloud = {
        backend_utils._cluster_name_on_cloud(prefix + str(i), clouds.GCP())
        for i in range(1, 501)
    }
    assert len(names_on_cloud) == 500
    assert all(len(name) <= 35 for name in names_on_cloud)


@pytest.mark.parametrize('retained_as', ['handle', 'yaml'])
def test_legacy_name_is_preserved_before_template_generation(
        monkeypatch, retained_as):
    old = 'logan-perm-v5litepod16-bv-65d5779e'
    if retained_as == 'handle':
        monkeypatch.setattr(backend_utils.global_user_state,
                            'get_handle_from_cluster_name',
                            lambda _: mock.Mock(cluster_name_on_cloud=old))
    else:
        monkeypatch.setattr(backend_utils.global_user_state,
                            'get_cluster_yaml_str',
                            lambda _: f'cluster_name: {old}')
    assert backend_utils._cluster_name_on_cloud('retained', clouds.GCP()) == old


def test_other_clouds_retain_name_format():
    name = 'a-very-long-existing-display-name'
    assert backend_utils._cluster_name_on_cloud(
        name, clouds.AWS()) == (common_utils.make_cluster_name_on_cloud(
            name, max_length=clouds.AWS.max_cluster_name_length()))
