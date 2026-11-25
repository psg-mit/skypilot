"""Conditional grants must survive provisioning permission checks."""
# pylint: disable=protected-access
import copy
from unittest import mock

import pytest

from sky.provision.gcp import config


@pytest.mark.parametrize('required_roles',
                         [['roles/compute.admin'], ['roles/storage.admin']])
def test_conditional_grant_is_not_an_unconditional_permission(required_roles):
    member = 'serviceAccount:worker@example.iam.gserviceaccount.com'
    binding = {
        'role': 'roles/compute.admin',
        'members': [member],
        'condition': {
            'expression': 'resource.name.startsWith("restricted/")'
        },
    }
    policy = {'version': 3, 'etag': 'original', 'bindings': [binding]}
    crm, iam = mock.MagicMock(), mock.MagicMock()
    crm.projects().getIamPolicy().execute.return_value = copy.deepcopy(policy)
    satisfied, proposed = config._is_permission_satisfied(
        {
            'projectId': 'project',
            'email': member.split(':', 1)[1]
        }, crm, iam, ['compute.instances.create'], required_roles)
    assert not satisfied
    crm.projects().getIamPolicy.assert_called_with(
        resource='project', body={'options': {
            'requestedPolicyVersion': 3
        }})
    iam.projects.assert_not_called()
    assert proposed['version'] == 3
    assert proposed['etag'] == 'original'
    assert proposed['bindings'][0] == binding
    if required_roles:
        assert proposed['bindings'][1] == {
            'role': required_roles[0],
            'members': [member]
        }


def test_unconditional_custom_role_still_satisfies_permissions():
    member = 'serviceAccount:worker@example.iam.gserviceaccount.com'
    crm, iam = mock.MagicMock(), mock.MagicMock()
    crm.projects().getIamPolicy().execute.return_value = {
        'version': 3,
        'bindings': [
            {
                'role': 'projects/project/roles/recovery',
                'members': [member],
                'condition': {
                    'expression': 'false'
                }
            },
            {
                'role': 'projects/project/roles/worker',
                'members': [member]
            },
        ],
    }
    iam.projects().roles().get().execute.return_value = {
        'includedPermissions': ['compute.instances.create']
    }
    iam.reset_mock()
    satisfied, _ = config._is_permission_satisfied(
        {
            'projectId': 'project',
            'email': member.split(':', 1)[1]
        }, crm, iam, ['compute.instances.create'], ['roles/compute.admin'])
    assert satisfied
    iam.projects().roles().get.assert_called_once_with(
        name='projects/project/roles/worker')
