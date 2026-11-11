"""Admission for parked native pool launches."""
import pytest

from sky.utils import controller_utils


@pytest.mark.parametrize('pool,consolidation,durable,exclude',
                         [(True, True, True, True), (False, True, True, False),
                          (True, False, True, False),
                          (True, True, False, False)])
def test_pool_admission_releases_only_opted_in_parked_launches(
        monkeypatch, pool, consolidation, durable, exclude):
    monkeypatch.setattr(controller_utils, '_is_consolidation_mode',
                        lambda _: consolidation)
    monkeypatch.setattr(controller_utils.restart_recovery, 'enabled',
                        lambda: durable)
    monkeypatch.setattr(controller_utils, '_get_request_parallelism',
                        lambda _: 152)
    monkeypatch.setattr(controller_utils.serve_state,
                        'total_number_terminating_replicas', lambda: 0)
    calls = []

    def count(exclude_parked=False):
        calls.append(exclude_parked)
        return 0 if exclude_parked else 152

    monkeypatch.setattr(controller_utils.serve_state,
                        'total_number_provisioning_replicas', count)
    assert controller_utils.can_provision(pool) is exclude
    assert calls == [exclude]
    monkeypatch.setattr(controller_utils.serve_state,
                        'total_number_provisioning_replicas', lambda **kw: 152)
    assert not controller_utils.can_provision(pool)
