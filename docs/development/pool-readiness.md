# Isolating pool readiness from failed preemption reconciliation

A failed provider or ownership check for one replica must retain that replica's
ownership and cleanup evidence without preventing successful setup probes for
other replicas from becoming READY. Do not relax the failed replica's teardown
checks or infer provider absence from an SSH failure.

Run the focused regression tests without provisioning resources:

```sh
pytest -n 0 tests/unit_tests/test_serve_replica_managers.py \
  tests/unit_tests/test_sky/test_replica_managers.py
```

For a controlled deployment, preserve the shared runtime and consistent state
backups and follow its native controller recovery gate. Verify an existing pool
with both a failed preemption reconciliation and an independently successful
setup job: the healthy replica should become READY on the next probe pass while
the failed replica retains its original ownership and unresolved cleanup. Check
the owning managed job's actual model metrics after dispatch. Passing setup or
publishing READY alone does not establish resumed training.
