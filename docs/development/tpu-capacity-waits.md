# Durable Cloud TPU capacity waits

API launches persist an allocation intent in `~/.sky/gcp_queue_waits/` before
the provider create call. While capacity is pending, they raise the native
`ExecutionPausedError` and poll from the shared asynchronous condition loop.
The launch executor is available for other requests. The original request,
queued-resource names, provider creation timestamps, nodes, route, owner labels,
input contract, and bounded deadlines remain unchanged on resume. An uncertain
create, missing resource, changed generation, or changed contract parks for
reconciliation without replacement or teardown. Native teardown retires the
journal only after the complete cluster's queued resources are absent.

For a persistent **single-supervisor** API installation, set
`SKYPILOT_DURABLE_TPU_REQUESTS=1` in its service environment. This retains the
request database, logs, client uploads and blobs across restarts. Back these up
with the native cluster/job databases and the queue journals. Do not enable
this mode on multiple API replicas sharing one request backend. Startup
requeues the original IDs for never-started launches/pool applications and
journaled capacity waits, before managed controller recovery starts. Requests
that already left the capacity-wait phase require reconciliation and are not
automatically replayed. User cancellation remains terminal.

Consolidated managed controllers reattach to the original API launch before
recovery cleanup. The stored request must match the exact native job/task
observation target, owner and cluster name. Completed launches also require
the current native cluster generation. Ambiguous bindings wait for
reconciliation. During the native restart gate, controller shutdown preserves
in-flight launches; an explicit CANCELLING task still cancels its request.

Before the first upgrade from a version without these journals, audit every
running launch against its native launch attempt, cluster generation, original
creation log, immutable task, and exact provider request incarnation. Import
only proven capacity waits into the journal while the service is stopped;
never infer ownership from a cluster-name prefix. Keep the service's native
controller recovery gate in place throughout maintenance. Preserve queued
requests and uploads: the old version's normal shutdown cancels API requests,
and its startup erases the local request database. Rollback therefore requires
the preserved state and an explicit reconciliation of every outstanding intent.

Pool operation threads use their actual start identity, rather than the persisted
replica status, to distinguish admission waits from completed operations. A
recovered or delayed status write must not restart a completed thread or prevent
the refresher from collecting peer operations. Terminal launch/teardown outcomes
still follow the normal ownership and cleanup checks; a stale status never
authorizes replaying a cloud request.

Controller recovery also advances the next worker ID past every retained
replica, including failed and terminating workers. A replacement must not
overwrite their records or collide with an original operation still in flight.

## Duplicate pool bindings

The cloud-label allocation lock serializes new queued TPU creates with native
teardown. READY journals continue to own their native incarnation; another
incarnation parks with its original API request, without failover or cleanup.
Existing nodes require the exact durable allocation receipt. New GCP names use
a longer truncated-name hash; retained handles and YAMLs keep their old names.
Hash length is not an ownership proof.

Do not use down/purge to remove a logical alias of a live TPU. The internal
native controller action `POST /controller/retire_replica_alias` accepts the
complete `sky.serve.pool_alias.AliasRetirement` binding. It checks the pool
hash, owner, workspace, version, both native cluster generations, original
successful launches and equivalent immutable inputs, original queue journal,
and exact ACTIVE queue/READY node identities and creation times. It requires
zero nonterminal pool jobs, active worker API requests, or outstanding launch
or teardown threads. Operators must separately verify current all-guest
identity and idleness before invoking this administrative action.

The controller holds its probe lock, pool scheduling lock, both native cluster
locks, and allocation lock. It first commits a private retirement intent in
`~/.sky/serve/alias_retirements/`. That intent quarantines the alias from job
selection and reserves its replica ID. It archives the exact alias record and
removes only the active logical alias registration. Original launch requests,
attempts/history, worker inputs, keeper journal, and provider allocations remain.
No provider lifecycle RPC or request cancellation is part of this action.

Back up retirement receipts with the native databases and capacity journals.
A lost reply is retried with the identical binding; a completed receipt is
returned without replaying changes or disturbing later jobs.
Receipt reconciliation precedes worker-operation checks and cluster locks; it
does not clear newer controller bookkeeping or block on subsequent launches,
teardowns, API operations or pool versions under the same owner/incarnation.
Controller startup
can finish a prepared intent after rechecking its fences. Changed ownership,
active jobs, or unavailable evidence retains quarantine for reconciliation.
Committed and prepared receipts both advance the replica ID high-water mark.

Keep pool dispatch held through a reviewed rollout and fresh guest audit.
After retirement, verify distinct physical allocations, unchanged provider
generations and guest boots, preserved requests/history, and unrelated jobs.
Do not roll back to code that ignores retirement intents while a prepared
intent remains or an old alias can be resurrected. Prefer completing the same
forward repair; restoring a duplicate active row is not a safe rollback.

## Manual verification

1. Run the GCP queued-resource and request executor/restart unit tests in an
   isolated HOME. They use fake provider responses and allocate no hardware.
2. Submit two canonical, quota-admitted TPU launches where hardware is pending.
   Observe `WAITING`, cleared executor PIDs, and availability for another API
   launch/pool application. Record the original API IDs and exact provider
   names/creation times; repeated polls must not create resources.
3. Restart the single service with its recovery gate and consistent backups.
   Confirm the same API IDs, task hashes, native attempts, and queued-resource
   incarnations survive. Verify API health, dashboard and native job ownership.
4. When hardware arrives, confirm the original requests proceed to setup and
   readiness. Cancel a separate explicitly disposable test through its owning
   controller; verify it never resumes and native teardown removes its exact
   queued resources. Do not cancel existing experiments as a test.
5. After pool recovery, observe several refresher cycles. Completed launch and
   teardown operations must leave their thread pools even when their saved
   replica status was stale. Verify no repeated thread-start errors, and match
   each replacement to its original request or a safely retired prior generation.
   New worker IDs must exceed all worker IDs retained at controller startup.

## Terminal allocation reconciliation

Native pool allocations persist their original pool hash, owner, workspace,
worker/version, immutable launch digest and native generation in the queue
journal before creation. Controllers reconcile at most two eligible receipts
per pass, with a persisted five-minute recheck interval. Cleanup is independent
of a mutable native cluster row. A missing row cannot certify teardown; native
state remains until the allocation journal is retired.

The reconciler holds the pool scheduling, native cluster and allocation locks.
It requires a terminal original creator, unchanged ownership and inputs, no
assigned jobs, and no executing or unrelated worker API operation. An exactly
equivalent successor parked in WAITING retains its original request. Every exact
queue generation must be FAILED/SUSPENDED or absent, and every guest must be
absent or independently PREEMPTED with matching identity. Live, changed or
unknown guests block cleanup. Deletion intent and operation handles survive
restart. A lost reply permits retirement only after independent absence; it
never blindly repeats deletion. Retire the journal only after every request and
guest is confirmed absent. Preserve requests and native history throughout.
Request retention excludes every creator referenced by an unretired journal.

Older receipts require an explicit reviewed enrollment using
`sky.serve.tpu_cleanup.bind_retained_allocation`. Supply the exact journal
SHA-256 and original `PoolAllocationOwner`. The native history, provider launch
lineage, original terminal request and frozen pool definition must agree.
Reconcile missing history from verified archives first; never fabricate an owner
from a resource prefix or clear an uncertain create. A durable
`cleanup_error` and `cleanup_recheck_at` expose unresolved receipts.

Manual verification: run `tests/unit_tests/test_tpu_cleanup.py`, the GCP queued
resource/wait tests, the replica-manager tests and request retention/restart
tests in an isolated runtime. After deployment observe at least two controller
passes, retained original request IDs and immutable inputs, exact provider
deletion/absence, successor progress, and unchanged active jobs/checkpoints.

Durable Cloud TPU teardown honors existing native locks and never unlinks them
or cancels other cluster requests to acquire ownership. A busy lock retains
the original teardown for bounded retry. Explicit controllers cancel only their
exact owned creator before cleanup; maintenance and successor fences survive.
