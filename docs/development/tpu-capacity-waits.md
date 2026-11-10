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
