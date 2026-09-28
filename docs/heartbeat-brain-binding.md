# Hypercube Brain heartbeat binding

The automatic heartbeat now calls the existing `HypercubeBrainEngine` every
cycle. The installed wheel includes `hypercube_brain` as well as `garvis`.

`RAW_PRE` supplies the read-only Git observation. During `PREDICT`, the brain
assesses the narrow claim that the configured repository is available, and
predicts the internal observation action's outcome from earlier verified cycles.
The existing prediction ledger freezes that assessment before internal execution.
`RAW_POST` carries the assessment into the cycle result. Sequence and return-phase
verification determine whether the binding calls `learn_action_effect`.
Failed or contradictory cycles do not train the binding.

The `brain` field in heartbeat health exposes the latest assessment. The
`brain_checkpoint` field stores the brain cycle count and verified outcome count
in the existing atomic heartbeat-state file. Restart reconstructs this narrow
action-effect memory. Working memory, episodes, and events are bounded in RAM;
they are not full durable conversational memory. Existing heartbeat state files
without brain history start a fresh brain binding and retain their sequence.

The SQLite prediction/result ledger is the durable recovery source. The atomic
JSON checkpoint stores `brain_result_rowid` alongside the learning counters.
On startup and before a new cycle, results after that cursor are reconciled in
commit order. A committed, verified sequence restores sequence progress; only
completed results with both verification flags and no contradictions train the
brain, and recovery still requires `LEARN` authority. Frozen predictions without
a committed result never train it. Older checkpoints use `last_cycle_id` to find
their ledger position, preventing previously learned cycles from being counted
again.

An interruption after SQLite commit or before checkpoint replacement therefore
replays the pending result on restart without executing its action again.
Counters and cursor are replaced together, so repeated restarts do not duplicate
learning. Recovery retains the result's original timestamp rather than reporting
old work as a fresh heartbeat. Keep the ledger and checkpoint together; ledger
deletion/replacement and concurrent services sharing one state directory are not
supported. This recovery covers process interruption, not storage corruption.

From the repository, after installing the project:

```bash
python -m garvis.heartbeat_cli run-once
python -m garvis.heartbeat_cli health
python -m garvis.heartbeat_cli daemon
```

The brain is advisory: it does not execute repair candidates, grant permissions,
or open network connections. Missing repository evidence remains contradicted;
the heartbeat itself can continue running. Repeated Git observations share one
source group and cannot become independent corroboration. Successful heartbeat
verification establishes only the internal sequence/phase outcome, not repository
health, general intelligence, consciousness, or the physical validity of the
creator's lattice hypothesis. The existing OAB computation is unchanged.

This integration connects the automatic heartbeat, not the conversational model
or camera/audio inputs. Those remain separate integration surfaces.
