# AGENTS.md — PPO_Robot (bird biped, custom SAC)

Guidance for AI coding agents working in this repo. Read before editing.

## What this project is
Reinforcement learning for a bird-like **bipedal walker** simulated in **MuJoCo**
(`simulation/robot.xml`). Despite the folder name, the training/eval path uses a
**custom SAC** agent (soft actor-critic), not SB3 PPO. A separate, simpler SB3-PPO
experiment lives in the `simulation` (`test.py` + `robot.py`).

## Hardware target (drives reward/limits)
- Legs: 90 KV BLDC + 10:1 planetary + 20 A (ODrive FOC) ≈ **18 Nm continuous** per hip/knee.
- Tail & neck: **servos** — non-load-bearing, COM-shift only (low torque, slow).
- ~4.2 kg robot, **25 Hz** control (`timestep 0.002` × `frame_skip 20` = 0.04 s).
- The env models this: leg `forcerange ±18` randomized per reset `N(18,2)` clipped `[13,19.8]`;
  tail/neck `forcerange ±8`, low `kp`/high `kv` (soft, slow servos). Don't raise these
  past hardware reality — it trains a policy the robot cannot execute (sim2real gap).

## Run commands (from this directory)
```bash
python train_mujoco.py        # train in sim (device auto-selects cuda, falls back to cpu)
python evaluate_mujoco.py     # MuJoCo viewer eval; auto-picks latest .pth in models/
python evaluate_mujoco_key.py # viewer eval with keyboard command override (manual_control)
python test_mujoco.py         # env smoke test (sine drive, no policy)
```
There is no test suite / linter. Quick sanity check = the commands above run, plus:
`python -m py_compile *.py simulation/robot.py`.

## Architecture (active files)
- `config.py` — all hyperparameters (`Config`). Edit here, not inline. Also holds
  `REWARD_WEIGHTS`, the domain-randomization ranges, and the curriculum settings.
- `sac_agent.py` — SAC. **`update_from_buf(norm_mean, norm_std, buffer)` is the only update
  path used** (3 args). There is no `update()` anymore. One merged `critic_optimizer` covers
  both Q-nets (equivalent to two, since `q1_loss + q2_loss` backprops together through
  disjoint params); the target update is a fused `torch._foreach_*` pair.
- `models.py` — `GaussianActor` (residual trunk, tanh-squashed) + `TwinQNetwork` (late fusion).
- `replay_buffer.py` — GPU-resident circular buffer; stores **raw** (unnormalized) states.
- `inputNorm.py` — `RunningMeanStd`. Normalization happens at update time, not store time,
  so the normalizer is **frozen** after `Config.NORM_UPDATE_STEPS` (`freeze()`). A still-drifting
  normalizer re-scales every buffered transition and makes the TD target non-stationary.
- `historyWrapper.py` — stacks last `history_len` (obs, action) pairs; policy input is
  `obs_dim*H + action_dim*H`. `Config.HISTORY_LEN = 1` → 30 dims, i.e. the policy sees
  `[current_obs, previous_action]`. **Changing it changes `state_dim` and invalidates checkpoints.**
- `utils.py` — `update_params` helper, `seed_everything`.
- `simulation/robot.py` — `BirdBipedEnv` (the reward, gait shaping, domain randomization).
- `simulation/robot.xml` — MuJoCo model, incl. foot `site`/`contact`/`framepos`/`framelinvel`
  sensors; `framelinvel` now feeds the slip penalty and `contact` feeds stance detection.

## Reward design (read before touching `simulation/robot.py`)
The task terms are **baseline-subtracted** so that doing nothing scores exactly **0**:

```
sigma_v   = max(sigma_min_speed, sigma_frac_speed * |v*|)
reward_v  = exp(-((vx - v*)/sigma_v)^2) - exp(-(v*/sigma_v)^2)      # 0 iff vx == 0
reward_y  = exp(-(yaw_err/sigma_y)^2)     - exp(-(yaw_err_ref/sigma_y)^2)
```
`yaw_err_ref` is the heading error captured when the command was issued, i.e. what a robot
that never rotated would still be carrying. The remaining terms are deliberately small
(`w_posture` 0.20, `w_alive` 0.05, `w_stance` 0.20) so they cannot out-earn locomotion.

**Do not reintroduce a constant "alive"/"heading" bonus or a flight penalty.** The previous
reward paid a motionless robot `alive + heading + feet ≈ 1.0`/step while a walking gait
earned `≈ 0.06`/step more from tracking — measured across 12 command/kernel configurations,
standing still strictly dominated every open-loop gait. Stance is now rewarded *only* when the
command asks the robot to hold still, and walking is shaped by a **slip** penalty on loaded
feet instead of by punishing flight.

The tracking kernel is `sigma`-scaled by the command magnitude on purpose: a fixed-width
Gaussian has headroom `1 - exp(-k*v*^2)`, which collapses to ≈0.09 at 0.3 m/s and is then
smaller than the cost of walking, so slow gaits stay net-negative however well they track.

Commands come from a **speed curriculum** (`cmd_speed_max`, seeded at 0.3 m/s) that grows
only when the policy's normalized tracking quality clears `CURRICULUM_UP_THRESHOLD`.

## Algorithm notes (PPO → SAC)
- Off-policy + auto-alpha. `TARGET_ENTROPY_FRAC = -0.5` is deliberate: a tanh-squashed policy
  in `[-1,1]^6` has a **maximum** differential entropy of `6·ln2 ≈ 4.16`, so the old `-1.0`
  (target −6) was unreachable and drove alpha to 0. Don't set the target below `-ln2·action_dim`.
- `GRADIENT_STEPS = 2` (UTD 2). Measured ≈130 updates/s on CUDA, ≈26 on CPU — the env itself
  runs ≈4200 steps/s, so CPU is gradient-bound at ~5× lower throughput.
- No reward normalization (only observation normalization). `REWARD_SCALE` is the knob keeping
  Q-targets in range; keep per-step reward O(1).
- Deterministic physics + per-reset randomization of torque cap, foot friction, motor `kp` and
  actuator filter lag — fine for SAC, and the kp/friction/lag ranges are the sim2real lever.

## Sim2real caveats
- Tail/neck must stay servo-band (`±8 Nm`, soft). Keep them non-load-bearing.
- The observation deliberately **excludes** base linear velocity and base height: on real
  hardware neither is directly measurable without a state estimator. If you add them for
  training, you must add the matching estimator to the real-robot path or the policy will
  not transfer.

## Conventions
- Hyperparameters live in `config.py`; don't hardcode new magic numbers in scripts.
- Do NOT commit weights/logs — `.gitignore` already excludes `runs/`, `*.pth`, `*.zip`, `*.pkl`,
  tfevents. Don't force-add them.

## Do not touch / known-dead (see README.md)
- `_archive/` — Godot-era TCP/threaded trainers, the Ant-v5 `evaluate_AntV5_example.py`,
  concatenation dumps (`comb*`), stale XML copies. Reference only; some are broken by design.
- `tcp/` — socket transport for the planned physical robot; still intended for future use.
- `evaluate_tcp.py` — real-robot eval path.
- `optim/lion.py` — present but currently unused.

## Known bugs elsewhere
- `_archive/train_tests_threaded*.py` call `update_from_buf(norm, buf)` (2 args) — broken vs
  the current 3-arg signature. Kept for reference; not part of the active path.
