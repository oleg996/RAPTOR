# AGENTS.md — PPO_Robot (bird biped, custom SAC)

Guidance for AI coding agents working in this repo. Read before editing.

## What this project is
Reinforcement learning for a bird-like **bipedal walker** simulated in **MuJoCo**
(`simulation/robot.xml`). Despite the folder name, the training/eval path uses a
**custom SAC** agent (soft actor-critic), not SB3 PPO. A separate, simpler SB3-PPO
experiment lives in the parent `../qwen_test` tree (`test.py` + `robot.py`).

## Hardware target (drives reward/limits)
- Legs: 90 KV BLDC + 10:1 planetary + 20 A (ODrive FOC) ≈ **18 Nm continuous** per hip/knee.
- Tail & neck: **servos** — non-load-bearing, COM-shift only (low torque, slow).
- ~4.2 kg robot, 50 Hz control (`timestep 0.002` × `frame_skip 20`).
- The env models this: leg `forcerange ±18` randomized per reset `N(18,2)` clipped `[13,19.8]`;
  tail/neck `forcerange ±8`, low `kp`/high `kv` (soft, slow servos). Don't raise these
  past hardware reality — it trains a policy the robot cannot execute (sim2real gap).

## Run commands (from this directory)
```bash
python train_mujoco.py        # train in sim (uses config.DEVICE; set cuda/cpu in config.py)
python evaluate_mujoco.py     # MuJoCo viewer eval; auto-picks latest .pth in models/
python test_mujoco.py         # env smoke test (sine drive, no policy)
```
There is no test suite / linter. Quick sanity check = the three commands above run, plus:
`python -m py_compile *.py simulation/robot.py`.

## Architecture (active files)
- `config.py` — all hyperparameters (`Config`). Edit here, not inline.
- `sac_agent.py` — SAC. **`update_from_buf(norm_mean, norm_std, buffer)` is the only update
  path used** (3 args). There is no `update()` anymore.
- `models.py` — `GaussianActor` (residual trunk, tanh-squashed) + `TwinQNetwork` (late fusion).
- `replay_buffer.py` — GPU-resident circular buffer; stores **raw** (unnormalized) states.
- `inputNorm.py` — `RunningMeanStd`. Normalization happens at update time, not store time.
- `historyWrapper.py` — stacks last `history_len` (obs, action) pairs; policy input is
  `obs_dim*H + action_dim*H` (H=25 → 700 dims). Changing `history_len` changes `state_dim`.
- `utils.py` — `update_params` helper.
- `simulation/robot.py` — `BirdBipedEnv` (the reward, gait shaping, torque randomization).
- `simulation/robot.xml` — MuJoCo model, incl. foot `site`/`contact`/`framepos`/`framelinvel`
  sensors used by the gait reward.

## The reward / gait shaping (the sensitive part)
The walker had a fast 2–4 Hz shuffle gait. Fixed by **reward shaping** in
`simulation/robot.py::_gait_reward()`, driven by the foot sensors in `robot.xml`:
- reward **stride length**, **swing duration**, **swing foot clearance**; penalize **backward
  foot speed at touchdown**; cut `cost_vertical_vel` 0.5→0.1 (strong vertical penalty forced
  the flat shuffle).
- Weights are the `self.w_*` / `self.min_swing` block in `__init__`. Tune incrementally;
  over-penalizing cadence makes it freeze/moonwalk.
If you change this reward, **retune from the PPO-tuned starting point and retrain** —
see algorithm note below.

## Algorithm note (PPO → SAC)
The reward/sensors are algorithm-agnostic, but the weights were sanity-checked under PPO.
SAC specifics that interact with reward changes:
- Off-policy + entropy (auto-alpha): event-driven bonuses (fired only at touchdown) can be
  slow to credit-assign early; if it stalls, raise exploration time or slightly raise the
  `w_*` weights rather than changing PPO logic.
- SAC is more sensitive to **reward scale/normalization**; there is no reward normalization
  here (only obs normalization). Keep rewards in a moderate range.
- The env is deterministic physics + per-reset torque/noise randomization — fine for SAC.

## Sim2real caveats
- Tail/neck must stay servo-band (`±8 Nm`, soft). Keep them non-load-bearing.
- If you add obs noise / action latency / dynamics randomization, put it in `reset()`/`step()`
  and keep it in sync with `../qwen_test/robot.py` (the two copies are intentionally mirrored).

## Conventions
- Hyperparameters live in `config.py`; don't hardcode new magic numbers in scripts.
- Env/model changes: keep `simulation/{robot.py,robot.xml}` and `../robot.py`,`../robot.xml`
  in sync (they are mirrored on purpose).
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
