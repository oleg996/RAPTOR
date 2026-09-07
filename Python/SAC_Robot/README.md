# PPO_Robot — Bird Biped Locomotion (custom SAC)

Reinforcement-learning training / evaluation for the bird-like bipedal robot defined
in `simulation/robot.xml`. Despite the folder name, the active algorithm is a **custom
SAC** agent (not SB3 PPO). The SB3-PPO experiment lives in the parent `qwen_test/`
folder (`test.py` + `robot.py` + `bird_ppo.zip`).

## Entry points

| Purpose | Run | Notes |
|---|---|---|
| Train (sim)        | `python train_mujoco.py`        | Uses `simulation/robot.xml`, history-stacked obs, GPU (see `config.DEVICE`). |
| Evaluate (sim)     | `python evaluate_mujoco.py`     | MuJoCo viewer, auto-picks latest `.pth` in `models/`. |
| Env smoke test     | `python test_mujoco.py`         | Sine-wave drive, no policy. |
| Evaluate (real)    | `python evaluate_tcp.py`        | Real-robot eval over TCP (uses `tcp/`). |

## Layout

- `config.py`        — all hyperparameters (`Config`).
- `sac_agent.py`     — SAC agent (`update_from_buf` is the active update path).
- `models.py`        — `GaussianActor` (residual + tanh-squashed) / `TwinQNetwork`.
- `replay_buffer.py` — GPU-resident circular buffer.
- `inputNorm.py`     — `RunningMeanStd` observation normalizer.
- `historyWrapper.py`— stacks the last N (obs, action) pairs into the policy input.
- `utils.py`         — `update_params` helper (was `untils.py`, typo fixed).
- `optim/lion.py`    — Lion optimizer (currently unused).
- `tcp/`             — socket transport for sim2real / real-robot data (`comm`, `decoder`, `Tcp_env`).
- `simulation/`      — MuJoCo model + `BirdBipedEnv`.
- `_archive/`        — generated / stale artifacts (kept, not used).

## Caveats (read before trusting a run)

- `simulation/{robot.py,robot.xml}` now **mirror** the top-level `../robot.py,../robot.xml`
  (gait shaping + per-reset leg-torque randomization + foot sensors). Keep the two in sync.
- The long-step reward was tuned under PPO; SAC may need the `w_*` weights retuned after
  training (see `simulation/robot.py` and `AGENTS.md`).
- `runs/` and `models/` are gitignored (large logs / weights).
- Godot-era TCP/threaded trainers and the `Ant-v5` evaluator live in `_archive/` (reference
  only; some are intentionally broken vs the current `sac_agent` API). See `_archive/README.txt`.

## Sim2real status

Hardware target: 90 KV BLDC + 10:1 planetary + 20 A (ODrive) ≈ 18 Nm continuous per leg
joint; tail/neck on servos (COM shift only, lower torque/speed). The top-level `../robot.py`
models this (torque clamp + randomization); this copy does not yet.
