# PPO_Robot — Bird Biped Locomotion (custom SAC)

Reinforcement-learning training / evaluation for the bird-like bipedal robot defined
in `simulation/robot.xml`. Despite the folder name, the active algorithm is a **custom
SAC** agent (not SB3 PPO).A separate, simpler SB3-PPO
experiment lives in the `simulation` (`test.py` + `robot.py`).

## Entry points

| Purpose | Run | Notes |
|---|---|---|
| Train (sim)        | `python train_mujoco.py`        | Uses `simulation/robot.xml`, obs normalization, auto cuda/cpu. |
| Evaluate (sim)     | `python evaluate_mujoco.py`     | MuJoCo viewer, auto-picks latest `.pth` in `models/`. |
| Evaluate (sim, manual) | `python evaluate_mujoco_key.py` | Arrow keys override speed/heading command. |
| Env smoke test     | `python test_mujoco.py`         | Sine-wave drive, no policy. |
| Evaluate (real)    | `python evaluate_tcp.py`        | Real-robot eval over TCP (uses `tcp/`). |

## Layout

- `config.py`        — all hyperparameters (`Config`), reward weights, DR ranges, curriculum.
- `sac_agent.py`     — SAC agent (`update_from_buf` is the active update path).
- `models.py`        — `GaussianActor` (residual + tanh-squashed) / `TwinQNetwork`.
- `replay_buffer.py` — GPU-resident circular buffer.
- `inputNorm.py`     — `RunningMeanStd` observation normalizer (frozen after warm-up).
- `historyWrapper.py`— stacks the last N (obs, action) pairs into the policy input.
- `utils.py`         — `update_params`, `seed_everything`.
- `optim/lion.py`    — Lion optimizer (currently unused).
- `tcp/`             — socket transport for sim2real / real-robot data (`comm`, `decoder`, `Tcp_env`).
- `simulation/`      — MuJoCo model + `BirdBipedEnv`.
- `_archive/`        — generated / stale artifacts (kept, not used).

## Caveats (read before trusting a run)
- **Existing checkpoints are invalid.** The reward and the commanded-observation semantics
  changed (baseline-subtracted tracking rewards, no command smoothing ramp), so `models/*.pth`
  must not be resumed from. `LOAD_MODEL` defaults to `False`.
- The reward is deliberately baseline-subtracted so that idling scores **0**. Watch
  `reward/speed_component` in TensorBoard: if it sits at 0 the policy has learned to stand
  still, which is no longer a reward-maximizing strategy but is the first thing to check when
  returns look flat.
- `runs/` and `models/` are gitignored (large logs / weights).
- Godot-era TCP/threaded trainers and the `Ant-v5` evaluator live in `_archive/` (reference
  only; some are intentionally broken vs the current `sac_agent` API). See `_archive/README.txt`.

## Sim2real status

Hardware target: 90 KV BLDC + 10:1 planetary + 20 A (ODrive) ≈ 18 Nm continuous per leg
joint; tail/neck on servos (COM shift only, lower torque/speed).
