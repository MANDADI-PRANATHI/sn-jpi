# SN-JPI reproduction

Independent clean-room reproduction of *Self-Normalized Jarzynski Policy Improvement for Nonstationary Reinforcement Learning*. The benchmark implements Section 6.1 at fixed temperature. The SHA-256 hashes in `source_sha256.txt` identify the paper and plan used.

## Run on Linux with Python 3.12

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-lock.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl
python checks.py
python theory.py
python run_matrix.py --workers 6
python report.py
python audit.py
python validate_results.py
```

The matrix contains 180 distinct training runs. Tables 2–4 share runs where the published schedules and methods coincide. Use `--protocol initial`, `--protocol dense`, `--protocol stress`, or `--protocol control` to run a subset. Add `--seed 0` to validate a full seed before launching the rest. The runner skips completed runs in the selected result directory and restarts incomplete runs from initialization. Use a fresh `--results` directory if you change the code or settings. For machines with limited RAM, reduce the worker count. Reports show an experimental row only after all its required seeds are complete. Figure panels require every matching series.

## Paper settings and implementation decisions

| Setting | Value or decision |
| --- | --- |
| JPI | Fixed β = 0.8, actor epochs 1/3/6, full-batch Adam and a fresh complete-episode batch after each update |
| Policy | Shared 64×64 tanh network, Gaussian with learned state-independent log standard deviation, PyTorch linear-layer default initialization, initial log standard deviation 0 |
| Training score | Undiscounted episode reward sum standardized once per batch, with population standard deviation and epsilon 1e-8 |
| Value target | Discounted reward-to-go with γ = 0.99 and zero bootstrap even at a time limit |
| Local coefficients | Adam learning rate 3e-4, value coefficient 0.5, JPI KL coefficient 0.01, entropy coefficient 0.01, global gradient-norm cap 0.5 |
| PPO | Local clipped policy-ratio surrogate, 10 shuffled epochs, 64-transition minibatches, clip ε = 0.2, normalized fixed reward-to-go minus old value |
| TRPO | sb3-contrib MlpPolicy, 64-transition critic minibatches, target KL 0.01, learning rate 3e-4, γ = 0.99 and pinned package defaults for remaining settings |
| SAC | Stable-Baselines3 MlpPolicy, Adam learning rate 3e-4, γ = 0.99, 100 training transitions of warm-up, replay capacity 100,000, batch 256 and one gradient step per eight environment interactions. Both variants use the same schedule |
| Change point | Start of the 1-indexed switch update. Discard the unfinished SB3 environment episode at that boundary. SAC-reset clears replay while retaining parameters, optimizers and the transition clock |
| Raw JPI epochs | Six throughout, including Table 2 whose raw rows match Table 4. Table 5 and stress protocols use this same undisclosed-setting assumption |
| Episode caps and action | 200 steps in all runs. Clip to bounds and apply the action-sign switch in the wrapper. JPI/PPO likelihoods use the latent Gaussian action |
| Evaluation | Deterministic policy mean, fresh reset and matched seeds across algorithms. MuJoCo reports length in sparse and dense tasks. Control Suite reports summed native reward |
| Seeds | 0–2 or 0–4 for training. Local update resets use seed + 1009 × update. Evaluation uses 1,000,000 + 100,000 × seed + 10 × update + episode index |
| Runtime | CPU PyTorch, one thread per process, deterministic algorithms enabled. Dependency versions are pinned. GPU behavior was outside the CPU reproduction |

The paper specifies the network structure, estimator, action transformation, reward formula and main schedules. It leaves optimizer coefficients, initialization details, original seeds, baseline schedules, Gaussian evaluation mode and software versions open. The choices above are frozen assumptions. The dense MuJoCo cap follows the same protocol as the sparse comparison. Section 6.2's proposed safeguards are inactive.

## Outputs

The machine-readable `configuration.json` lists every planned run and the fixed settings.

- `results/AUDIT.md` summarizes coverage, discrepancies and interpretation.
- `results/tables/` and `results/figures/` contain generated results. `paper_reference.csv` is used only for discrepancy comparisons.
- `results/raw/<run>/` contains the configuration manifest, per-episode evaluation, optimizer diagnostics, final checkpoint and completion marker. Training interactions exclude evaluation steps.
- `results/seed_summaries.csv` and `results/discrepancies.csv` contain seed summaries and paired published/reproduced values. Reported SE uses population standard deviation divided by √n and is descriptive.
- `results/theory.json` records the synthetic generator, temperature limit checks and first-order bias simulation. The original synthetic design was undisclosed.
- `results/checks.log` records the executed infrastructure and trainer checks.
