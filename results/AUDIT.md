# SN-JPI reproduction audit

Completed **180 of 180** distinct training runs.
Each completed run has its evaluation episodes, optimizer diagnostics, manifest and final checkpoint.

## Estimator checks

Table 1 used 64 synthetic trajectories and β = 0.8. The constant-return SN objective was 3.00000000 and its gradient norm was 1.41e-16. The raw objective was 2.98479635 and its gradient norm was 0.15435. The small-β centered-gradient error at β = 1e-5 was 1.75e-06. Different synthetic draws and score features explain the numerical comparison limits.

The analytical gradient, extreme logits, local temperature slope, and relative-tail bounds passed. The CVaR and variance checks covered 100 synthetic distributions. The bias simulation used 200,000 replicates at each batch size. Its measured biases were within two Monte Carlo standard errors of the first-order prediction at the tested sizes.

## Benchmark discrepancies

| Table | Complete rows | Mean absolute final difference | Mean absolute post-switch difference |
| --- | ---: | ---: | ---: |
| 2 | 8/8 | 17.99 | 9.74 |
| 3 | 6/6 | 40.97 | 23.96 |
| 4 | 6/6 | 31.54 | 22.67 |
| 5 | 8/8 | 23.94 | 15.41 |
| 6 | 12/12 | 30.52 | 17.01 |
| 7 | 18/18 | 6.98 | 3.70 |

Differences use survival steps in Tables 2–6 and native return in Table 7. The comparison uses the paper’s rounded numbers. Original run logs, seeds, optimizer settings and evaluation details were unavailable. The matched implementation uses the explicit choices recorded in the configuration ledger and does not tune toward the reference values.

## Endpoint and recovery comparisons

| Table | Task | Highest final mean | Highest post-switch mean |
| --- | --- | --- | --- |
| 2 | InvertedPendulum-v5 | raw6 (70.47) | raw6 (30.57) |
| 2 | Hopper-v5 | raw6 (115.00) | ppo (63.03) |
| 3 | InvertedPendulum-v5 | sn6 (58.13) | sn6 (27.55) |
| 3 | Hopper-v5 | sn3 (113.00) | sn6 (53.91) |
| 4 | InvertedPendulum-v5 | raw6 (70.47) | raw6 (30.57) |
| 4 | Hopper-v5 | raw6 (115.00) | raw6 (56.99) |
| 4 | InvertedDoublePendulum-v5 | raw6 (13.93) | sn6 (11.00) |
| 5 | InvertedPendulum-v5 | ppo (77.60) | ppo (52.52) |
| 5 | Hopper-v5 | trpo (130.27) | trpo (64.73) |
| 6 | InvertedPendulum-v5 | sac_reset (128.13) | sac_reset (115.52) |
| 6 | Hopper-v5 | sac_reset (193.07) | sac_reset (168.47) |
| 7 | cartpole/swingup | trpo (33.48) | trpo (25.76) |
| 7 | walker/walk | trpo (11.43) | trpo (8.48) |
| 7 | cheetah/run | trpo (16.29) | trpo (6.70) |

## Matched six-epoch comparison

Positive differences favor SN-JPI. Paper differences use its rounded means.

| Table | Task | SN minus raw final | Paper final difference | SN minus raw post-switch | Paper post-switch difference |
| --- | --- | ---: | ---: | ---: | ---: |
| 4 | InvertedPendulum-v5 | -12.33 | +26.60 | -3.02 | +6.20 |
| 4 | Hopper-v5 | -10.60 | +41.10 | -3.08 | +21.00 |
| 4 | InvertedDoublePendulum-v5 | -0.13 | +10.70 | +0.29 | +3.90 |
| 5 | InvertedPendulum-v5 | +12.47 | -17.40 | +1.53 | -6.20 |
| 5 | Hopper-v5 | +18.93 | -4.40 | +15.89 | -19.90 |
| 6 | InvertedPendulum-v5 | +5.00 | +9.20 | -0.48 | -8.30 |
| 6 | Hopper-v5 | -7.33 | +25.10 | -4.59 | +28.40 |
| 7 | cartpole/swingup | +1.70 | -9.24 | -0.03 | -11.40 |
| 7 | walker/walk | -1.77 | -5.02 | -0.51 | -3.91 |
| 7 | cheetah/run | +0.28 | +5.56 | +0.42 | +4.16 |

## Interpretation

The rankings summarize these seeds and budgets. The experiments measure the response to a known action-sign switch. SAC-reset uses its known timing to discard replay, and both interaction overshoot and optimizer schedules vary by method. The descriptive comparisons do not establish general superiority or a causal relationship between tail weighting and recovery.
