
# Distribution-Shift-Aware Reinforcement Learning for Compiler Pass Ordering

This project investigates whether reinforcement learning can learn LLVM optimization-pass sequences that generalize to unseen programs.

## Research question

Can a PPO agent learn optimization-pass sequences that improve a compiler optimization objective and generalize to held-out benchmark programs?

## Initial scope

- Environment: CompilerGym LLVM (`llvm-v0`)
- Observation: LLVM Autophase features
- Initial reward: `IrInstructionCountOz`
- RL algorithm: Proximal Policy Optimization (PPO)
- Baselines: random action selection and greedy action selection
- Evaluation: held-out programs, optimization budgets, and multiple random seeds

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Test the environment:

```bash
python scripts/smoke_test.py
```

## Random baseline pilot

`scripts/run_random_baseline.py` connects the configuration loader, CompilerGym environment, and random policy. It runs one episode for every training benchmark and seed. Validation and test programs are left unused. Each episode applies at most `max_episode_steps` randomly chosen LLVM passes, then records the result.

Run it from the repository root:

```bash
python scripts/run_random_baseline.py --config configs/default.json --seeds 7 17 27 --output results/random_baseline.csv
```

Those seeds and that output path are the defaults. Generated files under `results/` stay ignored. The same command with `--output docs/checkin2/random_baseline.csv` writes the check-in copy, which is tracked with the runner.

Important CSV fields:

- `benchmark`, `split`, and `seed` identify the training program and the action-selection seed. `split` is `train` for this pilot.
- `steps_taken` is the number of passes applied. `actions` is a JSON list of pass indices, so a sequence can be inspected or replayed.
- `initial_ir_count` and `final_ir_count` are raw `IrInstructionCount` observations taken on the original program and after the sequence. `instruction_delta` is `initial_ir_count - final_ir_count`. `instruction_reduction_pct` is that delta as a percentage of the initial count, and only when the initial count is positive. A negative percentage means the sequence increased the instruction count.
- `total_reward` is the sum of per-step `IrInstructionCountOz` rewards. That reward is scaled against the `-Oz` baseline, so it is a different number from `instruction_reduction_pct`.
- `elapsed_seconds` is wall-clock time for the optimization procedure: resetting to the original program and applying the sampled passes. It is not the execution time of the compiled program, and it does not include environment construction or the separate instruction-count reads.
- `status` is `ok`, `error`, or `aborted`. A failed or aborted row keeps the benchmark, seed, and `error` text. Counts, the percentage, and the cumulative reward are left blank when that optimization outcome is not available. Blank cells are missing measurements, not zeros.

A negative `instruction_reduction_pct` means the sampled sequence increased the IR instruction count.

## Workflow

1. Verify CompilerGym installation and benchmark access.
2. Collect initial program observations.
3. Implement and evaluate baseline methods.
4. Train PPO.
5. Evaluate on held-out programs.
6. Analyze results and create plots.
