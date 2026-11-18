# Reinforcement Learning for LLVM Compiler Pass Ordering

Can a PPO agent learn LLVM optimization-pass sequences that reduce IR instruction
count and **generalize to programs it never saw during training**?

The repo trains PPO on a set of programs, then compares it with random, best-of-N
random, greedy-search and LLVM `-Oz`/`-O3` on held-out programs, under a matched
step budget and with multiple seeds.

## Status: read this first

| Piece | State |
|---|---|
| Pipeline code (env wrapper, baselines, PPO, evaluation, plots) | Written; **15 unit tests pass** and every script runs end to end on the **mock backend** |
| Real CompilerGym backend | **Not yet run.** CompilerGym is archived; the first thing to do on your machine is `scripts/smoke_test.py` |
| Benchmark URIs in `configs/default.json` | Plausible, **unverified**. `smoke_test.py --check-config` tells you which ones load |
| Results | **None yet.** Numbers from the mock backend say nothing about LLVM |

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # numpy, gymnasium, SB3, torch, pandas, matplotlib

# 1. Prove the whole pipeline works, no CompilerGym needed (~3 min)
python -m pytest -q tests
python scripts/smoke_test.py --mock
python scripts/run_baselines.py --mock
python scripts/train_ppo.py --mock --seeds 7 --timesteps 30000
python scripts/evaluate.py --mock
python scripts/make_plots.py
```

Then switch to the real environment (see [CompilerGym setup](#compilergym-setup)) and drop `--mock`:

```bash
python scripts/smoke_test.py                 # one benchmark, one action, fork()
python scripts/smoke_test.py --check-config  # every URI in the config
python scripts/collect_features.py
python scripts/run_baselines.py              # add --skip-greedy first if fork() is slow
python scripts/train_ppo.py                  # 3 seeds by default
python scripts/evaluate.py
python scripts/make_plots.py
```

> The mock backend (`src/compiler_pass_rl/mock_env.py`) imitates CompilerGym's API so the
> code can be tested without LLVM. **Never report mock numbers as results.**

## Design

- **State**: `log1p(Autophase)` (56) + fraction of budget used (1) + normalized action histogram (124).
  Raw Autophase counts span orders of magnitude and train badly with an MLP.
- **Action**: choose one of CompilerGym's discrete LLVM passes.
- **Reward**: `IrInstructionCountOz` per step, clipped to ±`reward_clip` for training only.
  Cumulative reward ≈ `(initial − final) / (initial − oz)`; verify this on your install.
- **Episode**: fixed budget of `max_episode_steps` (default 20).
- **Training distribution**: every `reset()` samples a program from the **train split**, so PPO
  learns a policy rather than memorizing one program.
- **Splits**: `train` / `validation` / `test` (same dataset, different programs) plus an optional
  `shift` split from other benchmark families (NPB, BLAS) for genuine distribution shift.
  `get_splits` raises if train overlaps any held-out split.

## Metric

`pct_vs_oz = 100 × (oz_count − final_count) / oz_count` on IR instruction counts.
Positive = fewer instructions than LLVM `-Oz`; 0 = parity. Also reported: `pct_vs_init`,
geometric-mean ratio vs `-Oz`, and the fraction of programs that beat `-Oz`.

**This is not runtime and not binary size.** Do not claim speedups from it.

## Methods compared

| Method | Notes |
|---|---|
| `random` | Uniform random actions, same step budget |
| `random_search_N` | Best of N random episodes. Uses N× the steps, so it is a *stronger*, not budget-matched, baseline |
| `greedy` | Each step: try candidate actions on `env.fork()` copies, keep the best; stop when none improves. Samples `greedy_candidates` actions per step (set `null` to try all) |
| `ppo` | The trained stochastic policy (actions sampled), averaged over episodes and seeds |
| `ppo_argmax` | Same policy, always taking the most likely action |
| `llvm_oz` / `llvm_o3` | The compiler's own pipelines. Reference only: they use far more than 20 passes |

**Why both PPO rows?** Passes are mostly idempotent, and an argmax policy tends to repeat one
action and waste its budget, while the sampled policy PPO trained as does not. On the mock this
was a 30+ point difference. Report both; don't silently pick the better one.

## CompilerGym setup

CompilerGym is archived and targets older Python and `gym`. Expect friction.

- Try Python 3.8/3.9 in a separate environment (`requirements-compilergym.txt` has *suggested,
  unverified* pins). Prebuilt wheels existed for Linux and Intel macOS.
- **Apple Silicon:** I'm not aware of a native build. Use a Linux x86-64 container/VM, or Rosetta.
- The wrapper talks to the raw CompilerGym env directly and never imports `gym`, so Gymnasium /
  Stable-Baselines3 versions do not have to match CompilerGym's `gym`.
- Log what you did in `docs/environment_exploration.md`.

## Layout

```
configs/default.json      all experiment settings, splits, seeds
src/compiler_pass_rl/
  environment.py          compat helpers + CompilerPassEnv (Gymnasium wrapper)
  mock_env.py             CompilerGym-API stand-in for tests/dev
  baselines.py            random, random search, greedy, -Oz/-O3 references
  ppo_agent.py            Stable-Baselines3 PPO
  evaluation.py           metrics, PPO evaluation, summaries, generalization gap
  plotting.py             figures
scripts/                  smoke_test, collect_features, run_baselines, train_ppo, evaluate, make_plots
tests/                    15 tests (mock backend)
docs/, experiments/       specs, exploration log, report outline, run tracking
```

## Caveats to state in any write-up

- Instruction count ≠ runtime or code size.
- cBench programs share code and domains (e.g. `stringsearch`/`stringsearch2`, `blowfish`/`rijndael`),
  so same-dataset "test" programs are only mildly out-of-distribution.
- Greedy samples candidate actions by default; say so, or run with `greedy_candidates: null`.
- `-Oz`/`-O3` are not step-budget-matched. Say so.
- Report mean ± std over seeds, never a single run.
