# Project Specifications

## Research question

Can a reinforcement-learning policy trained on a subset of LLVM benchmark programs learn
optimization-pass sequences that generalize to unseen programs, and how does it compare with
LLVM's own pipelines and simple search strategies under a matched step budget?

## Environment and data

CompilerGym `llvm-v0`. Train on 12 cBench programs; evaluate on held-out cBench programs
(validation/test) and on other benchmark families (NPB, BLAS) as a distribution-shift split.
Confirm every URI with `python scripts/smoke_test.py --check-config`.

## Markov decision process

- State: `log1p(Autophase)`, fraction of budget used, normalized action histogram.
- Action: one of CompilerGym's discrete LLVM passes.
- Transition: apply the pass; observe the new program.
- Reward: `IrInstructionCountOz` per step (clipped for training). Verify its semantics.
- Termination: fixed budget of 20 steps, or environment termination.

## Methods

PPO (Stable-Baselines3, MLP); random; best-of-N random search; step-wise greedy via `env.fork()`;
LLVM `-Oz` / `-O3` as references (not budget-matched).

## Evaluation

Primary metric: `pct_vs_oz` on IR instruction count (see README). Report mean ± std over PPO
seeds, the generalization gap (train − test), the fraction of programs beating `-Oz`, and both
sampled and argmax PPO. Evaluate train, validation, test and shift splits.

## Out of scope until the pipeline is validated

Action masking, multi-objective rewards, execution-time measurement, larger benchmark suites,
budget-sensitivity sweeps.
