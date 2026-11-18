# Environment Exploration Log

Fill this in as you run things. Leave fields blank until you have actual observations.

## System information

- Date:
- Computer / processor:
- OS version:
- Python version:
- CompilerGym version:
- Installation command:

## Smoke test (`python scripts/smoke_test.py`)

- Benchmark URI:
- Environment initialized:
- Observation shape (expected 56):
- Action-space type and size (expected discrete, 124):
- Reward space:
- `env.fork()` works:
- Errors and resolutions:

## Benchmark check (`python scripts/smoke_test.py --check-config`)

- URIs that failed (and replacements):
- Instruction counts look sane (init > oz for most programs):

## Reward semantics check

- Does cumulative reward ≈ (initial − final) / (initial − oz) on a few programs?
- Programs with a tiny (initial − oz) denominator (huge rewards):

## Timing

- Seconds per step:
- Seconds per `fork()`:
- Seconds for the full baseline run:

## Follow-up tasks

- Record compatibility issues.
- Decide whether `greedy_candidates` can be `null` (all actions) given fork cost.
