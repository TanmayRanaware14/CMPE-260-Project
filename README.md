
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

## Workflow

1. Verify CompilerGym installation and benchmark access.
2. Collect initial program observations.
3. Implement and evaluate baseline methods.
4. Train PPO.
5. Evaluate on held-out programs.
6. Analyze results and create plots.
