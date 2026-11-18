# Experiment Tracking

For every experiment record: ID and date, git commit, Python and dependency versions,
splits, seeds, reward definition, step budget, hyperparameters, output files, deviations.

The scripts already save config snapshots next to results
(`results/baselines.config.json`, `results/models/train_config.json`).
Copy a finished run's `results/` and `figures/` into `experiments/<id>/` before starting another.

Do not compare runs that used different reward definitions, step budgets or splits without
documenting the differences. Never mix mock-backend and real results.
