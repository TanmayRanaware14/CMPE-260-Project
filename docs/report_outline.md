# Tentative Report Outline

1. Introduction: motivation, research question, contributions
2. Background: LLVM optimization, PPO, ML-guided compilation
3. Problem formulation: state, action, reward, episode budget
4. Experimental setup: CompilerGym version, splits, PPO config, baselines, metric
5. Results: baseline comparison, learning curves, per-benchmark results
6. Distribution shift: held-out programs, other benchmark families, generalization gap, failures
7. Discussion: sampled vs argmax policy, greedy vs PPO, limitations
8. Threats to validity: instruction count vs runtime, benchmark similarity, small test set,
   sampled greedy candidates, non-matched `-Oz`/`-O3` budgets
9. Conclusion and future work: action masking, larger budgets, runtime measurement
