"""A small, dependency-free stand-in for CompilerGym's LLVM environment.

Purpose
-------
CompilerGym is archived and hard to install on many machines. This mock
mimics the *API surface* this project uses (reset/step/fork/close,
``env.observation[...]``, ``env.action_space.n``) so that the entire
pipeline -- baselines, PPO training, evaluation, plotting -- can be developed,
unit-tested and run in CI without LLVM.

It is NOT a model of LLVM. Numbers produced with the mock backend say
nothing about real compiler behaviour and must never be reported as results.

Mock dynamics (deliberately simple but with learnable structure)
----------------------------------------------------------------
* A program is a vector of 51 category counts (like a coarse Autophase).
  Programs come in 4 "families" with different dominant categories, so
  held-out families exercise distribution shift.
* Each of the 124 actions is a no-op, a shrinking pass (reduces some
  categories) or an expanding pass (increases some categories).
* Some shrinking passes are far more effective if a specific prerequisite
  action was applied earlier (order dependence).
* Re-applying an action has no further effect (idempotent passes).
* "Oz" / "O3" baselines are fixed reference sequences, so an agent can
  beat them.
"""

import copy
import re
import zlib

import numpy as np

N_ACTIONS = 124
N_FEATURES = 56
N_CATEGORIES = 51
N_FAMILIES = 4
_TABLE_SEED = 1234
_OZ_LEN = 8
_O3_LEN = 11


def _build_action_table():
    rng = np.random.default_rng(_TABLE_SEED)
    table = []
    for action in range(N_ACTIONS):
        kind = str(rng.choice(["noop", "shrink", "expand"], p=[0.40, 0.50, 0.10]))
        targets = rng.choice(N_CATEGORIES, size=int(rng.integers(3, 7)), replace=False)
        if kind == "noop":
            factor = 0.0
        elif kind == "shrink":
            factor = float(rng.uniform(0.05, 0.45))
        else:
            factor = -float(rng.uniform(0.05, 0.25))
        prereq = None
        if kind == "shrink" and rng.random() < 0.3:
            candidate = int(rng.integers(0, N_ACTIONS))
            prereq = candidate if candidate != action else None
        table.append({"kind": kind, "targets": targets, "factor": factor, "prereq": prereq})
    return table


_ACTION_TABLE = _build_action_table()


def _reference_sequence(length):
    shrink = [a for a in range(N_ACTIONS) if _ACTION_TABLE[a]["kind"] == "shrink"]
    shrink.sort(key=lambda a: -_ACTION_TABLE[a]["factor"])
    return shrink[:length]


def _program_counts(uri):
    """Deterministic initial category counts for a benchmark URI."""
    rng = np.random.default_rng(zlib.crc32(uri.encode("utf-8")))
    match = re.search(r"(\d+)$", uri)
    index = int(match.group(1)) if match else int(zlib.crc32(uri.encode()) % 997)
    family = (index // 4) % N_FAMILIES
    weights = np.ones(N_CATEGORIES)
    family_rng = np.random.default_rng(10_000 + family)
    boosted = family_rng.choice(N_CATEGORIES, size=12, replace=False)
    weights[boosted] *= 6.0
    scale = float(rng.uniform(10, 30))
    return rng.poisson(weights * scale).astype(np.int64)


def _apply(counts, applied, action):
    """Apply one action in place; return nothing."""
    spec = _ACTION_TABLE[action]
    if action in applied:
        return
    applied.add(action)
    if spec["kind"] == "shrink":
        prereq = spec["prereq"]
        strength = 1.0 if prereq is None or prereq in applied else 0.2
        delta = spec["factor"] * strength
        t = spec["targets"]
        counts[t] = np.floor(counts[t] * (1.0 - delta)).astype(np.int64)
    elif spec["kind"] == "expand":
        t = spec["targets"]
        counts[t] = np.ceil(counts[t] * (1.0 + abs(spec["factor"]))).astype(np.int64)


def _simulate(initial, sequence):
    counts = initial.copy()
    applied = set()
    for action in sequence:
        _apply(counts, applied, action)
    return int(counts.sum())


class _Discrete:
    def __init__(self, n, seed=0):
        self.n = n
        self._rng = np.random.default_rng(seed)

    def sample(self):
        return int(self._rng.integers(0, self.n))

    def seed(self, seed):
        self._rng = np.random.default_rng(seed)


class _ObservationView:
    def __init__(self, env):
        self._env = env

    def __getitem__(self, name):
        env = self._env
        if name == "Autophase":
            return env._autophase()
        if name == "IrInstructionCount":
            return int(env._counts.sum())
        if name == "IrInstructionCountO0":
            return int(env._initial.sum())
        if name == "IrInstructionCountOz":
            return _simulate(env._initial, _reference_sequence(_OZ_LEN))
        if name == "IrInstructionCountO3":
            return _simulate(env._initial, _reference_sequence(_O3_LEN))
        raise KeyError(name)


class MockLLVMEnv:
    """Mimics the subset of the CompilerGym LLVM API that this project uses."""

    def __init__(
        self,
        benchmark="benchmark://mock-v0/0",
        observation_space="Autophase",
        reward_space="IrInstructionCountOz",
        new_api=False,
        seed=0,
    ):
        self.observation_space_name = observation_space
        self.reward_space_name = reward_space
        self.new_api = new_api
        self.action_space = _Discrete(N_ACTIONS, seed=seed)
        self.observation = _ObservationView(self)
        self.benchmark = None
        self._initial = None
        self._counts = None
        self._applied = set()
        self._set_benchmark(benchmark)

    def _set_benchmark(self, uri):
        self.benchmark = str(uri)
        self._initial = _program_counts(self.benchmark)
        self._counts = self._initial.copy()
        self._applied = set()

    def _autophase(self):
        total = int(self._counts.sum())
        features = np.zeros(N_FEATURES, dtype=np.int64)
        features[:N_CATEGORIES] = self._counts
        features[51] = total
        features[52] = total // 8
        features[53] = total // 20
        features[55] = total // 10
        return features

    def reset(self, benchmark=None, **kwargs):
        if benchmark is not None:
            self._set_benchmark(benchmark)
        else:
            self._counts = self._initial.copy()
            self._applied = set()
        obs = self._autophase()
        if self.new_api:
            return obs, {}
        return obs

    def step(self, action):
        action = int(action)
        if not 0 <= action < N_ACTIONS:
            raise ValueError(f"Invalid action {action}")
        before = int(self._counts.sum())
        _apply(self._counts, self._applied, action)
        after = int(self._counts.sum())
        oz = self.observation["IrInstructionCountOz"]
        denominator = max(int(self._initial.sum()) - oz, 1)
        reward = (before - after) / denominator
        obs = self._autophase()
        info = {"action_had_no_effect": before == after}
        if self.new_api:
            return obs, reward, False, False, info
        return obs, reward, False, info

    def fork(self):
        return copy.deepcopy(self)

    def close(self):
        pass
