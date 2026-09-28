"""GRPO group-relative advantage.

For each "group" of G trajectories generated from the same prompt (= same
video + query in our streaming setup), the advantage of trajectory i is

    A_i = (r_i - mean(r_1, ..., r_G)) / (std(r_1, ..., r_G) + eps)

That is, advantages are *standardised within the group*. This eliminates
the need for a learnt critic (the group serves as the baseline).

Edge cases:
  - G == 1 → advantage undefined; we return 0 (no signal).
  - All rewards equal → std == 0; we return 0 for all (no signal either).
"""

from __future__ import annotations

import math
from typing import Sequence


def group_relative_advantage(
    rewards: Sequence[float],
    eps: float = 1e-8,
) -> list[float]:
    """Standardise rewards within a single group.

    Args:
        rewards: scalar reward per trajectory in the group. ``len`` is G.
        eps: numerical floor for std.

    Returns:
        List of advantages, same length as ``rewards``.
    """
    G = len(rewards)
    if G == 0:
        return []
    if G == 1:
        return [0.0]
    mu = sum(rewards) / G
    var = sum((r - mu)**2 for r in rewards) / G
    sigma = math.sqrt(var)
    if sigma < eps:
        return [0.0] * G
    return [(r - mu) / (sigma + eps) for r in rewards]


def batched_advantages(
    rewards_per_group: Sequence[Sequence[float]],
    eps: float = 1e-8,
) -> list[list[float]]:
    """Apply group-relative standardisation to multiple groups.

    Typical use: ``rewards_per_group[b]`` is the list of G rewards for the
    b-th prompt in a training-batch of B prompts.
    """
    return [group_relative_advantage(g, eps=eps) for g in rewards_per_group]
