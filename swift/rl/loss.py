"""GRPO loss computation.

Per-token form (TRL / Shao et al. 2024 style):

    ratio_t = exp(actor_logp_t - rollout_logp_t)
    pg_loss = - min(ratio_t * A, clip(ratio_t, 1-ε, 1+ε) * A)
    kl_t    = exp(ref_logp_t - actor_logp_t) - (ref_logp_t - actor_logp_t) - 1
              (Schulman k3 estimator, unbiased and >= 0)
    L_t     = pg_loss + β * kl_t

The trajectory loss is the per-token loss MEAN over the trajectory's
action positions (action tokens, NOT the entire input_ids sequence).
The batch loss averages over (B groups × G trajectories).

This module is GPU-agnostic; it operates on plain torch tensors so the
trainer can choose precision / device.
"""

from __future__ import annotations

import torch


def grpo_per_token_loss(
    actor_logps: torch.Tensor,  # (T,) actor log-probs of sampled tokens
    rollout_logps: torch.Tensor,  # (T,) logp used to SAMPLE the tokens
    ref_logps: torch.Tensor,  # (T,) reference policy logp
    advantage: float,  # scalar advantage for this trajectory
    clip_eps: float = 0.2,
    kl_coef: float = 0.04,
) -> dict[str, torch.Tensor]:
    """Compute per-token GRPO loss for one trajectory.

    All log-prob tensors must be shape (T,) on the same device. ``advantage``
    is a Python float (one per trajectory).

    Returns a dict with:
        'loss'    : scalar tensor, the mean loss over T action positions
        'pg_loss' : mean policy-gradient loss (no kl)
        'kl'      : mean k3 KL estimate
        'ratio'   : mean importance ratio (for monitoring)
        'clip_frac': fraction of tokens whose ratio was clipped
    """
    log_ratio = actor_logps - rollout_logps
    ratio = log_ratio.exp()
    clipped_ratio = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)

    # Unclipped PG (negative because we maximise reward via -loss).
    pg_unclipped = -ratio * advantage
    pg_clipped = -clipped_ratio * advantage
    # GRPO/PPO trick: take the worse-for-the-objective of the two.
    pg = torch.maximum(pg_unclipped, pg_clipped)

    # k3 KL estimator (Schulman 2020): exp(d) - d - 1 with d = ref - actor.
    # Mean is unbiased and >= 0 token-wise.
    d = ref_logps - actor_logps
    kl = d.exp() - d - 1.0
    kl = kl.clamp_min(0.0)  # numerical safety

    loss_per_token = pg + kl_coef * kl

    # ``ratio`` is monitoring-only.  A float32 CUDA reduction can report
    # 0.99999988 for an all-ones vector even though every token has an exact
    # zero log-ratio.  Reduce in float64 so the summary does not look like a
    # rollout/replay mismatch; the policy-gradient computation above remains
    # in its original dtype.
    ratio_mean = ratio.detach().mean(dtype=torch.float64)

    return {
        'loss': loss_per_token.mean(),
        'pg_loss': pg.mean(),
        'kl': kl.mean(),
        'ratio': ratio_mean,
        'clip_frac': ((ratio != clipped_ratio).float()).mean(),
    }


def grpo_batch_loss(
    trajectories: list[dict],
    advantages: list[float],
    clip_eps: float = 0.2,
    kl_coef: float = 0.04,
) -> dict[str, torch.Tensor]:
    """Aggregate GRPO loss over a batch of trajectories.

    Each ``trajectory`` dict must contain torch tensors:
        'actor_logps'   : (T,)
        'rollout_logps' : (T,)
        'ref_logps'     : (T,)

    ``advantages[i]`` is the scalar advantage for ``trajectories[i]``.

    Returns the same metric keys as ``grpo_per_token_loss`` but averaged
    over the batch.
    """
    if len(trajectories) != len(advantages):
        raise ValueError('trajectories and advantages must have same length')
    if not trajectories:
        raise ValueError('empty batch')

    out_keys = ('loss', 'pg_loss', 'kl', 'ratio', 'clip_frac')
    accum = {k: 0.0 for k in out_keys}
    accum['loss'] = None  # need grad-carrying tensor
    n = len(trajectories)

    losses = []
    for traj, adv in zip(trajectories, advantages):
        m = grpo_per_token_loss(
            actor_logps=traj['actor_logps'],
            rollout_logps=traj['rollout_logps'],
            ref_logps=traj['ref_logps'],
            advantage=adv,
            clip_eps=clip_eps,
            kl_coef=kl_coef,
        )
        losses.append(m['loss'])
        for k in ('pg_loss', 'kl', 'ratio', 'clip_frac'):
            accum[k] += float(m[k].item())

    batch_loss = torch.stack(losses).mean()
    return {
        'loss': batch_loss,
        'pg_loss': torch.tensor(accum['pg_loss'] / n),
        'kl': torch.tensor(accum['kl'] / n),
        'ratio': torch.tensor(accum['ratio'] / n),
        'clip_frac': torch.tensor(accum['clip_frac'] / n),
    }
