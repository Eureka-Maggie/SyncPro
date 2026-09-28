"""GRPO entry point backed by the production dual-branch streaming runner.

Per optimizer step:

1. Run query-bs prompts x G stochastic trajectories through
   ``StreamModel + ProactiveBatchRunner``.  Within every query, A/B decode
   synchronously from the same pre-P state and use the production cache/state
   merge.
2. Compute the hard-window F1 timing reward from branch A's visible actions.
3. Re-encode every sampled trajectory with the *training* template.  Require
   byte-for-byte token identity with inference physical history, then compute
   actor log-probabilities with the actual dual FlexAttention mask, MRoPE
   overlap, linear-recurrence fork/merge, and causal-conv correction.
4. Apply a group-relative policy-gradient update to only the four dual-LoRA
   matrix families. Dropout is disabled in rollout and actor/reference replay.
   When KL is enabled, the reference is the frozen SFT dual-LoRA snapshot and
   uses the same physical history, template, mask, and replay implementation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import numpy as np
import os
import random
import signal
import shutil
import sys
import threading
import time
import torch
import torch.distributed as dist
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from PIL import Image
from torch.optim import AdamW
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from typing import Optional

from examples.rl.dual_grpo_actor_rebalance import (
    ACTOR_RANK_REBALANCE_CHOICES,
    ACTOR_RANK_REBALANCE_NONE,
    actor_rebalance_enabled,
    actor_rows_for_rank,
    compact_actor_groups,
    plan_gathered_actor_group_costs,
    plan_gathered_actor_groups,
)
from examples.rl.proactive_timing_data import behavior_metadata, file_sha256
from examples.rl.proactive_timing_data import frame_times as canonical_frame_times
from examples.rl.proactive_timing_data import load_verified_split, segment_for_training, timing_profile
from examples.rl.dual_grpo_schedule import (BALANCED_FRAMES, FIXED_EPOCH,
                                            BEHAVIOR_SOURCE_MIXED_DATA, FULL_POSITIVE_DATA, LENGTH_BUCKETED,
                                            deterministic_batch_indices,
                                            estimated_stream_work, rank_prompt_slot_shards, shard_loads,
                                            timing_epoch_indices)
from swift import get_template
from swift.rl import (DUAL_POLICY_REPLAY_CONTRACT_VERSION, HARD_WINDOW_F1, TIMING_REWARD_MODES,
                      DualBranchSampler, DualRewardWeights, aggregate_dual_reward,
                      group_relative_advantage, grpo_per_token_loss)
from swift.template.step_mask.dual_lora import clear_ctx as clear_dual_ctx
from swift.template.step_mask.meta import build_step_mask_meta
from swift.template.step_mask.patch import (actor_fused_lm_head_input_backward_enabled,
                                             actor_fused_lm_head_projection,
                                             actor_lm_head_execution_plan,
                                             clear_step_mask_context,
                                             defer_actor_lm_head_projection,
                                             use_actor_precomputed_visual_features)
from swift.template.step_mask.position_overlap import apply_overlap_to_position_ids_tensor
from swift.utils.torch_utils import to_device

TRAINABLE_PARTS = (
    'lora_A_up',
    'lora_A_down',
    'lora_B_up',
    'lora_B_down',
)
INFERENCE_ALIGNED_FLEX_KERNEL_OPTIONS = (
    'BLOCK_M=32,BLOCK_N=32,num_stages=1,num_warps=4'
)
ACTOR_LM_HEAD_CHECKPOINT_DECISIONS = 32
SKELETON_DUAL_TARGET = '<silent><|im_end|>\n<silent>'
ACTOR_REPLAY_CONTRACT_VERSION = DUAL_POLICY_REPLAY_CONTRACT_VERSION
ACTOR_BACKWARD_NUMERICS_VERSION = 2
POLICY_DECISION_ALL = 'all'
POLICY_DECISION_SCOPES = (POLICY_DECISION_ALL, )

# Resume must not silently change policy, actor numerics, reward, or gradient.
# Operational settings such as max_steps and checkpoint cadence may change.
RESUME_CONTRACT_KEYS = (
    'actor_linear_mode',
    'streaming_cache',
    'rollout_query_batch_mode',
    'batch_schedule',
    'timing_data_variant',
    'timing_epoch_steps',
    'rank_sharding',
    'window',
    'actor_replay_contract_version',
    'actor_backward_numerics_version',
    'query_batch_size',
    'group_size',
    'rollout_trajectory_microbatch_size',
    'reuse_frozen_visual_features',
    'defer_actor_lm_head',
    'actor_fused_frozen_linear_backward',
    'actor_fused_trainable_linear_input_backward',
    'actor_segmented_rmsnorm_backward',
    'actor_fused_gated_rmsnorm_backward',
    'actor_fused_lm_head_input_backward',
    'actor_trajectory_microbatch_size',
    'actor_packed_token_budget',
    'actor_rank_rebalance',
    'actor_rank_rebalance_min_gain',
    'max_frames',
    'max_tokens',
    'max_kv_len',
    'temperature',
    'top_p',
    'learning_rate',
    'weight_decay',
    'clip_eps',
    'kl_coef',
    'sft_reference_model',
    'max_grad_norm',
    'gradient_checkpointing',
    'gradient_checkpointing_policy',
    'actor_flex_kernel_options',
    'max_logp_drift',
    'max_mean_logp_drift',
    'validate_actor_isolation',
    'max_actor_isolation_logp_drift',
    'distributed_exact_check_every',
    'system_prompt',
    'seed',
    'reward_format',
    'reward_timing',
    'reward_sigma',
    'timing_reward_mode',
    'policy_decision_scope',
    'deterministic_algorithms',
    'max_sequence_length',
)


@dataclass
class DualCase:
    row_index: int
    query: str
    image_paths: list[str]
    gt: dict
    meta: dict = field(default_factory=dict)


@dataclass
class DualDecision:
    route: str
    frame_index: int
    token_index: int
    token_id: int
    position: int
    rollout_logp: float
    history: list[int]


@dataclass
class DualTrajectory:
    prompt_slot: int
    group_index: int
    case: DualCase
    messages: list[dict]
    frames: list[dict]
    physical_ids: np.ndarray
    seed: int
    images: list = field(default_factory=list)
    image_embeds: list[torch.Tensor] = field(default_factory=list)
    image_grids: list[tuple[int, int, int]] = field(default_factory=list)
    reward: Optional[dict] = None


class FP32MasterAdamW:
    """AdamW with FP32 master copies for BF16 dual-LoRA parameters.

    The loaded policy stays BF16 for rollout and actor replay, but an LR such
    as 1e-6 is often below one BF16 ULP for trained LoRA weights.  Updating
    BF16 parameters directly can therefore be a silent no-op.  Master copies
    accumulate every small update in FP32 and are quantized back to the model
    after each step, matching standard mixed-precision optimizer semantics.
    """

    def __init__(self, model_parameters, **adamw_kwargs):
        self.model_parameters = list(model_parameters)
        self.master_parameters = [
            torch.nn.Parameter(
                parameter.detach().to(dtype=torch.float32).clone(),
                requires_grad=True,
            ) for parameter in self.model_parameters
        ]
        self.inner = AdamW(self.master_parameters, **adamw_kwargs)
        self.last_model_update_norm = 0.0

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self.model_parameters:
            parameter.grad = None if set_to_none else torch.zeros_like(parameter)
        self.inner.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, master_gradients=None):
        if (master_gradients is not None and len(master_gradients) != len(self.master_parameters)):
            raise ValueError('master gradient count does not match optimizer parameters')
        for index, (model_parameter, master_parameter) in enumerate(zip(self.model_parameters, self.master_parameters)):
            source_gradient = (model_parameter.grad if master_gradients is None else master_gradients[index])
            if (source_gradient is not None and tuple(source_gradient.shape) != tuple(master_parameter.shape)):
                raise ValueError(f'master gradient shape mismatch at {index}: '
                                 f'{tuple(source_gradient.shape)} != '
                                 f'{tuple(master_parameter.shape)}')
            master_parameter.grad = (None if source_gradient is None else source_gradient.detach().to(
                device=master_parameter.device, dtype=torch.float32))
        result = self.inner.step()
        squared_update = torch.zeros((), device=self.master_parameters[0].device, dtype=torch.float32)
        for model_parameter, master_parameter in zip(self.model_parameters, self.master_parameters):
            updated = master_parameter.to(model_parameter.dtype)
            difference = updated.float() - model_parameter.float()
            squared_update.add_(difference.square().sum())
            model_parameter.copy_(updated)
        self.last_model_update_norm = float(squared_update.sqrt().item())
        return result

    def state_dict(self) -> dict:
        return {
            'inner_optimizer': self.inner.state_dict(),
            'master_weights': [parameter.detach().cpu() for parameter in self.master_parameters],
        }

    @torch.no_grad()
    def load_state_dict(self, state: dict) -> None:
        weights = state.get('master_weights')
        if not isinstance(weights, list):
            raise ValueError('optimizer state lacks FP32 master_weights')
        if len(weights) != len(self.master_parameters):
            raise ValueError('FP32 master parameter count mismatch: '
                             f'{len(weights)} != {len(self.master_parameters)}')
        for index, (source, master,
                    model_parameter) in enumerate(zip(weights, self.master_parameters, self.model_parameters)):
            if tuple(source.shape) != tuple(master.shape):
                raise ValueError(f'FP32 master shape mismatch at {index}: '
                                 f'{tuple(source.shape)} != {tuple(master.shape)}')
            master.copy_(source.to(device=master.device, dtype=torch.float32))
            model_parameter.copy_(master.to(model_parameter.dtype))
        self.inner.load_state_dict(state['inner_optimizer'])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--model', required=True, help='Full dual-branch checkpoint directory.')
    parser.add_argument(
        '--resume_from_checkpoint', default=None, help='RL checkpoint to load model+optimizer/RNG from.')
    parser.add_argument('--timing_train_jsonl', required=True, help='Canonical proactive_timing_v1 training split.')
    parser.add_argument('--timing_val_jsonl', required=True)
    parser.add_argument('--timing_manifest', required=True)
    parser.add_argument(
        '--timing_data_variant',
        choices=(FULL_POSITIVE_DATA, ),
        default=FULL_POSITIVE_DATA,
        help='Canonical timing-data epoch construction. Non-full variants '
        'materialize an auditable fixed-length epoch.',
    )
    parser.add_argument(
        '--timing_epoch_steps',
        type=int,
        default=0,
        help='Optimizer steps in one materialized timing epoch; zero keeps '
        'the complete positive clip pool.',
    )
    parser.add_argument(
        '--video_root',
        action='append',
        default=[],
        metavar='SOURCE=PATH',
        help='Override a manifest video root; repeat for multiple roots.')
    parser.add_argument('--cases', type=int, nargs='+', default=None, help='Optional zero-based rows/qids.')
    parser.add_argument('--system_prompt', default=None, help='Override only when it matches checkpoint contract.')
    parser.add_argument('--output_dir', required=True)

    parser.add_argument('--max_steps', type=int, default=None, help='Updates; canonical data defaults to one epoch.')
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--query_batch_size', type=int, default=1, help='Number of distinct query groups per step.')
    parser.add_argument('--group_size', type=int, default=4, help='Stochastic trajectories per query (G).')
    parser.add_argument(
        '--rollout_trajectory_microbatch_size',
        '--rollout_query_batch_size',
        dest='rollout_trajectory_microbatch_size',
        type=int,
        default=0,
        help='Trajectory rows fused by dual infer at once; 0 means one '
        'query_bs-sized generation wave. The legacy '
        '--rollout_query_batch_size spelling is accepted but does not '
        'mean query_bs.',
    )
    parser.add_argument(
        '--rollout_query_batch_mode',
        choices=('strict', 'throughput'),
        default='throughput',
        help='Different-query rollout scheduler. throughput fuses query '
        'decode rows; strict keeps query_bs scheduling but executes '
        'each query independently as the numerical reference.',
    )
    parser.add_argument(
        '--reuse_frozen_visual_features',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Reuse each query/frame frozen visual-tower output across G '
        'rollouts and actor replay. This changes no language cache or branch '
        'state. The generic entry keeps this opt-in; the production entry '
        'enables the CUDA-validated path.',
    )
    parser.add_argument(
        '--defer_actor_lm_head',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Checkpoint the exact actor LM-head/policy tail from selected '
        'hidden states so vocabulary-sized logits are not retained. The '
        'generic entry keeps this opt-in for A/B gates; production enables it.',
    )
    parser.add_argument(
        '--actor_fused_frozen_linear_backward',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Keep production-shaped actor forwards while fusing the '
        'backward-only input derivatives of frozen shape-sensitive Linear '
        'modules. Trainable Linear modules remain on exact segmented '
        'backward. The generic entry keeps this opt-in; production enables it.',
    )
    parser.add_argument(
        '--actor_fused_trainable_linear_input_backward',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Fuse only the input derivative of shape-sensitive trainable '
        'LoRA Linear modules while retaining the original segmented, reverse '
        'weight-gradient reduction. The generic entry keeps this opt-in; '
        'production enables the CUDA-validated path.',
    )
    parser.add_argument(
        '--actor_segmented_rmsnorm_backward',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Use a manual frozen RMSNorm input derivative while preserving '
        'the exact production forward and backward execution groups. The '
        'generic entry keeps this opt-in; production enables the '
        'CUDA-validated path.',
    )
    parser.add_argument(
        '--actor_fused_gated_rmsnorm_backward',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Keep each FLA gated-RMSNorm forward and its FP32 reciprocal '
        'RMS unchanged while fusing the frozen-norm input/gate derivative. '
        'The generic entry keeps this opt-in; production enables the '
        'CUDA-validated path.',
    )
    parser.add_argument(
        '--actor_fused_lm_head_input_backward',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Keep every production-shaped frozen LM-head forward, but fuse '
        'its row-independent input derivative in bitwise-validated blocks of '
        'at most eight rows. Requires --defer_actor_lm_head. The generic '
        'entry keeps this opt-in; production enables it.',
    )
    parser.add_argument(
        '--batch_schedule',
        choices=(FIXED_EPOCH, ),
        default=FIXED_EPOCH,
        help='Fixed global-query order from the released balanced schedule.',
    )
    parser.add_argument(
        '--rank_sharding',
        choices=(BALANCED_FRAMES, ),
        default=BALANCED_FRAMES,
        help='Distributed query assignment. balanced_frames_v1 keeps global '
        'prompt slots/seeds fixed while balancing a metadata-only stream-work '
        'estimate across ranks.',
    )
    parser.add_argument(
        '--actor_trajectory_microbatch_size',
        '--actor_query_batch_size',
        dest='actor_trajectory_microbatch_size',
        type=int,
        default=0,
        help='Independent trajectory rows packed into one actor forward; '
        '0 means one trajectory. This is deliberately independent of '
        'query_bs. The legacy --actor_query_batch_size spelling is '
        'accepted for compatibility.',
    )
    parser.add_argument(
        '--actor_packed_token_budget',
        type=int,
        default=0,
        help='Maximum sum of physical sequence tokens in one packed actor '
        'microbatch. Zero disables the token budget. A trajectory longer '
        'than the budget still runs alone.',
    )
    parser.add_argument(
        '--actor_rank_rebalance',
        choices=ACTOR_RANK_REBALANCE_CHOICES,
        default=ACTOR_RANK_REBALANCE_NONE,
        help='Optional post-rollout complete-G actor redistribution. none '
        'preserves rollout rank ownership; post_rollout_groups_v1 balances '
        'sampled decision counts without changing rollout/reward/advantages.',
    )
    parser.add_argument(
        '--actor_rank_rebalance_min_gain',
        type=float,
        default=0.0,
        help='Minimum relative decrease in the predicted maximum per-rank '
        'sampled-decision load required before moving complete query groups. '
        'Zero still skips plans with no improvement.',
    )
    parser.add_argument('--max_frames', type=int, default=14)
    parser.add_argument('--max_tokens', type=int, default=64, help='Per-branch token cap, including structural tokens.')
    parser.add_argument('--max_kv_len', type=int, default=8192)
    parser.add_argument('--max_sequence_length', type=int, default=32768)
    parser.add_argument(
        '--streaming_cache',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Use the same preallocated hybrid cache as '
        'production eval (default: true).')
    parser.add_argument(
        '--window',
        type=int,
        default=1,
        help='Step-mask visual window; must match '
        'STEP_MASK_WINDOW (SFT contract default: 1).')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument(
        '--warmup',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Opt in to synthetic 1/128/1280-token full-model '
        'warmup; normal runs compile real shapes on demand.')
    parser.add_argument(
        '--actor_linear_mode',
        choices=('training_chunk', 'inference_aligned'),
        default='inference_aligned',
        help='Linear-attention actor replay: inference_aligned uses an '
        'autograd-safe recurrent rule at production decode boundaries; '
        'training_chunk retains the SFT chunk kernel for diagnostics.',
    )
    parser.add_argument(
        '--distributed_backend',
        choices=('nccl', ),
        default='nccl',
        help='Backend used when launched by torchrun (WORLD_SIZE>1).')
    parser.add_argument(
        '--distributed_timeout_seconds',
        type=int,
        default=1800,
        help='Fail-closed collective timeout for multi-GPU training.')
    parser.add_argument(
        '--distributed_exact_check_every',
        type=int,
        default=1,
        help='Hash every trainable BF16 tensor every N updates and, when '
        'distributed, require exact equality across ranks; 0 disables the '
        'post-update exactness gate.')

    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--learning_rate', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--clip_eps', type=float, default=0.2)
    parser.add_argument('--kl_coef', type=float, default=0.0)
    parser.add_argument(
        '--max_logp_drift',
        type=float,
        default=0.0,
        help='Fail if rollout vs training-layout actor logp '
        'drifts more than this before an update '
        '(default: require exact same-weight replay).')
    parser.add_argument(
        '--max_mean_logp_drift',
        type=float,
        default=0.0,
        help='Fail if a trajectory has systematic mean '
        'rollout/replay logp drift above this value.')
    parser.add_argument(
        '--validate_actor_isolation',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Run no-grad one-trajectory actor references before every '
        'multi-trajectory actor forward and compare selected logps. '
        'Intended for CUDA correctness gates.',
    )
    parser.add_argument(
        '--max_actor_isolation_logp_drift',
        type=float,
        default=0.0,
        help='Maximum packed-vs-one-trajectory actor logp drift when '
        '--validate_actor_isolation is active (default: exact).',
    )
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--gradient_checkpointing', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        '--gradient_checkpointing_policy',
        choices=(
            'all',
            'linear_only',
            'linear_alternating',
            'linear_sparse',
        ),
        default='all',
        help='Which decoder layers use outer activation checkpointing. '
        'The linear policies checkpoint all, every second, or every third '
        'linear-attention layer respectively; '
        'full-attention segments retain their own exact inner checkpoint. '
        'linear_alternating/sparse trade substantial VRAM for speed and are '
        'not suitable defaults for long streams.',
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--deterministic_algorithms',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Require deterministic CUDA algorithms and fixed cuBLAS '
        'workspace (default: true; opt out only for speed experiments).')
    parser.add_argument(
        '--debug_micro_gradient_digests',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Hash every actor microbatch gradient before accumulation. '
        'This is an expensive CUDA correctness diagnostic and does not '
        'change the optimizer input.')
    parser.add_argument(
        '--profile_actor_phases',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Record CUDA-event time for actor forward, autograd backward, '
        'and deterministic FP64 gradient capture. Intended for speed '
        'diagnostics; disabled in production by default.',
    )
    parser.add_argument(
        '--log_rollout_policy_trace',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Log compact per-decision rollout metadata/logp for cross-mode '
        'CUDA drift gates; disabled for normal training logs.',
    )

    parser.add_argument('--reward_format', type=float, default=0.0)
    parser.add_argument('--reward_timing', type=float, default=1.0)
    parser.add_argument('--reward_sigma', type=float, default=2.0)
    parser.add_argument('--timing_reward_mode', choices=TIMING_REWARD_MODES, default=HARD_WINDOW_F1)
    parser.add_argument(
        '--policy_decision_scope',
        choices=POLICY_DECISION_SCOPES,
        default=POLICY_DECISION_ALL,
        help='Policy rows receiving trajectory advantage. a_action keeps the '
        'full dual rollout/replay contract but applies timing GRPO only to '
        "branch A's first <speak>/<silent> decision per frame.",
    )

    parser.add_argument('--eval_every', type=int, default=0, help='Validate every N updates; 0 means final only.')
    parser.add_argument('--eval_at_start', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--eval_query_batch_size', type=int, default=1)
    parser.add_argument('--eval_max_tokens', type=int, default=256)
    parser.add_argument('--eval_max_cases', type=int, default=0, help='0 evaluates the full validation split.')

    parser.add_argument('--report_to_swanlab', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--swanlab_project', default='mss_dual_grpo')
    parser.add_argument('--swanlab_exp_name', default=None)
    parser.add_argument('--swanlab_workspace', default=None)
    parser.add_argument('--swanlab_mode', choices=('cloud', 'local', 'offline', 'disabled'), default='cloud')
    parser.add_argument(
        '--swanlab_init_timeout_seconds',
        type=int,
        default=180,
        help='Rank-zero wall-clock timeout for SwanLab initialization. Other '
        'ranks wait at the same startup rendezvous instead of entering rollout.',
    )
    parser.add_argument(
        '--swanlab_run_id',
        default=None,
        help='Existing SwanLab run ID for migrating a pre-manifest checkpoint. '
        'Normal AUTO_RESUME reads the persisted ID from the output directory.',
    )

    parser.add_argument('--save_every', type=int, default=20, help='0 disables periodic full checkpoints.')
    parser.add_argument('--save_final', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--max_shard_size', default='5GB')
    parser.add_argument('--log_path', default=None)
    parser.add_argument(
        '--dry_run_rollout', action='store_true', help='Run rollout+alignment+logp gate without update.')
    args = parser.parse_args(argv)

    # The public release intentionally fixes all removed ablation objectives
    # to zero. These attributes remain because the optimized actor path shares
    # bookkeeping code with the original trainer.
    args.silent_rate_loss_weight = 0.0
    args.action_outside_loss_weight = 0.0
    args.action_coverage_loss_weight = 0.0
    args.action_count_loss_weight = 0.0

    finite_fields = (
        'learning_rate',
        'weight_decay',
        'clip_eps',
        'kl_coef',
        'max_logp_drift',
        'max_mean_logp_drift',
        'max_actor_isolation_logp_drift',
        'actor_rank_rebalance_min_gain',
        'max_grad_norm',
        'temperature',
        'top_p',
        'reward_format',
        'reward_timing',
        'reward_sigma',
        'silent_rate_loss_weight',
        'action_outside_loss_weight',
        'action_coverage_loss_weight',
        'action_count_loss_weight',
    )
    nonfinite = [field for field in finite_fields if not np.isfinite(getattr(args, field))]
    if nonfinite:
        parser.error(f'numerical arguments must be finite: {nonfinite}')

    if args.distributed_timeout_seconds < 1:
        parser.error('distributed_timeout_seconds must be >=1')
    if args.swanlab_init_timeout_seconds < 1:
        parser.error('swanlab_init_timeout_seconds must be >=1')
    if args.distributed_exact_check_every < 0:
        parser.error('distributed_exact_check_every must be >=0')
    if (args.max_steps is not None and args.max_steps < 1) or args.query_batch_size < 1:
        parser.error('max_steps (when set) and query_batch_size must be >= 1')
    if args.num_train_epochs < 1:
        parser.error('num_train_epochs must be >= 1')
    if (args.rollout_trajectory_microbatch_size < 0
            or args.actor_trajectory_microbatch_size < 0
            or args.actor_packed_token_budget < 0):
        parser.error('rollout/actor trajectory microbatch sizes and token budget must be >= 0')
    if not 0 <= args.actor_rank_rebalance_min_gain < 1:
        parser.error('actor_rank_rebalance_min_gain must be in [0, 1)')
    if not args.gradient_checkpointing and args.gradient_checkpointing_policy != 'all':
        parser.error('gradient_checkpointing_policy requires gradient_checkpointing')
    if args.group_size < 2:
        parser.error('group_size must be >= 2 for group-relative advantage')
    if args.max_frames < 1 or args.max_tokens < 4:
        parser.error('max_frames must be >=1 and max_tokens must be >=4')
    if args.max_kv_len < 1 or args.max_sequence_length < 1:
        parser.error('max_kv_len and max_sequence_length must be >=1')
    if args.learning_rate <= 0:
        parser.error('learning_rate must be >0')
    if args.weight_decay < 0:
        parser.error('weight_decay must be >=0')
    if not 0 < args.clip_eps < 1:
        parser.error('clip_eps must be in (0, 1)')
    if args.max_grad_norm <= 0:
        parser.error('max_grad_norm must be >0')
    if args.reward_sigma <= 0:
        parser.error('reward_sigma must be >0')
    if (args.reward_format != 0.0 or args.reward_timing != 1.0
            or args.timing_reward_mode != HARD_WINDOW_F1):
        parser.error('the public recipe requires reward_format=0, reward_timing=1, '
                     'and timing_reward_mode=hard_window_f1_v1')
    if args.timing_epoch_steps < 0:
        parser.error('timing_epoch_steps must be non-negative')
    if args.timing_data_variant == FULL_POSITIVE_DATA:
        if args.timing_epoch_steps:
            parser.error('full_positive_v1 requires timing_epoch_steps=0')
    else:
        if args.timing_train_jsonl is None:
            parser.error('materialized timing data variants require --timing_train_jsonl')
        if args.timing_epoch_steps < 1:
            parser.error('materialized timing data variants require timing_epoch_steps>=1')
        if args.batch_schedule != FIXED_EPOCH:
            parser.error('materialized timing data variants require batch_schedule=fixed_epoch_v1')
    if args.policy_decision_scope != POLICY_DECISION_ALL:
        parser.error('the public balanced hard-F1 recipe requires policy_decision_scope=all')
    if args.save_every < 0:
        parser.error('save_every must be >=0')
    if args.top_p <= 0 or args.top_p > 1:
        parser.error('top_p must be in (0, 1]')
    if args.temperature <= 0:
        parser.error('temperature must be > 0')
    if args.window < 1:
        parser.error('window must be >=1 to activate the training mask')
    if args.max_logp_drift < 0 or args.max_mean_logp_drift < 0:
        parser.error('logp drift thresholds must be non-negative')
    if args.max_mean_logp_drift > args.max_logp_drift:
        parser.error('mean logp drift limit must not exceed the max limit')
    clip_safe_log_drift = min(np.log1p(args.clip_eps), -np.log1p(-args.clip_eps))
    if args.max_logp_drift >= clip_safe_log_drift:
        parser.error('max_logp_drift must keep the same-weight ratio strictly inside '
                     'the configured GRPO clip interval')
    if args.max_actor_isolation_logp_drift < 0:
        parser.error('max_actor_isolation_logp_drift must be non-negative')
    if args.kl_coef < 0:
        parser.error('kl_coef must be non-negative')
    if args.kl_coef > 0 and args.top_p < 1.0:
        parser.error(
            'KL requires --top_p 1: independently truncated actor/reference '
            'policies can have different support and infinite sampled-token KL'
        )
    if args.cases is not None:
        if len(set(args.cases)) != len(args.cases) or min(args.cases) < 0:
            parser.error('--cases must be unique non-negative row indices')
        if len(args.cases) < args.query_batch_size:
            parser.error('--cases must contain at least query_batch_size distinct queries; '
                         f'got {len(args.cases)} cases for query_bs={args.query_batch_size}')
    if args.eval_every < 0 or args.eval_query_batch_size < 1 or args.eval_max_tokens < 1 or args.eval_max_cases < 0:
        parser.error('invalid validation cadence/batch/token/case setting')
    args.actor_replay_contract_version = ACTOR_REPLAY_CONTRACT_VERSION
    args.actor_backward_numerics_version = ACTOR_BACKWARD_NUMERICS_VERSION
    args.actor_flex_kernel_options = INFERENCE_ALIGNED_FLEX_KERNEL_OPTIONS
    args.sft_reference_model = (
        str(Path(args.model).expanduser().resolve()) if args.kl_coef > 0 else None
    )
    return args


def _init_distributed(args) -> None:
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if world_size < 1 or not 0 <= rank < world_size:
        raise RuntimeError(f'invalid distributed identity rank={rank} world_size={world_size}')
    if world_size > 1:
        if not dist.is_available():
            raise RuntimeError('torch.distributed is unavailable')
        if args.query_batch_size % world_size != 0:
            raise RuntimeError('global query_bs must be divisible by WORLD_SIZE: '
                               f'{args.query_batch_size} % {world_size} != 0')
        if local_rank < 0 or local_rank >= torch.cuda.device_count():
            raise RuntimeError(f'LOCAL_RANK={local_rank} is outside visible CUDA devices')
        torch.cuda.set_device(local_rank)
        args.device = f'cuda:{local_rank}'
        dist.init_process_group(
            backend=args.distributed_backend,
            device_id=torch.device(args.device),
            init_method='env://',
            timeout=timedelta(seconds=args.distributed_timeout_seconds),
        )
        if dist.get_rank() != rank or dist.get_world_size() != world_size:
            raise RuntimeError('process-group identity differs from torchrun env')
    elif args.device == 'cuda':
        args.device = 'cuda:0'
    args.rank = rank
    args.local_rank = local_rank
    args.world_size = world_size
    args.local_query_batch_size = args.query_batch_size // world_size
    args.distributed = world_size > 1


def _rank_print(args, message: str, *, all_ranks: bool = False) -> None:
    if all_ranks or args.rank == 0:
        print(f'[rank {args.rank}/{args.world_size}] {message}', flush=True)


def _rank0_call_or_raise(args, description: str, function):
    """Run rank-zero I/O and propagate its failure before any rank continues."""
    result = None
    local_exception = None
    error = None
    if args.rank == 0:
        try:
            result = function()
        except Exception as exc:
            local_exception = exc
            error = {
                'type': type(exc).__name__,
                'message': str(exc),
            }
    if args.distributed:
        payload = [error]
        dist.broadcast_object_list(payload, src=0)
        error = payload[0]
    if error is not None:
        if local_exception is not None:
            raise local_exception
        raise RuntimeError(f'rank 0 failed to {description}: ' f'{error["type"]}: {error["message"]}')
    return result


def _resolve_triton_autotune_lock(args, cache_dir: Path) -> Path:
    configured_lock = os.environ.get('MSS_TRITON_AUTOTUNE_LOCK')
    if configured_lock is not None:
        configured_lock = configured_lock.strip()
        if not configured_lock:
            raise RuntimeError('MSS_TRITON_AUTOTUNE_LOCK must not be empty when set')
        return Path(configured_lock).expanduser().resolve()

    local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', args.world_size))
    if args.world_size > local_world_size:
        raise RuntimeError('multi-node deterministic RL requires an explicit ' 'cross-node MSS_TRITON_AUTOTUNE_LOCK')

    # The checkpoint/output mount is often FUSE-backed and rejects
    # fcntl.flock with EOPNOTSUPP. A stable hash lets every torchrun rank on
    # this node share one local lock without coupling unrelated jobs.
    lock_key = hashlib.sha256(str(cache_dir).encode('utf-8')).hexdigest()[:20]
    return Path(f'/tmp/mss_triton_autotune_{lock_key}.lock')


def _triton_autotune_key_lock_path(base_lock_path: Path, function_cache_key, tuning_key, configs) -> Path:
    identity = '\x1f'.join([str(function_cache_key), str(tuning_key)] + [str(config) for config in configs])
    digest = hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]
    return Path(f'{base_lock_path}.{digest}')


def _probe_advisory_lock(lock_path: Path) -> None:
    """Fail before model load when the selected lock filesystem lacks flock."""
    import fcntl

    try:
        with lock_path.open('a+b') as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    except OSError as exc:
        raise RuntimeError('Triton autotune lock does not support fcntl.flock: '
                           f'{lock_path}. Use a node-local /tmp path for one node, or an '
                           'explicit cross-node lock path on a flock-capable shared '
                           'filesystem for multi-node training.') from exc


def _require_cacheable_distributed_autotune(args, tuning_key, configs) -> None:
    """Reject Triton keys that cannot publish one common config to all ranks."""
    if args.world_size <= 1:
        return
    has_pre_hook = any(getattr(config, 'pre_hook', None) is not None for config in configs)
    if not tuning_key or has_pre_hook:
        raise RuntimeError('deterministic distributed RL reached a Triton autotune key '
                           'that cannot be persisted (empty key or config pre_hook); '
                           'refusing rank-local benchmark selections')


def _install_serialized_triton_autotune(args) -> dict:
    """Make cold Triton config selection common to every distributed rank."""
    if not args.deterministic_algorithms:
        return {
            'enabled': False,
            'cache_dir': None,
            'lock_path': None,
            'lock_scope': None,
        }
    if os.environ.get('FLA_CACHE_RESULTS', '1') != '1':
        raise RuntimeError('deterministic RL requires FLA_CACHE_RESULTS=1')

    output_dir = Path(args.output_dir).expanduser().resolve()
    cache_dir = Path(os.environ.get('MSS_TRITON_CACHE_DIR', output_dir / '.triton_cache')).expanduser().resolve()
    lock_path = _resolve_triton_autotune_lock(args, cache_dir)

    def prepare_autotune_paths():
        cache_dir.mkdir(parents=True, exist_ok=True)
        lock_path.parent.mkdir(parents=True, exist_ok=True)

    _rank0_call_or_raise(args, 'prepare Triton autotune paths', prepare_autotune_paths)
    os.environ['FLA_CACHE_RESULTS'] = '1'
    os.environ['TRITON_CACHE_DIR'] = str(cache_dir)

    _probe_advisory_lock(lock_path)
    import fcntl
    from triton.runtime.autotuner import Autotuner
    from triton.runtime.jit import JITFunction

    installed = getattr(Autotuner, '_mss_serialized_cache', None)
    requested = (str(cache_dir), str(lock_path), 'per_key_v1')
    if installed is not None:
        if installed != requested:
            raise RuntimeError('Triton autotune serialization was already installed with '
                               f'{installed}, requested={requested}')
    else:
        original = Autotuner.check_disk_cache

        def serialized_check_disk_cache(tuner, tuning_key, configs, bench_fn):
            # Triton's persistent cache has no inter-process exclusion.
            _require_cacheable_distributed_autotune(args, tuning_key, configs)
            # Serialize equal persistent-cache keys so exactly one rank
            # benchmarks and publishes their common config. Independent keys
            # can safely autotune on different GPUs in parallel.
            fn = tuner.fn
            while not isinstance(fn, JITFunction):
                fn = fn.fn
            key_lock_path = _triton_autotune_key_lock_path(lock_path, fn.cache_key, tuning_key, configs)
            with key_lock_path.open('a+b') as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                try:
                    return original(tuner, tuning_key, configs, bench_fn)
                finally:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

        Autotuner.check_disk_cache = serialized_check_disk_cache
        Autotuner._mss_serialized_cache = requested
    return {
        'enabled': True,
        'cache_dir': str(cache_dir),
        'lock_path': str(lock_path),
        'lock_scope': 'per_persistent_key',
    }


def _timing_video_roots(manifest: dict, overrides: list[str]) -> dict[str, list[Path]]:
    roots = {
        source: [Path(value).expanduser().resolve() for value in values]
        for source, values in manifest.get('video_roots', {}).items()
    }
    overridden: dict[str, list[Path]] = {}
    for value in overrides:
        source, separator, path = value.partition('=')
        if not separator or not source or not path:
            raise ValueError(f'--video_root must be SOURCE=PATH, got {value!r}')
        overridden.setdefault(source, []).append(Path(path).expanduser().resolve())
    roots.update(overridden)
    return roots


def _resolve_timing_video(sample: dict, roots: dict[str, list[Path]]) -> str:
    root_key = sample['video']['root_key']
    candidates = [root / sample['video']['path'] for root in roots.get(root_key, [])]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(f'{sample["sample_id"]} video was not found under roots '
                            f'{[str(path) for path in roots.get(root_key, [])]}: '
                            f'{sample["video"]["path"]}')


def load_timing_cases(path: str,
                      manifest_path: str,
                      *,
                      max_frames: Optional[int],
                      video_root_overrides: list[str],
                      indices: Optional[list[int]] = None) -> tuple[list[DualCase], dict]:
    """Load canonical timing data; bounded ``max_frames`` enables training clips."""
    samples, manifest = load_verified_split(path, manifest_path)
    roots = _timing_video_roots(manifest, video_root_overrides)
    selected = range(len(samples)) if indices is None else indices
    invalid = [index for index in selected if index < 0 or index >= len(samples)]
    if invalid:
        raise IndexError(f'timing JSONL rows not found: {invalid}')
    expanded = []
    resolved_videos = {}
    for row_index in selected:
        sample = samples[row_index]
        clips = segment_for_training(sample, max_frames) if max_frames is not None else [sample]
        video_key = (sample['video']['root_key'], sample['video']['path'])
        if video_key not in resolved_videos:
            resolved_videos[video_key] = _resolve_timing_video(sample, roots)
        video_path = resolved_videos[video_key]
        for clip in clips:
            expanded.append((row_index, clip, video_path))

    cases = []
    for case_index, (row_index, sample, video_path) in enumerate(expanded):
        times = canonical_frame_times(sample)
        behavior = behavior_metadata(sample)
        cases.append(
            DualCase(
                row_index=case_index,
                query=sample['query'],
                image_paths=[],
                gt={
                    'gt_windows': [dict(window) for window in sample['gt_windows']],
                    'frame_targets': [],
                },
                meta={
                    'data_source': sample['source'],
                    'source': sample['source'],
                    'task': sample['task'],
                    'sample_id': sample['sample_id'],
                    'parent_sample_id': sample.get('parent_sample_id', sample['sample_id']),
                    'source_row_index': row_index,
                    'video_path': video_path,
                    'frame_times': times,
                    'stream': dict(sample['stream']),
                    'timing_profile': timing_profile(sample),
                    **behavior,
                },
            ))
    if len(cases) < 1:
        raise ValueError(f'timing split contains no selected cases: {path}')
    info = {
        'path': str(Path(path).expanduser().resolve()),
        'sha256': file_sha256(path),
        'manifest_path': str(Path(manifest_path).expanduser().resolve()),
        'manifest_sha256': file_sha256(manifest_path),
        'source_rows': len(selected),
        'cases': len(cases),
        'video_roots': {source: [str(root) for root in values]
                        for source, values in roots.items()},
    }
    return cases, info


def _materialize_timing_epoch(
    cases: list[DualCase],
    *,
    epoch_steps: int,
    query_batch_size: int,
    seed: int,
    variant: str,
) -> tuple[list[DualCase], dict]:
    """Copy a deterministic epoch plan into fixed sequential case slots."""
    indices = timing_epoch_indices(
        [case.meta for case in cases],
        epoch_steps=epoch_steps,
        query_batch_size=query_batch_size,
        seed=seed,
        variant=variant,
        work_units=[_case_estimated_work(case) for case in cases],
    )
    materialized = []
    schedule_rows = []
    for epoch_slot, pool_index in enumerate(indices):
        source = cases[pool_index]
        case = copy.copy(source)
        case.row_index = epoch_slot
        case.meta = {
            **source.meta,
            'pool_case_id': source.row_index,
            'epoch_slot': epoch_slot,
            'timing_data_variant': variant,
        }
        materialized.append(case)
        row = {
            'epoch_slot': epoch_slot,
            'pool_case_id': source.row_index,
            'sample_id': source.meta['sample_id'],
            'source': source.meta['source'],
            'timing_profile': source.meta['timing_profile'],
        }
        if variant == BEHAVIOR_SOURCE_MIXED_DATA:
            row['behavior_category'] = source.meta['behavior_category']
        schedule_rows.append(row)
    encoded = json.dumps(schedule_rows, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    source_counts = {}
    profile_counts = {}
    behavior_counts = {}
    source_behavior_counts = {}
    for row in schedule_rows:
        source_counts[row['source']] = source_counts.get(row['source'], 0) + 1
        key = f'{row["source"]}/{row["timing_profile"]}'
        profile_counts[key] = profile_counts.get(key, 0) + 1
        behavior = row.get('behavior_category')
        if behavior is not None:
            behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1
            key = f'{row["source"]}/{behavior}'
            source_behavior_counts[key] = source_behavior_counts.get(key, 0) + 1
    runtime = {
        'variant': variant,
        'epoch_steps': epoch_steps,
        'scheduled_cases': len(materialized),
        'schedule_sha256': hashlib.sha256(encoded).hexdigest(),
        'source_counts': source_counts,
        'source_profile_counts': profile_counts,
    }
    if variant == BEHAVIOR_SOURCE_MIXED_DATA:
        runtime.update({
            'behavior_counts': behavior_counts,
            'source_behavior_counts': source_behavior_counts,
        })
    return materialized, runtime


def _training_batch(
    cases: list[DualCase],
    step: int,
    query_batch_size: int,
    seed: int,
    batch_schedule: str = LENGTH_BUCKETED,
) -> list[DualCase]:
    """Return a deterministic shuffled full-data batch without intra-batch duplicates."""
    work_units = [_case_estimated_work(case) for case in cases]
    indices = deterministic_batch_indices(
        len(cases),
        step,
        query_batch_size,
        seed,
        work_units=work_units,
        strategy=batch_schedule,
    )
    return [cases[index] for index in indices]


def _case_frame_count(case: DualCase) -> int:
    frame_times = case.meta.get('frame_times')
    return len(frame_times) if frame_times is not None else len(case.image_paths)


def _case_estimated_work(case: DualCase) -> int:
    return estimated_stream_work(max(1, _case_frame_count(case)))


def _clear_image_processor_cache(sm) -> None:
    image_processor = getattr(sm.processor, 'image_processor', None)
    clear = getattr(image_processor, 'clear', None)
    if callable(clear):
        clear()


def _load_pil_frames(cases: list[DualCase]) -> dict[int, list[Image.Image]]:
    loaded = {}
    for case in cases:
        if case.row_index in loaded:
            continue
        frames = []
        video_path = case.meta.get('video_path')
        frame_times = case.meta.get('frame_times')
        if video_path is not None:
            from examples.infer.proactive_eval.video_reader import VideoFrameReader
            if not frame_times:
                raise ValueError(f'benchmark case {case.row_index} has no frame_times')
            reader = VideoFrameReader(video_path)
            try:
                frames = [reader.read_frame(float(timestamp)).convert('RGB') for timestamp in frame_times]
            finally:
                reader.close()
        else:
            for path in case.image_paths:
                with Image.open(path) as image:
                    frames.append(image.convert('RGB'))
        loaded[case.row_index] = frames
    return loaded


def rollout_trajectories(
    sm,
    tasks,
    *,
    args,
    capture_branch_logits: bool = False,
    stats: Optional[dict] = None,
) -> list[DualTrajectory]:
    """Run task tuples ``(prompt_slot, group_index, case, seed)``."""
    from examples.infer.proactive_eval.runner import ProactiveBatchRunner

    tasks = list(tasks)
    if not tasks:
        return []
    unique_cases = list({task[2].row_index: task[2] for task in tasks}.values())
    pil_frames = _load_pil_frames(unique_cases)
    chunk_size = (args.rollout_trajectory_microbatch_size or args.local_query_batch_size)
    if chunk_size < 1:
        raise ValueError('rollout_query_batch_size must be >= 1 or 0=all')
    reuse_visual_features = bool(getattr(args, 'reuse_frozen_visual_features', False))
    prepared_image_cache = {}
    prepared_image_uses = 0
    trajectories = []
    for chunk_start in range(0, len(tasks), chunk_size):
        chunk = tasks[chunk_start:chunk_start + chunk_size]
        samplers = [
            DualBranchSampler(
                sm.tokenizer,
                max_response_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=seed,
            ) for _, _, _, seed in chunk
        ]
        batch = ProactiveBatchRunner(
            sm,
            [case.query for _, _, case, _ in chunk],
            evict=True,
            max_tokens=args.max_tokens,
            use_streaming_cache=args.streaming_cache,
            max_kv_len=args.max_kv_len,
            query_batch_mode=args.rollout_query_batch_mode,
            samplers=samplers,
        )
        frame_records = [[] for _ in chunk]
        max_ticks = max(_case_frame_count(case) for _, _, case, _ in chunk)
        for frame_idx in range(max_ticks):
            frames = [
                pil_frames[case.row_index][frame_idx] if frame_idx < _case_frame_count(case) else None
                for _, _, case, _ in chunk
            ]
            prepared_features = None
            if reuse_visual_features:
                prepared_features = []
                for frame, (_, _, case, _) in zip(frames, chunk):
                    if frame is None:
                        prepared_features.append(None)
                        continue
                    cache_key = (case.row_index, frame_idx)
                    if cache_key not in prepared_image_cache:
                        prepared_image_cache[cache_key] = sm.prepare_image_features_for_reuse(frame)
                    prepared_features.append(prepared_image_cache[cache_key])
                    prepared_image_uses += 1
            step_kwargs = {}
            if prepared_features is not None:
                step_kwargs['prepared_image_features'] = prepared_features
            if capture_branch_logits:
                step_kwargs['capture_branch_logits'] = True
            outputs = batch.step(frames, **step_kwargs)
            for row, output in enumerate(outputs):
                if frames[row] is None:
                    if output is not None:
                        raise RuntimeError('inactive rollout row produced output')
                    continue
                if output is None or output.get('mode') == 'SKIP':
                    raise RuntimeError(f'active rollout row {row} produced {output!r}')
                output = dict(output)
                output['frame_idx'] = frame_idx
                frame_times = chunk[row][2].meta.get('frame_times')
                if frame_times is not None:
                    output['t'] = float(frame_times[frame_idx])
                if output['B_forced_stop'] or output['A_forced_stop']:
                    raise RuntimeError('grammar-constrained rollout reached safety close; '
                                       'this indicates a sampler/runner contract bug')
                frame_records[row].append(output)

        for row, (prompt_slot, group_index, case, seed) in enumerate(chunk):
            stream = batch.streams[row]
            trajectories.append(
                DualTrajectory(
                    prompt_slot=prompt_slot,
                    group_index=group_index,
                    case=case,
                    messages=copy.deepcopy(stream.messages),
                    frames=frame_records[row],
                    physical_ids=stream.runner.state.input_ids_np.copy(),
                    seed=seed,
                    images=pil_frames[case.row_index],
                    image_embeds=([
                        prepared_image_cache[(case.row_index, frame_index)].image_embeds
                        for frame_index in range(_case_frame_count(case))
                    ] if reuse_visual_features else []),
                    image_grids=([
                        tuple(
                            int(value) for value in prepared_image_cache[
                                (case.row_index, frame_index)
                            ].image_grid_thw.flatten().tolist())
                        for frame_index in range(_case_frame_count(case))
                    ] if reuse_visual_features else []),
                ))
        del batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if stats is not None:
        stats.update({
            'enabled': reuse_visual_features,
            'unique_images': len(prepared_image_cache),
            'trajectory_image_uses': prepared_image_uses,
            'reused_image_uses': prepared_image_uses - len(prepared_image_cache),
        })
    return trajectories


def _timing_metric_summary(rows: list[dict]) -> dict:
    if not rows:
        return {'n': 0}
    precision = _mean([row['precision'] for row in rows])
    recall = _mean([row['recall'] for row in rows])
    summary = {
        'n': len(rows),
        'mean_f1': _mean([row['f1'] for row in rows]),
        'macro_precision': precision,
        'macro_recall': recall,
        'macro_f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        'first_speak_accuracy': _mean([row['first_speak_accuracy'] for row in rows]),
        'silent_rate': _mean([row['silent'] for row in rows]),
        'mean_speak_count': _mean([row['speak_count'] for row in rows]),
        'tp': sum(row['tp'] for row in rows),
        'fp': sum(row['fp'] for row in rows),
        'fn': sum(row['fn'] for row in rows),
        'out_of_window_speak_count': sum(row['out_of_window_speak_count'] for row in rows),
        'duplicate_speak_count': sum(row['duplicate_speak_count'] for row in rows),
    }
    return summary


def _merge_timing_summaries(rows: list[dict]) -> Optional[dict]:
    rows = [row for row in rows if row and int(row.get('n', 0)) > 0]
    if not rows:
        return None
    total = sum(int(row['n']) for row in rows)
    if total < 1:
        return {'n': 0}

    def weighted(key):
        return sum(float(row[key]) * int(row['n']) for row in rows) / total

    precision = weighted('macro_precision')
    recall = weighted('macro_recall')
    summary = {
        'n': total,
        'mean_f1': weighted('mean_f1'),
        'macro_precision': precision,
        'macro_recall': recall,
        'macro_f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        'first_speak_accuracy': weighted('first_speak_accuracy'),
        'silent_rate': weighted('silent_rate'),
        'mean_speak_count': weighted('mean_speak_count'),
        **{
            key: sum(int(row[key]) for row in rows)
            for key in ('tp', 'fp', 'fn', 'out_of_window_speak_count', 'duplicate_speak_count')
        },
    }
    return summary


@torch.no_grad()
def evaluate_timing_cases(sm, cases: list[DualCase], *, args, step: int, output_dir: Path) -> dict:
    """Run benchmark-style greedy infer on the complete canonical val streams."""
    from examples.infer.proactive_eval.runner import ProactiveBatchRunner
    from examples.infer.proactive_eval.video_reader import VideoFrameReader

    selected = cases
    predictions = []
    rewards = []
    started = time.perf_counter()
    prediction_dir = output_dir / 'validation_predictions'
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for chunk_start in range(0, len(selected), args.eval_query_batch_size):
        chunk = selected[chunk_start:chunk_start + args.eval_query_batch_size]
        batch = ProactiveBatchRunner(
            sm,
            [case.query for case in chunk],
            evict=True,
            max_tokens=args.eval_max_tokens,
            use_streaming_cache=args.streaming_cache,
            max_kv_len=args.max_kv_len,
            query_batch_mode=args.rollout_query_batch_mode,
        )
        readers = [VideoFrameReader(case.meta['video_path']) for case in chunk]
        frame_rows = [[] for _ in chunk]
        try:
            max_ticks = max(_case_frame_count(case) for case in chunk)
            for frame_index in range(max_ticks):
                images = [
                    readers[index].read_frame(case.meta['frame_times'][frame_index])
                    if frame_index < _case_frame_count(case) else None for index, case in enumerate(chunk)
                ]
                outputs = batch.step(images)
                for index, output in enumerate(outputs):
                    if images[index] is None:
                        if output is not None:
                            raise RuntimeError('inactive validation row produced output')
                        continue
                    if output is None or output.get('mode') == 'SKIP':
                        raise RuntimeError(f'active validation row produced {output!r}')
                    row = dict(output)
                    row['t'] = float(chunk[index].meta['frame_times'][frame_index])
                    row['frame_idx'] = frame_index
                    frame_rows[index].append(row)
        finally:
            for reader in readers:
                reader.close()

        for case, frames in zip(chunk, frame_rows):
            reward = aggregate_dual_reward(frames, case.gt, DualRewardWeights())
            reward['source'] = case.meta['source']
            rewards.append(reward)
            predictions.append({
                'step':
                step,
                'sample_id':
                case.meta['sample_id'],
                'source':
                case.meta['source'],
                'task':
                case.meta['task'],
                'gt_windows':
                case.gt['gt_windows'],
                'speaks': [{
                    't': frame['t'],
                    'content': frame.get('speak', ''),
                } for frame in frames if frame.get('mode') == 'SPEAK'],
                'metrics': {
                    key: reward[key]
                    for key in ('tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'first_speak_accuracy', 'speak_count',
                                'silent', 'out_of_window_speak_count', 'duplicate_speak_count')
                },
            })
        del batch
        _clear_image_processor_cache(sm)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    by_source = {
        source: _timing_metric_summary([row for row in rewards if row['source'] == source])
        for source in sorted({row['source']
                              for row in rewards})
    }
    summary = {
        'step': step,
        'elapsed_s': time.perf_counter() - started,
        'overall': _timing_metric_summary(rewards),
        'sources': by_source,
    }
    prediction_path = prediction_dir / f'step-{step}.rank{args.rank}.jsonl'
    prediction_path.write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in predictions),
        encoding='utf-8',
    )
    summary['prediction_path'] = str(prediction_path)
    return summary


def _validation_shards(cases: list[DualCase], world_size: int) -> list[list[DualCase]]:
    """Balance complete streams across ranks by frame count."""
    shards = [[] for _ in range(world_size)]
    loads = [0] * world_size
    for case in sorted(cases, key=lambda row: (-_case_frame_count(row), row.row_index)):
        rank = min(range(world_size), key=lambda value: (loads[value], value))
        shards[rank].append(case)
        loads[rank] += _case_frame_count(case)
    return shards


def _aggregate_validation_summaries(rows: list[dict], step: int) -> dict:
    sources = sorted({source for row in rows for source in row['sources']})
    return {
        'step': step,
        'elapsed_s': max((row['elapsed_s'] for row in rows), default=0.0),
        'overall': _merge_timing_summaries([row['overall'] for row in rows]),
        'sources':
        {source: _merge_timing_summaries([row['sources'].get(source) for row in rows])
         for source in sources},
        'prediction_paths': [row['prediction_path'] for row in rows],
    }


def run_timing_validation(sm, cases: list[DualCase], *, args, step: int, output_dir: Path) -> Optional[dict]:
    sm.model.eval()
    selected = cases[:args.eval_max_cases] if args.eval_max_cases else cases
    shards = _validation_shards(selected, args.world_size)
    local = evaluate_timing_cases(sm, shards[args.rank], args=args, step=step, output_dir=output_dir)
    if args.distributed:
        rows = [None] * args.world_size
        dist.all_gather_object(rows, local)
    else:
        rows = [local]
    if args.rank != 0:
        return None
    summary = _aggregate_validation_summaries(rows, step)
    with (output_dir / 'validation.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(summary, ensure_ascii=False) + '\n')
    return summary


def build_rollout_tasks(selected_cases: list[DualCase],
                        *,
                        group_size: int,
                        seed: int,
                        step: int,
                        prompt_slot_offset: int = 0,
                        prompt_slots=None):
    """Create G waves, each containing one trajectory per distinct query.

    When ``rollout_trajectory_microbatch_size == query_batch_size``, this ordering makes
    every fused chunk a real multi-query batch instead of G replicas of the
    first prompt.  Prompt/group identifiers keep reward normalization
    independent of execution order.
    """
    if prompt_slots is None:
        prompt_slots = range(prompt_slot_offset, prompt_slot_offset + len(selected_cases))
    else:
        if prompt_slot_offset != 0:
            raise ValueError('prompt_slot_offset and explicit prompt_slots are mutually exclusive')
        prompt_slots = [int(slot) for slot in prompt_slots]
        if len(prompt_slots) != len(selected_cases) or len(set(prompt_slots)) != len(prompt_slots):
            raise ValueError('prompt_slots must contain one distinct slot per selected case')

    tasks = []
    for group_index in range(group_size):
        for prompt_slot, case in zip(prompt_slots, selected_cases):
            trajectory_seed = (seed + step * 1_000_003 + prompt_slot * 10_007 + group_index)
            tasks.append((prompt_slot, group_index, case, trajectory_seed))
    return tasks


def _advantages_by_query(trajectories: list[DualTrajectory], *, prompt_slots, group_size: int) -> list[float]:
    """Normalize rewards only across the G rollouts of each query."""
    prompt_slots = [int(slot) for slot in prompt_slots]
    expected_keys = {(slot, group_index) for slot in prompt_slots for group_index in range(group_size)}
    actual_keys = [(int(trajectory.prompt_slot), int(trajectory.group_index)) for trajectory in trajectories]
    if len(actual_keys) != len(expected_keys) or set(actual_keys) != expected_keys:
        raise RuntimeError('rollouts do not cover each (query, group) exactly once: '
                           f'expected={sorted(expected_keys)} actual={sorted(actual_keys)}')

    advantages = [None] * len(trajectories)
    for prompt_slot in prompt_slots:
        indices = [index for index, trajectory in enumerate(trajectories) if trajectory.prompt_slot == prompt_slot]
        rewards = [float(trajectories[index].reward['total']) for index in indices]
        if not np.isfinite(rewards).all():
            raise RuntimeError(f'query {prompt_slot} has a non-finite reward')
        for index, advantage in zip(indices, group_relative_advantage(rewards)):
            advantages[index] = float(advantage)
    if not np.isfinite(advantages).all():
        raise RuntimeError('group-relative advantage is non-finite')
    return advantages


def _actor_group_cost_summaries(
    trajectories: list[DualTrajectory],
    *,
    prompt_slots: list[int],
    group_size: int,
) -> list[dict]:
    """Describe complete query groups without copying actor payloads."""
    prompt_slots = [int(slot) for slot in prompt_slots]
    expected = {
        (prompt_slot, group_index)
        for prompt_slot in prompt_slots
        for group_index in range(group_size)
    }
    actual = [
        (int(trajectory.prompt_slot), int(trajectory.group_index))
        for trajectory in trajectories
    ]
    if len(actual) != len(expected) or set(actual) != expected:
        raise RuntimeError('actor cost summary must cover each local (query, group) once')

    summaries = []
    for prompt_slot in sorted(prompt_slots):
        rows = [
            trajectory for trajectory in trajectories
            if int(trajectory.prompt_slot) == prompt_slot
        ]
        case_ids = {int(trajectory.case.row_index) for trajectory in rows}
        if len(case_ids) != 1:
            raise RuntimeError(f'actor query {prompt_slot} contains multiple cases')
        summaries.append({
            'prompt_slot': prompt_slot,
            'case_id': next(iter(case_ids)),
            'sampled_decision_cost': sum(
                len(trajectory_decisions(trajectory)) for trajectory in rows),
        })
    return summaries


def _rebalance_actor_trajectories(
    sm,
    trajectories: list[DualTrajectory],
    advantages: list[float],
    *,
    global_selected_cases: list[DualCase],
    local_prompt_slots: list[int],
    args,
) -> tuple[list[DualTrajectory], list[float], dict]:
    """Optionally move complete query groups after rollout.

    Rollout-owned trajectories remain authoritative for reward reporting,
    fingerprints, and group diagnostics.  Actor replay receives a separate
    list.  Groups that stay local reuse their exact PIL/frozen-visual
    objects; only incoming groups are rebuilt from canonical case metadata.
    """
    local_prompt_slots = [int(slot) for slot in local_prompt_slots]
    local_cost = sum(len(trajectory_decisions(row)) for row in trajectories)
    minimum_gain = float(args.actor_rank_rebalance_min_gain)
    base_info = {
        'mode': args.actor_rank_rebalance,
        'effective': False,
        'skip_reason': 'disabled',
        'rollout_prompt_slots': local_prompt_slots,
        'actor_prompt_slots': local_prompt_slots,
        'actor_case_ids': [int(global_selected_cases[slot].row_index) for slot in local_prompt_slots],
        'sampled_decision_loads': None,
        'current_sampled_decision_loads': None,
        'planned_sampled_decision_loads': None,
        'predicted_max_load_relative_gain': None,
        'minimum_relative_gain': minimum_gain,
        'local_sampled_decision_load': local_cost,
        'moved_out_query_count': 0,
        'incoming_query_count': 0,
        'rebuilt_image_count': 0,
    }
    if not actor_rebalance_enabled(args.actor_rank_rebalance, args.world_size):
        # Exact no-op: preserve list identity, ordering, PIL objects, and GPU
        # visual tensors.  In particular, post_rollout_groups_v1 on one rank
        # must behave exactly like none.
        return trajectories, advantages, base_info
    if not args.distributed or not dist.is_initialized():
        raise RuntimeError('actor rank rebalance requires an initialized distributed process group')

    local_costs = _actor_group_cost_summaries(
        trajectories,
        prompt_slots=local_prompt_slots,
        group_size=args.group_size,
    )
    gathered_costs = [None] * args.world_size
    dist.all_gather_object(gathered_costs, local_costs)
    cost_plan = plan_gathered_actor_group_costs(
        gathered_costs,
        world_size=args.world_size,
        local_query_capacity=args.local_query_batch_size,
    )
    current_loads = [int(value) for value in cost_plan['current_sampled_decision_loads']]
    planned_loads = [int(value) for value in cost_plan['sampled_decision_loads']]
    predicted_gain = float(cost_plan['max_load_relative_gain'])
    plan_info = {
        'current_sampled_decision_loads': current_loads,
        'planned_sampled_decision_loads': planned_loads,
        'predicted_max_load_relative_gain': predicted_gain,
    }
    if predicted_gain <= minimum_gain:
        return trajectories, advantages, {
            **base_info,
            **plan_info,
            'skip_reason': 'below_minimum_gain',
        }

    local_groups = compact_actor_groups(
        trajectories,
        advantages,
        prompt_slots=local_prompt_slots,
        group_size=args.group_size,
    )
    gathered_groups = [None] * args.world_size
    dist.all_gather_object(gathered_groups, local_groups)
    plan = plan_gathered_actor_groups(
        gathered_groups,
        world_size=args.world_size,
        local_query_capacity=args.local_query_batch_size,
        group_size=args.group_size,
    )
    for field_name in (
        'prompt_slots_by_rank',
        'sampled_decision_loads',
        'current_prompt_slots_by_rank',
        'current_sampled_decision_loads',
        'max_load_relative_gain',
    ):
        if plan[field_name] != cost_plan[field_name]:
            raise RuntimeError(
                f'actor cost/full-payload plans differ at {field_name}')
    compact_rows, planned_advantages = actor_rows_for_rank(
        plan,
        rank=args.rank,
        group_size=args.group_size,
    )
    actor_prompt_slots = [int(slot) for slot in plan['prompt_slots_by_rank'][args.rank]]

    case_by_prompt_slot = dict(enumerate(global_selected_cases))
    if sorted(case_by_prompt_slot) != list(range(args.query_batch_size)):
        raise RuntimeError('global selected cases do not cover global prompt slots')
    for prompt_slot, group in plan['groups_by_prompt_slot'].items():
        expected_case_id = int(case_by_prompt_slot[prompt_slot].row_index)
        if int(group['case_id']) != expected_case_id:
            raise RuntimeError('actor payload case/prompt mapping differs from the '
                               f'global rollout batch at slot {prompt_slot}')

    original_by_key = {}
    for trajectory, advantage in zip(trajectories, advantages):
        key = (int(trajectory.prompt_slot), int(trajectory.group_index))
        if key in original_by_key:
            raise RuntimeError(f'duplicate local rollout trajectory key: {key}')
        original_by_key[key] = (trajectory, float(advantage))

    incoming_slots = sorted(set(actor_prompt_slots) - set(local_prompt_slots))
    moved_out_slots = sorted(set(local_prompt_slots) - set(actor_prompt_slots))

    # Reporting/fingerprints use only rollout token metadata.  Drop visual
    # objects for moved-out groups before rebuilding incoming groups so the
    # CUDA allocator can reuse that memory instead of holding both copies at
    # the actor-rebalance peak.
    for trajectory in trajectories:
        if int(trajectory.prompt_slot) in moved_out_slots:
            trajectory.images = []
            trajectory.image_embeds = []
            trajectory.image_grids = []

    incoming_cases = [case_by_prompt_slot[slot] for slot in incoming_slots]
    incoming_images = _load_pil_frames(incoming_cases)
    prepared_by_slot = {}
    if args.reuse_frozen_visual_features:
        for prompt_slot in incoming_slots:
            case = case_by_prompt_slot[prompt_slot]
            features = [
                sm.prepare_image_features_for_reuse(frame)
                for frame in incoming_images[case.row_index]
            ]
            prepared_by_slot[prompt_slot] = (
                [item.image_embeds for item in features],
                [tuple(int(value) for value in item.image_grid_thw.flatten().tolist())
                 for item in features],
            )

    actor_trajectories = []
    actor_advantages = []
    for row, planned_advantage in zip(compact_rows, planned_advantages):
        key = (int(row['prompt_slot']), int(row['group_index']))
        if key in original_by_key:
            trajectory, original_advantage = original_by_key[key]
            if original_advantage != planned_advantage:
                raise RuntimeError(f'local actor advantage changed for {key}')
        else:
            prompt_slot = key[0]
            case = case_by_prompt_slot[prompt_slot]
            images = incoming_images[case.row_index]
            if len(images) != len(row['frames']):
                raise RuntimeError('incoming actor frame count differs from canonical case: '
                                   f'slot={prompt_slot} images={len(images)} '
                                   f'frames={len(row["frames"])}')
            prepared = prepared_by_slot.get(prompt_slot)
            trajectory = DualTrajectory(
                prompt_slot=prompt_slot,
                group_index=key[1],
                case=case,
                messages=copy.deepcopy(row['messages']),
                frames=copy.deepcopy(row['frames']),
                physical_ids=np.asarray(row['physical_ids'], dtype=np.int64).copy(),
                seed=int(row['seed']),
                images=images,
                image_embeds=([] if prepared is None else prepared[0]),
                image_grids=([] if prepared is None else prepared[1]),
                reward=copy.deepcopy(row['reward']),
            )
        actor_trajectories.append(trajectory)
        actor_advantages.append(float(planned_advantage))

    expected_actor_keys = {
        (slot, group_index)
        for slot in actor_prompt_slots
        for group_index in range(args.group_size)
    }
    actual_actor_keys = {
        (int(row.prompt_slot), int(row.group_index)) for row in actor_trajectories
    }
    if len(actor_trajectories) != len(expected_actor_keys) or actual_actor_keys != expected_actor_keys:
        raise RuntimeError('rebuilt actor rows do not cover the assigned complete-G groups')

    return actor_trajectories, actor_advantages, {
        **base_info,
        **plan_info,
        'effective': True,
        'skip_reason': None,
        'actor_prompt_slots': actor_prompt_slots,
        'actor_case_ids': [int(case_by_prompt_slot[slot].row_index) for slot in actor_prompt_slots],
        'sampled_decision_loads': planned_loads,
        'local_sampled_decision_load': planned_loads[args.rank],
        'moved_out_query_count': len(moved_out_slots),
        'incoming_query_count': len(incoming_slots),
        'rebuilt_image_count': sum(len(incoming_images[case.row_index]) for case in incoming_cases),
    }


def _summarize_group_diagnostics(rows: list[dict]) -> dict:
    """Summarize how often stochastic groups differ in tokens and reward."""
    group_count = len(rows)
    trajectory_diverse = sum(int(row['unique_trajectory_count'] > 1) for row in rows)
    informative = sum(int(row['informative']) for row in rows)
    silent_observed = [row for row in rows if row['silent_observed']]
    all_silent = sum(int(row['all_silent']) for row in silent_observed)
    all_advantages = [value for row in rows for value in row['advantages']]
    advantage_mean = _mean(all_advantages) if all_advantages else 0.0
    denominator = max(group_count, 1)
    silent_denominator = max(len(silent_observed), 1)
    return {
        'group_count': group_count,
        'trajectory_diverse_group_count': trajectory_diverse,
        'trajectory_diverse_group_rate': trajectory_diverse / denominator,
        'informative_group_count': informative,
        'informative_group_rate': informative / denominator,
        'silent_observed_group_count': len(silent_observed),
        'all_silent_group_count': all_silent,
        'all_silent_group_rate': all_silent / silent_denominator,
        'mean_unique_trajectories': sum(row['unique_trajectory_count'] for row in rows) / denominator,
        'mean_unique_rewards': sum(row['unique_reward_count'] for row in rows) / denominator,
        'mean_reward_std': sum(row['reward_std'] for row in rows) / denominator,
        'mean_reward_range': sum(row['reward_range'] for row in rows) / denominator,
        'mean_abs_advantage': sum(abs(value) for value in all_advantages) / max(len(all_advantages), 1),
        'advantage_std': (sum((value - advantage_mean)**2 for value in all_advantages)
                          / max(len(all_advantages), 1))**0.5,
    }


def _group_sampling_diagnostics(trajectories, advantages, rollout_fingerprints) -> tuple[dict, list[dict]]:
    """Report trajectory diversity separately from reward diversity per query."""
    if not (len(trajectories) == len(advantages) == len(rollout_fingerprints)):
        raise ValueError('trajectory/advantage/fingerprint count mismatch')
    fingerprints = {
        (int(row['prompt_slot']), int(row['group_index'])): row for row in rollout_fingerprints
    }
    if len(fingerprints) != len(rollout_fingerprints):
        raise RuntimeError('duplicate rollout fingerprint key')

    rows = []
    for prompt_slot in sorted({int(trajectory.prompt_slot) for trajectory in trajectories}):
        indices = sorted(
            (index for index, trajectory in enumerate(trajectories)
             if int(trajectory.prompt_slot) == prompt_slot),
            key=lambda index: int(trajectories[index].group_index),
        )
        group_rows = [trajectories[index] for index in indices]
        case_ids = {int(trajectory.case.row_index) for trajectory in group_rows}
        if len(case_ids) != 1:
            raise RuntimeError(f'query {prompt_slot} contains multiple cases: {sorted(case_ids)}')
        rewards = [float(trajectory.reward['total']) for trajectory in group_rows]
        group_advantages = [float(advantages[index]) for index in indices]
        reward_mean = _mean(rewards)
        advantage_mean = _mean(group_advantages)
        silent_values = [trajectory.reward.get('silent') for trajectory in group_rows]
        silent_observed = all(value is not None for value in silent_values)
        digests = [
            fingerprints[(prompt_slot, int(trajectory.group_index))]['physical_ids_sha256']
            for trajectory in group_rows
        ]
        rows.append({
            'prompt_slot': prompt_slot,
            'case_id': next(iter(case_ids)),
            'rewards': rewards,
            'advantages': group_advantages,
            'unique_trajectory_count': len(set(digests)),
            'unique_reward_count': len(set(rewards)),
            'reward_std': (_mean([(value - reward_mean)**2 for value in rewards]))**0.5,
            'reward_range': max(rewards) - min(rewards),
            'mean_abs_advantage': _mean([abs(value) for value in group_advantages]),
            'advantage_std': (_mean([(value - advantage_mean)**2
                                     for value in group_advantages]))**0.5,
            'informative': any(abs(value) > 0 for value in group_advantages),
            'silent_observed': silent_observed,
            'all_silent': silent_observed and all(bool(value) for value in silent_values),
        })
    return _summarize_group_diagnostics(rows), rows


def _global_mean_loss_contribution(loss: torch.Tensor, *, query_batch_size: int, group_size: int) -> torch.Tensor:
    """Scale one trajectory so cross-rank SUM equals the global batch mean."""
    return loss / (query_batch_size * group_size)


def trajectory_decisions(trajectory: DualTrajectory) -> list[DualDecision]:
    decisions = []
    for frame_index, frame in enumerate(trajectory.frames):
        for route in ('B', 'A'):
            tokens = [int(token) for token in frame[f'{route}_tokens']]
            positions = [int(position) for position in frame[f'{route}_token_positions']]
            logps = list(frame[f'{route}_token_logps'])
            if not (len(tokens) == len(positions) == len(logps)):
                raise RuntimeError(f'{route} rollout token metadata length mismatch')
            history = []
            for token_index, (token, position, logp) in enumerate(zip(tokens, positions, logps)):
                if logp is None:
                    raise RuntimeError('constrained RL rollout contains a non-policy token '
                                       f'inside {route} output at physical position {position}')
                rollout_logp = float(logp)
                if not np.isfinite(rollout_logp):
                    raise RuntimeError('constrained RL rollout contains non-finite logp at '
                                       f'{route} physical position {position}: {rollout_logp}')
                decisions.append(
                    DualDecision(
                        route=route,
                        frame_index=frame_index,
                        token_index=token_index,
                        token_id=token,
                        position=position,
                        rollout_logp=rollout_logp,
                        history=list(history),
                    ))
                history.append(token)
    decisions.sort(key=lambda decision: decision.position)
    if not decisions:
        raise RuntimeError('trajectory contains no sampled policy decisions')
    positions = [decision.position for decision in decisions]
    if len(set(positions)) != len(positions):
        raise RuntimeError('duplicate sampled physical positions')
    return decisions


def trajectory_rollout_fingerprint(
    trajectory: DualTrajectory,
    *,
    include_trace: bool = False,
) -> dict:
    """Stable exact token/logp identity for cross-mode CUDA comparisons."""
    physical_ids = np.asarray(trajectory.physical_ids, dtype='<i8').reshape(-1)
    physical_digest = hashlib.sha256(physical_ids.tobytes(order='C')).hexdigest()
    decisions = trajectory_decisions(trajectory)
    trace_rows = [
        {
            'route': decision.route,
            'frame_index': int(decision.frame_index),
            'token_index': int(decision.token_index),
            'token_id': int(decision.token_id),
            'position': int(decision.position),
            # JSON decimal rendering is implementation dependent; float.hex()
            # preserves the exact IEEE-754 rollout value for digest comparison.
            'rollout_logp_hex': float(decision.rollout_logp).hex(),
        } for decision in decisions
    ]
    trace_digest = hashlib.sha256(
        json.dumps(
            trace_rows,
            ensure_ascii=True,
            separators=(',', ':'),
            sort_keys=True,
        ).encode('utf-8')).hexdigest()
    result = {
        'case_id': int(trajectory.case.row_index),
        'prompt_slot': int(trajectory.prompt_slot),
        'group_index': int(trajectory.group_index),
        'seed': int(trajectory.seed),
        'frame_count': len(trajectory.frames),
        'physical_token_count': int(physical_ids.size),
        'sampled_decision_count': len(decisions),
        'sampled_decisions_by_route':
        {route: sum(decision.route == route for decision in decisions)
         for route in ('B', 'A')},
        'physical_ids_sha256': physical_digest,
        'rollout_policy_trace_sha256': trace_digest,
    }
    if include_trace:
        result['rollout_policy_trace_fields'] = [
            'route',
            'frame_index',
            'token_index',
            'token_id',
            'position',
            'rollout_logp',
        ]
        result['rollout_policy_trace'] = [[
            decision.route,
            int(decision.frame_index),
            int(decision.token_index),
            int(decision.token_id),
            int(decision.position),
            float(decision.rollout_logp),
        ] for decision in decisions]
    return result


def _without_dual_spans(ids: np.ndarray, meta) -> np.ndarray:
    keep = np.ones(len(ids), dtype=np.bool_)
    for route, spans in (('B', meta.b_spans), ('A', meta.a_spans)):
        for start, end in spans:
            if not 0 <= start < end <= len(ids):
                raise RuntimeError(f'illegal {route} span [{start}, {end}) for ' f'length {len(ids)}')
            if not keep[start:end].all():
                raise RuntimeError(f'overlapping dual span [{start}, {end})')
            keep[start:end] = False
    return ids[keep]


def encode_training_trajectory(template, trajectory: DualTrajectory, assistant_role_id: int):
    """Build exact-token actor input with training multimodal/layout logic.

    Generated token IDs are authoritative.  Decoding them to message text and
    tokenizing again is not safe because BPE decode->encode is not generally
    idempotent.  A fixed legal dual target lets the template build the system,
    query, image tensors, source-fork, and chat scaffold.  We prove that the
    non-branch scaffold is identical, then replace only the variable sequence
    with the exact physical rollout IDs.
    """
    skeleton_messages = copy.deepcopy(trajectory.messages)
    assistant_count = 0
    for message in skeleton_messages:
        if message.get('role') == 'assistant':
            message['content'] = SKELETON_DUAL_TARGET
            assistant_count += 1
    if assistant_count != len(trajectory.frames):
        raise RuntimeError(f'rollout has {assistant_count} assistant turns but '
                           f'{len(trajectory.frames)} frame records')
    encoded = template.encode({
        'messages': skeleton_messages,
        'images': trajectory.images or trajectory.case.image_paths,
    })
    skeleton_ids = np.asarray(encoded['input_ids'], dtype=np.int64)
    encoded_ids = np.asarray(trajectory.physical_ids, dtype=np.int64)
    skeleton_meta = build_step_mask_meta(
        skeleton_ids,
        assistant_role_id=assistant_role_id,
        include_branch_source=True,
    )
    meta = build_step_mask_meta(
        encoded_ids,
        assistant_role_id=assistant_role_id,
        include_branch_source=True,
    )
    expected_turns = len(trajectory.frames)
    if (not skeleton_meta.is_dual_branch or not meta.is_dual_branch or len(skeleton_meta.b_spans) != expected_turns
            or len(skeleton_meta.a_spans) != expected_turns or len(meta.b_spans) != expected_turns
            or len(meta.a_spans) != expected_turns):
        raise RuntimeError('skeleton/rollout dual turn count mismatch: '
                           f'expected={expected_turns} '
                           f'skeleton_B/A={len(skeleton_meta.b_spans)}/'
                           f'{len(skeleton_meta.a_spans)} '
                           f'rollout_B/A={len(meta.b_spans)}/{len(meta.a_spans)}')
    skeleton_scaffold = _without_dual_spans(skeleton_ids, skeleton_meta)
    rollout_scaffold = _without_dual_spans(encoded_ids, meta)
    if not np.array_equal(skeleton_scaffold, rollout_scaffold):
        common = min(len(skeleton_scaffold), len(rollout_scaffold))
        mismatch = np.flatnonzero(skeleton_scaffold[:common] != rollout_scaffold[:common])
        first = int(mismatch[0]) if mismatch.size else common
        raise RuntimeError('rollout/training non-branch scaffold divergence: '
                           f'first={first} rollout_len={len(rollout_scaffold)} '
                           f'train_len={len(skeleton_scaffold)}')

    decisions = trajectory_decisions(trajectory)
    labels = np.full(len(encoded_ids), -100, dtype=np.int64)
    for decision in decisions:
        position = decision.position
        if not 0 < position < len(encoded_ids):
            raise RuntimeError(f'illegal sampled position {position}')
        if int(encoded_ids[position]) != decision.token_id:
            raise RuntimeError(f'sampled token mismatch at {position}: '
                               f'{decision.token_id} != {encoded_ids[position]}')
        labels[position] = decision.token_id
        route_mask = meta.route_A if decision.route == 'A' else meta.route_B
        other_mask = meta.route_B if decision.route == 'A' else meta.route_A
        if not bool(route_mask[position]) or bool(other_mask[position]):
            raise RuntimeError(f'sampled {decision.route} token has wrong training route '
                               f'at physical position {position}')
        spans = meta.a_spans if decision.route == 'A' else meta.b_spans
        span_start, span_end = spans[decision.frame_index]
        prediction_position = position - 1
        if not (span_start < position < span_end and span_start <= prediction_position < span_end):
            raise RuntimeError(f'sampled {decision.route} token for frame '
                               f'{decision.frame_index} lies outside its training span: '
                               f'target={position} prediction={prediction_position} '
                               f'span=[{span_start}, {span_end})')
        if (not bool(route_mask[prediction_position]) or bool(other_mask[prediction_position])):
            raise RuntimeError(f'loss logit for sampled {decision.route} token has wrong '
                               f'training route at physical position {prediction_position}')
    encoded['input_ids'] = encoded_ids.tolist()
    encoded['labels'] = labels.tolist()
    if encoded.get('loss_scale') is not None:
        encoded['loss_scale'] = [1.0 if label >= 0 else 0.0 for label in labels]
    encoded['length'] = len(encoded_ids)
    return encoded, decisions


def _prepare_model_inputs(
    collated: dict,
    prediction_positions: torch.Tensor,
    device,
    *,
    reuse_frozen_visual_features: bool = False,
) -> dict:
    inputs = dict(collated)
    for key in ('labels', 'text_position_ids', 'loss_scale', 'channel'):
        inputs.pop(key, None)
    if reuse_frozen_visual_features:
        pixel_values = inputs.get('pixel_values')
        if not isinstance(pixel_values, torch.Tensor) or pixel_values.numel() < 1:
            raise RuntimeError('frozen visual replay requires non-empty collated pixel values')
        if inputs.get('pixel_values_videos') is not None:
            raise RuntimeError('frozen image replay does not support mixed video pixel values')
        # The scoped actor-visual hook consumes the already verified image
        # embeddings before patch projection. Keep a non-None, zero-element
        # sentinel so Swift still takes its image replacement path without
        # copying unused raw patches to the GPU.
        inputs['pixel_values'] = pixel_values.new_empty((0, ))
    inputs['logits_to_keep'] = prediction_positions
    inputs['use_cache'] = False
    return to_device(inputs, device)


def _policy_decisions(trajectory, policy_sampler):
    decisions = []
    for decision in trajectory_decisions(trajectory):
        deterministic = policy_sampler.is_structurally_deterministic(stream=decision.route, history=decision.history)
        if deterministic:
            if abs(decision.rollout_logp) > 1e-6:
                raise RuntimeError('structurally deterministic rollout token has non-zero '
                                   f'logp at {decision.position}: {decision.rollout_logp}')
            continue
        decisions.append(decision)
    if not decisions:
        raise RuntimeError('trajectory contains no non-deterministic decisions')
    return decisions


def _policy_loss_decision_indices(trajectory, policy_sampler, scope: str) -> list[int]:
    """Select score-function rows without changing rollout or actor replay."""
    if scope != POLICY_DECISION_ALL:
        raise ValueError(f'unsupported policy decision scope: {scope!r}')
    decisions = _policy_decisions(trajectory, policy_sampler)
    return list(range(len(decisions)))


def _a_action_speak_probabilities(
    actor_logps: torch.Tensor,
    decisions,
    *,
    speak_token_id: int,
    silent_token_id: int,
    frame_count: int,
) -> torch.Tensor:
    """Recover differentiable ``p(SPEAK)`` for every frame action row."""
    if actor_logps.ndim != 1 or actor_logps.numel() != len(decisions):
        raise ValueError('actor log-probabilities must align one-to-one with decisions')
    if frame_count < 1:
        raise ValueError('frame_count must be positive')
    if not bool(torch.isfinite(actor_logps.detach()).all().item()):
        raise ValueError('actor log-probabilities must be finite')
    if bool((actor_logps.detach() > 1e-6).any().item()):
        raise ValueError('selected-token log-probabilities must not be positive')

    action_rows = [
        (index, decision) for index, decision in enumerate(decisions)
        if decision.route == 'A' and not decision.history
    ]
    action_frames = [int(decision.frame_index) for _, decision in action_rows]
    if action_frames != list(range(frame_count)):
        raise ValueError('expected one A action per frame: '
                         f'action_frames={action_frames} frame_count={frame_count}')

    speak_probabilities = []
    for index, decision in action_rows:
        selected_logp = actor_logps[index]
        if int(decision.token_id) == int(speak_token_id):
            speak_probability = selected_logp.exp()
        elif int(decision.token_id) == int(silent_token_id):
            speak_probability = -torch.expm1(selected_logp)
        else:
            raise ValueError('A action must be <speak> or <silent>: '
                             f'token_id={decision.token_id}')
        speak_probabilities.append(speak_probability)
    return torch.stack(speak_probabilities)


def expected_silent_rate_loss(
    actor_logps: torch.Tensor,
    decisions,
    *,
    speak_token_id: int,
    silent_token_id: int,
    target_speak_count: int,
    frame_count: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Calibrate expected A-branch activity without rewarding false positives.

    At an A action row the grammar admits only ``<speak>`` and ``<silent>``.
    Therefore the selected-token log-probability is sufficient to recover
    ``p(SPEAK)`` regardless of which action was sampled. Matching its sum to
    the number of GT events supplies a gradient on an all-silent GRPO plateau;
    hard-window F1 remains responsible for *when* those events should occur.
    """
    if not 1 <= target_speak_count <= frame_count:
        raise ValueError('target_speak_count must be in [1, frame_count]')
    speak_probabilities = _a_action_speak_probabilities(
        actor_logps,
        decisions,
        speak_token_id=speak_token_id,
        silent_token_id=silent_token_id,
        frame_count=frame_count,
    )

    expected_speak_count = speak_probabilities.sum()
    target = expected_speak_count.new_tensor(float(target_speak_count))
    loss = (expected_speak_count - target).abs() / (expected_speak_count + target)
    diagnostics = {
        'expected_speak_count': float(expected_speak_count.detach().item()),
        'target_speak_count': float(target_speak_count),
        'expected_silent_rate': float(1.0 - expected_speak_count.detach().item() / frame_count),
        'target_silent_rate': float(1.0 - target_speak_count / frame_count),
    }
    return loss, diagnostics


def action_timing_auxiliary_loss(
    actor_logps: torch.Tensor,
    decisions,
    *,
    speak_token_id: int,
    silent_token_id: int,
    frame_times: list[float],
    gt_windows: list[dict],
    epsilon: float = 1e-6,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Dense A-action timing loss with disjoint event responsibilities.

    Frames outside all GT windows are trained toward SILENT. Frames inside
    overlapping windows are assigned to exactly one nearest-center event
    region, so one action is never counted twice. Equal-center windows are
    grouped as simultaneous targets and require that many actions.
    """
    if not 0 < epsilon < 0.5:
        raise ValueError('epsilon must be in (0, 0.5)')
    frame_count = len(frame_times)
    probabilities = _a_action_speak_probabilities(
        actor_logps.float(),
        decisions,
        speak_token_id=speak_token_id,
        silent_token_id=silent_token_id,
        frame_count=frame_count,
    )
    try:
        times = [float(value) for value in frame_times]
    except (TypeError, ValueError) as exc:
        raise ValueError('frame_times must be finite numbers') from exc
    if any(not np.isfinite(value) for value in times):
        raise ValueError('frame_times must be finite numbers')
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError('frame_times must be strictly increasing')
    if not gt_windows:
        raise ValueError('action timing loss requires at least one GT window')

    grouped: dict[float, dict] = {}
    for index, window in enumerate(gt_windows):
        if not isinstance(window, dict):
            raise ValueError(f'gt_windows[{index}] must be an object')
        try:
            start = float(window['start_s'])
            end = float(window['end_s'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'gt_windows[{index}] must be finite') from exc
        if not np.isfinite(start) or not np.isfinite(end) or start > end:
            raise ValueError(f'gt_windows[{index}] must be a finite ordered interval')
        center = 0.5 * (start + end)
        if center not in grouped:
            grouped[center] = {
                'start': start,
                'end': end,
                'target_count': 1,
            }
        else:
            grouped[center]['start'] = min(grouped[center]['start'], start)
            grouped[center]['end'] = max(grouped[center]['end'], end)
            grouped[center]['target_count'] += 1

    regions = [
        {
            **region,
            'center': center,
            'frame_indices': [],
        }
        for center, region in sorted(grouped.items())
    ]
    outside_indices = []
    for frame_index, timestamp in enumerate(times):
        candidates = [
            (abs(timestamp - region['center']), region_index)
            for region_index, region in enumerate(regions)
            if region['start'] <= timestamp <= region['end']
        ]
        if not candidates:
            outside_indices.append(frame_index)
            continue
        _, region_index = min(candidates)
        regions[region_index]['frame_indices'].append(frame_index)

    for index, region in enumerate(regions):
        if len(region['frame_indices']) < region['target_count']:
            raise ValueError('GT event region has fewer discrete action frames '
                             f'than targets: region={index} frames='
                             f'{len(region["frame_indices"])} targets='
                             f'{region["target_count"]}')

    zero = probabilities.sum() * 0.0
    if outside_indices:
        outside = probabilities.index_select(
            0,
            torch.tensor(outside_indices, device=probabilities.device),
        )
        safe_silent_probability = (1.0 - outside) * (1.0 - epsilon) + epsilon
        outside_loss = -safe_silent_probability.log().mean()
        expected_outside = outside.sum()
    else:
        outside_loss = zero
        expected_outside = zero

    target_total = sum(region['target_count'] for region in regions)
    coverage_terms = []
    count_terms = []
    hit_probabilities = []
    expected_inside = zero
    for region in regions:
        region_probabilities = probabilities.index_select(
            0,
            torch.tensor(region['frame_indices'], device=probabilities.device),
        )
        target_count = int(region['target_count'])
        expected_count = region_probabilities.sum()
        expected_inside = expected_inside + expected_count

        # Capped Poisson-binomial DP: state[target_count] is the probability
        # of observing at least target_count actions in this region.
        distribution = [probabilities.new_ones(())]
        distribution.extend(probabilities.new_zeros(()) for _ in range(target_count))
        for probability in region_probabilities:
            updated = [distribution[0] * (1.0 - probability)]
            updated.extend(
                distribution[count] * (1.0 - probability)
                + distribution[count - 1] * probability
                for count in range(1, target_count)
            )
            updated.append(
                distribution[target_count]
                + distribution[target_count - 1] * probability
            )
            distribution = updated
        hit_probability = distribution[target_count]
        hit_probabilities.append(hit_probability)
        safe_hit_probability = hit_probability * (1.0 - epsilon) + epsilon
        coverage_terms.append(-target_count * safe_hit_probability.log())
        count_terms.append((expected_count - target_count).square() / target_count)

    coverage_loss = torch.stack(coverage_terms).sum() / target_total
    count_loss = torch.stack(count_terms).sum() / target_total
    diagnostics = {
        'action_expected_inside_speak_count': float(expected_inside.detach().item()),
        'action_expected_outside_speak_count': float(expected_outside.detach().item()),
        'action_mean_event_hit_probability': float(torch.stack(hit_probabilities).mean().detach().item()),
        'action_event_region_count': float(len(regions)),
        'action_target_event_count': float(target_total),
        'action_outside_frame_count': float(len(outside_indices)),
    }
    return {
        'action_outside_loss': outside_loss,
        'action_coverage_loss': coverage_loss,
        'action_count_loss': count_loss,
    }, diagnostics


def _target_speak_count(trajectory) -> int:
    """Return the number of timing events represented by one training clip."""
    gt = trajectory.case.gt
    if 'gt_windows' in gt:
        events = gt['gt_windows']
    elif 'gt_event_times' in gt:
        events = gt['gt_event_times']
    else:
        raise ValueError('silent-rate calibration requires GT timing events')
    count = len(events)
    if not 1 <= count <= len(trajectory.frames):
        raise ValueError('GT timing-event count must be in [1, frame_count]: '
                         f'events={count} frames={len(trajectory.frames)}')
    return count


def _actor_lm_head_positions(
    encoded_ids,
    all_decisions,
    *,
    assistant_role_id: int,
) -> list[int]:
    """Include every rollout row sharing an LM-head call with a decision."""
    meta = build_step_mask_meta(
        np.asarray(encoded_ids, dtype=np.int64),
        assistant_role_id=assistant_role_id,
        include_branch_source=True,
    )
    required = {int(decision.position) - 1 for decision in all_decisions}
    expanded = []
    for b_span, a_span in zip(meta.b_spans, meta.a_spans):
        b_start, b_end = map(int, b_span)
        a_start, a_end = map(int, a_span)
        max_length = max(b_end - b_start, a_end - a_start)
        for offset in range(max_length):
            group = []
            if b_start + offset < b_end:
                group.append(b_start + offset)
            if a_start + offset < a_end:
                group.append(a_start + offset)
            if required.intersection(group):
                expanded.extend(group)

    if len(expanded) != len(set(expanded)):
        raise RuntimeError('actor LM-head execution positions are duplicated')
    missing = sorted(required - set(expanded))
    if missing:
        raise RuntimeError('actor LM-head execution groups miss decision positions: ' f'{missing[:16]}')
    return expanded


def _actor_position_contract(
    collated,
    independent_position_rows,
    rows,
    offsets,
    *,
    assistant_role_id: int,
):
    """Prove packed actor positions equal independent-query positions.

    Swift keeps a physical text-position row for padding-free boundaries and
    passes three Qwen3.5 MRoPE rows (temporal/height/width) to the model. The
    Qwen collator computes each row independently before concatenation. This
    check compares all four axes at every physical token, both before and
    after applying the dual-branch overlap used by the model pre-forward hook.
    """
    total_length = sum(row[3] for row in rows)
    text_positions = collated.get('text_position_ids')
    model_positions = collated.get('position_ids')
    if (text_positions is None or tuple(text_positions.shape) != (1, total_length) or model_positions is None
            or tuple(model_positions.shape) != (3, 1, total_length)):
        raise RuntimeError('actor requires Swift text + Qwen3.5 T/H/W position axes: '
                           f'text={None if text_positions is None else tuple(text_positions.shape)} '
                           f'mrope={None if model_positions is None else tuple(model_positions.shape)} '
                           f'expected=(1,{total_length})+(3,1,{total_length})')
    packed_four_axes = torch.cat([text_positions.unsqueeze(0), model_positions], dim=0)

    sample_spans = []
    global_b_spans = []
    global_a_spans = []
    independent_positions = []
    local_metas = []
    branch_pair_count = 0
    for row_index, ((encoded, _, _, sequence_length), start) in enumerate(zip(rows, offsets)):
        end = start + sequence_length
        sample_spans.append((start, end))
        row_positions = independent_position_rows[row_index]
        if (not isinstance(row_positions, torch.Tensor) or tuple(row_positions.shape) != (4, 1, sequence_length)):
            raise RuntimeError(f'Qwen collator row {row_index} lacks four position axes: '
                               f'{None if row_positions is None else tuple(row_positions.shape)} '
                               f'!= (4,1,{sequence_length})')
        if not torch.equal(packed_four_axes[:, :, start:end], row_positions):
            mismatch = torch.nonzero(
                packed_four_axes[:, :, start:end] != row_positions,
                as_tuple=False,
            )[0].detach().cpu().tolist()
            raise RuntimeError('packing changed a query position coordinate before overlap: '
                               f'row={row_index} axis/batch/local={mismatch}')

        meta = build_step_mask_meta(
            encoded['input_ids'],
            assistant_role_id=assistant_role_id,
            include_branch_source=True,
        )
        if not meta.is_dual_branch or len(meta.b_spans) != len(meta.a_spans):
            raise RuntimeError(f'actor row {row_index} lacks paired dual spans for MRoPE')
        local_metas.append(meta)
        branch_pair_count += len(meta.b_spans)
        global_b_spans.extend((start + span_start, start + span_end) for span_start, span_end in meta.b_spans)
        global_a_spans.extend((start + span_start, start + span_end) for span_start, span_end in meta.a_spans)
        independent_positions.append(
            apply_overlap_to_position_ids_tensor(
                row_positions,
                meta.b_spans,
                meta.a_spans,
                sample_spans=[(0, sequence_length)],
            ))

    packed_after_overlap = apply_overlap_to_position_ids_tensor(
        packed_four_axes,
        global_b_spans,
        global_a_spans,
        sample_spans=sample_spans,
    )
    branch_starts_equal = True
    for row_index, (start, independent, meta) in enumerate(zip(offsets, independent_positions, local_metas)):
        sequence_length = rows[row_index][3]
        end = start + sequence_length
        if not torch.equal(packed_after_overlap[:, :, start:end], independent):
            mismatch = torch.nonzero(
                packed_after_overlap[:, :, start:end] != independent,
                as_tuple=False,
            )[0].detach().cpu().tolist()
            raise RuntimeError('packing changed a query position coordinate after overlap: '
                               f'row={row_index} axis/batch/local={mismatch}')
        for b_span, a_span in zip(meta.b_spans, meta.a_spans):
            b_start = int(b_span[0])
            a_start = int(a_span[0])
            current_equal = torch.equal(
                independent[1:, 0, b_start],
                independent[1:, 0, a_start],
            )
            branch_starts_equal = branch_starts_equal and current_equal
            if not current_equal:
                raise RuntimeError('dual branches do not share all three MRoPE start axes: '
                                   f'row={row_index} B={b_span} A={a_span}')

    return {
        'axis_names': ['text_position', 'mrope_t', 'mrope_h', 'mrope_w'],
        'all_tokens_pre_overlap_exact': True,
        'all_tokens_post_overlap_exact': True,
        'text_resets_equal_query_offsets': True,
        'branch_starts_equal_all_mrope_axes': branch_starts_equal,
        'query_offsets': [int(value) for value in offsets],
        'query_lengths': [int(row[3]) for row in rows],
        'branch_pair_count': branch_pair_count,
    }


def _alignment_summary(actor_logps, rollout_logps, decisions, all_decisions, *, packed_offset: int):
    if (actor_logps.ndim != 1 or actor_logps.shape != rollout_logps.shape or actor_logps.numel() != len(decisions)
            or not decisions):
        raise RuntimeError('actor/rollout policy-logp shape differs from decisions: '
                           f'actor={tuple(actor_logps.shape)} '
                           f'rollout={tuple(rollout_logps.shape)} '
                           f'decisions={len(decisions)}')
    nonfinite = [
        name for name, values in (
            ('actor', actor_logps),
            ('rollout', rollout_logps),
        ) if not bool(torch.isfinite(values.detach()).all().item())
    ]
    if nonfinite:
        raise RuntimeError(f'non-finite policy log-probability: {nonfinite}')
    absolute_drift = (actor_logps.detach() - rollout_logps.detach()).abs()
    signed_log_ratio = actor_logps.detach() - rollout_logps.detach()
    importance_ratio = signed_log_ratio.exp()
    # Copy once instead of synchronizing CUDA separately for every diagnostic
    # scalar below.  The Python reductions intentionally preserve the previous
    # diagnostic values and do not participate in the loss graph.
    absolute_drift_values = absolute_drift.cpu().tolist()
    decision_kind = ['action' if not decision.history else 'content' for decision in decisions]
    worst_index = int(absolute_drift.argmax().item())
    worst_decision = decisions[worst_index]

    alignment = {
        'sampled_decisions': len(all_decisions),
        'policy_decisions': len(decisions),
        'structural_decisions_excluded': len(all_decisions) - len(decisions),
        'worst_logp_drift': {
            'route': worst_decision.route,
            'kind': decision_kind[worst_index],
            'frame_index': worst_decision.frame_index,
            'token_index': worst_decision.token_index,
            'token_id': worst_decision.token_id,
            'history': list(worst_decision.history),
            'target_position': worst_decision.position,
            'prediction_position': worst_decision.position - 1,
            'packed_target_position': (packed_offset + worst_decision.position),
            'packed_prediction_position': (packed_offset + worst_decision.position - 1),
            'rollout_logp': float(rollout_logps[worst_index].item()),
            'actor_logp': float(actor_logps[worst_index].detach().item()),
            'signed_log_ratio': float(signed_log_ratio[worst_index].item()),
            'absolute_logp_drift': float(absolute_drift[worst_index].item()),
            'importance_ratio': float(importance_ratio[worst_index].item()),
        },
        'policy_decisions_by_route':
        {route: sum(decision.route == route for decision in decisions)
         for route in ('B', 'A')},
        'max_logp_drift_by_route': {
            route: max(
                (absolute_drift_values[index]
                 for index, decision in enumerate(decisions) if decision.route == route),
                default=0.0,
            )
            for route in ('B', 'A')
        },
        'max_logp_drift_by_kind': {
            kind: max(
                (absolute_drift_values[index] for index, current in enumerate(decision_kind) if current == kind),
                default=0.0,
            )
            for kind in ('action', 'content')
        },
        'mean_logp_drift': float(absolute_drift.mean().item()),
        'mean_logp_drift_by_kind': {
            kind:
            (sum(absolute_drift_values[index] for index, current in enumerate(decision_kind) if current == kind)
             / max(sum(current == kind for current in decision_kind), 1))
            for kind in ('action', 'content')
        },
        'same_weight_importance_ratio_min': float(importance_ratio.min().item()),
        'same_weight_importance_ratio_max': float(importance_ratio.max().item()),
        'target_position_min': min(decision.position for decision in decisions),
        'target_position_max': max(decision.position for decision in decisions),
        'prediction_position_min': min(decision.position - 1 for decision in decisions),
        'prediction_position_max': max(decision.position - 1 for decision in decisions),
        'packed_target_position_min': packed_offset + min(decision.position for decision in decisions),
        'packed_target_position_max': packed_offset + max(decision.position for decision in decisions),
        'packed_prediction_position_min': packed_offset + min(decision.position - 1 for decision in decisions),
        'packed_prediction_position_max': packed_offset + max(decision.position - 1 for decision in decisions),
    }
    return alignment


def actor_logps_for_trajectories(model,
                                 template,
                                 trajectories,
                                 policy_sampler,
                                 *,
                                 assistant_role_id: int,
                                 device,
                                 validate_position_contract=True,
                                 policy_logits_observer=None,
                                 reuse_frozen_visual_features=False,
                                 defer_actor_lm_head=False):
    """Replay several independent queries in one padding-free train forward."""
    trajectories = list(trajectories)
    if not trajectories:
        raise ValueError('actor replay batch is empty')
    rows = []
    offsets = []
    packed_prediction_positions = []
    packed_lm_head_positions = []
    policy_logit_rows = []
    offset = 0
    for trajectory in trajectories:
        encoded, all_decisions = encode_training_trajectory(template, trajectory, assistant_role_id)
        decisions = _policy_decisions(trajectory, policy_sampler)
        sequence_length = len(encoded['input_ids'])
        lm_head_positions = _actor_lm_head_positions(
            encoded['input_ids'],
            all_decisions,
            assistant_role_id=assistant_role_id,
        )
        position_to_lm_row = {position: row for row, position in enumerate(lm_head_positions)}
        lm_row_offset = len(packed_lm_head_positions)
        policy_logit_rows.append([lm_row_offset + position_to_lm_row[decision.position - 1] for decision in decisions])
        packed_lm_head_positions.extend(offset + position for position in lm_head_positions)
        offsets.append(offset)
        packed_prediction_positions.extend(offset + decision.position - 1 for decision in decisions)
        rows.append((encoded, all_decisions, decisions, sequence_length))
        offset += sequence_length

    if packed_prediction_positions != sorted(packed_prediction_positions):
        raise RuntimeError('packed actor prediction positions are not ordered')
    if len(set(packed_prediction_positions)) != len(packed_prediction_positions):
        raise RuntimeError('packed actor prediction positions are duplicated')
    if len(set(packed_lm_head_positions)) != len(packed_lm_head_positions):
        raise RuntimeError('packed actor LM-head positions are duplicated')

    independent_position_rows = None
    if validate_position_contract:
        get_position_ids = getattr(template, '_get_position_ids', None)
        if get_position_ids is None:
            raise RuntimeError('Qwen actor template lacks _get_position_ids')
        independent_position_rows = []
        for encoded, _, _, _ in rows:
            position_inputs = dict(encoded)
            position_inputs['input_ids'] = torch.tensor(encoded['input_ids'], dtype=torch.long).unsqueeze(0)
            independent_position_rows.append(get_position_ids(position_inputs).detach().clone())
    collator_rows = [dict(encoded) for encoded, _, _, _ in rows]
    collated = template.data_collator(collator_rows)
    input_ids = collated['input_ids']
    if input_ids.ndim != 2 or input_ids.shape != (1, offset):
        raise RuntimeError('padding-free actor batch has wrong physical shape: '
                           f'{tuple(input_ids.shape)} != (1, {offset})')
    for row_index, ((encoded, _, _, sequence_length), start) in enumerate(zip(rows, offsets)):
        expected = torch.as_tensor(encoded['input_ids'], dtype=input_ids.dtype, device=input_ids.device)
        actual = input_ids[0, start:start + sequence_length]
        if not torch.equal(actual, expected):
            raise RuntimeError(f'packed actor row {row_index} changed physical token ids')

    # The physical tensor is [1, sum(L_i)], but every query must remain an
    # independent logical sample. These boundaries are consumed by all three
    # stateful paths: full attention, linear recurrence, and causal conv. Do
    # not merely trust that the collator emitted *some* cu_seqlens: prove that
    # they are exactly the offsets of the trajectories we just packed.
    expected_boundaries = offsets + [offset]
    for key in ('cu_seq_lens_q', 'cu_seq_lens_k'):
        boundaries = collated.get(key)
        if boundaries is None:
            raise RuntimeError(f'padding-free actor batch lacks authoritative {key}')
        actual_boundaries = [int(value) for value in boundaries.detach().cpu().flatten()]
        if actual_boundaries != expected_boundaries:
            raise RuntimeError(f'packed actor {key} does not match query boundaries: '
                               f'actual={actual_boundaries} '
                               f'expected={expected_boundaries}')

    text_position_ids = collated.get('text_position_ids')
    if (text_position_ids is None or text_position_ids.ndim != 2 or tuple(text_position_ids.shape) != (1, offset)):
        raise RuntimeError('packed actor lacks the expected Qwen text-position axis: '
                           f'{None if text_position_ids is None else tuple(text_position_ids.shape)} '
                           f'!= (1, {offset})')
    position_resets = torch.nonzero(text_position_ids[0] == 0, as_tuple=True)[0].detach().cpu().tolist()
    if position_resets != offsets:
        raise RuntimeError('packed actor MRoPE sample resets do not match query offsets: '
                           f'actual={position_resets} expected={offsets}')

    if validate_position_contract:
        position_contract = _actor_position_contract(
            collated,
            independent_position_rows,
            rows,
            offsets,
            assistant_role_id=assistant_role_id,
        )
    else:
        position_contract = {'validated': False}

    prediction_positions = torch.tensor(packed_lm_head_positions, dtype=torch.long)
    model_inputs = _prepare_model_inputs(
        collated,
        prediction_positions,
        device,
        reuse_frozen_visual_features=reuse_frozen_visual_features,
    )
    visual_features = []
    expected_image_grids = []
    if reuse_frozen_visual_features:
        for trajectory in trajectories:
            if (len(trajectory.image_embeds) != len(trajectory.images)
                    or len(trajectory.image_grids) != len(trajectory.images)):
                raise RuntimeError('trajectory lacks one frozen visual feature/grid per image: '
                                   f'case={trajectory.case.row_index} '
                                   f'features={len(trajectory.image_embeds)} '
                                   f'grids={len(trajectory.image_grids)} '
                                   f'images={len(trajectory.images)}')
            visual_features.extend(trajectory.image_embeds)
            expected_image_grids.extend(trajectory.image_grids)
        image_grids = model_inputs.get('image_grid_thw')
        if image_grids is None or int(image_grids.shape[0]) != len(visual_features):
            raise RuntimeError('actor image grids do not match frozen visual features: '
                               f'grids={None if image_grids is None else int(image_grids.shape[0])} '
                               f'features={len(visual_features)}')
        actual_image_grids = [
            tuple(int(value) for value in row)
            for row in image_grids.detach().cpu().tolist()
        ]
        if actual_image_grids != expected_image_grids:
            raise RuntimeError('actor image-grid order differs from frozen visual features')
    visual_context = (use_actor_precomputed_visual_features(visual_features)
                      if visual_features else nullcontext())
    total_lm_head_rows = len(packed_lm_head_positions)
    total_decisions = len(packed_prediction_positions)
    if defer_actor_lm_head:
        lm_head = model.get_output_embeddings()
        if lm_head is None:
            raise RuntimeError('actor model has no LM head to defer')
        with visual_context, defer_actor_lm_head_projection():
            outputs = model(**model_inputs)
        selected_hidden_states = outputs.logits
        hidden_size = int(getattr(lm_head, 'in_features', 0))
        if (selected_hidden_states is None
                or selected_hidden_states.shape[:2] != (1, total_lm_head_rows)
                or int(selected_hidden_states.shape[-1]) != hidden_size):
            raise RuntimeError('deferred actor LM head returned the wrong hidden-state shape: '
                               f'hidden={None if selected_hidden_states is None else tuple(selected_hidden_states.shape)} '
                               f'expected=(1, {total_lm_head_rows}, {hidden_size})')
        actor_logp_rows = _checkpointed_actor_logps_from_hidden(
            selected_hidden_states,
            policy_logit_rows,
            [row[2] for row in rows],
            policy_sampler,
            lm_head,
            policy_logits_observer=policy_logits_observer,
        )
    else:
        with visual_context:
            outputs = model(**model_inputs)
        logits = outputs.logits
        if logits is None or logits.shape[:2] != (1, total_lm_head_rows):
            raise RuntimeError('execution-aligned lm-head output has the wrong row count: '
                               f'logits={None if logits is None else tuple(logits.shape)} '
                               f'rows={total_lm_head_rows}')
        actor_logp_rows = []
        for row_index, (logit_rows, (_, _, decisions, _)) in enumerate(zip(policy_logit_rows, rows)):
            indices = torch.tensor(logit_rows, device=logits.device, dtype=torch.long)
            row_logits = logits[0].index_select(0, indices)
            if policy_logits_observer is not None:
                policy_logits_observer(row_index, decisions, row_logits.detach())
            actor_logp_rows.append(_actor_logps_from_full_logits(row_logits, decisions, policy_sampler))

    result = []
    for (encoded, all_decisions, decisions,
         sequence_length), start, actor_logps in zip(rows, offsets, actor_logp_rows):
        rollout_logps = torch.tensor(
            [decision.rollout_logp for decision in decisions],
            device=actor_logps.device,
            dtype=actor_logps.dtype,
        )
        alignment = _alignment_summary(
            actor_logps,
            rollout_logps,
            decisions,
            all_decisions,
            packed_offset=start,
        )
        alignment['packed_sequence_length'] = offset
        alignment['actor_replay_batch_size'] = len(trajectories)
        alignment['packed_sample_bounds'] = [int(start), int(start + sequence_length)]
        alignment['position_contract'] = {
            **position_contract,
            'sample_index': len(result),
            'sample_bounds': [int(start), int(start + sequence_length)],
        }
        result.append((
            actor_logps,
            rollout_logps,
            sequence_length,
            alignment,
        ))
    if sum(len(rows) for rows in policy_logit_rows) != total_decisions:
        raise RuntimeError('policy LM-head row mapping changed the decision count')
    return result


def _checkpointed_actor_logps_from_hidden(
    selected_hidden_states,
    policy_logit_rows,
    decision_rows,
    policy_sampler,
    lm_head,
    *,
    policy_logits_observer=None,
    checkpoint_decisions=ACTOR_LM_HEAD_CHECKPOINT_DECISIONS,
):
    """Project and score sampled rows while retaining only hidden states.

    The outer Qwen forward has already selected every LM-head row required by
    production's synchronous dual-branch execution.  Recompute the exact
    grouped LM-head projection and policy transformation during backward so
    vocabulary-sized logits and top-p intermediates do not survive the actor
    forward.  This remains valid if ``lm_head`` becomes trainable later.
    """
    if selected_hidden_states.ndim != 3 or int(selected_hidden_states.shape[0]) != 1:
        raise ValueError('actor hidden states must have shape [1, rows, hidden]')
    if len(policy_logit_rows) != len(decision_rows) or not decision_rows:
        raise ValueError('actor policy row/decision groups are inconsistent')

    if checkpoint_decisions < 1:
        raise ValueError('actor LM-head checkpoint size must be positive')

    device = selected_hidden_states.device
    index_rows = []
    decision_counts = []
    for logit_rows, decisions in zip(policy_logit_rows, decision_rows):
        if len(logit_rows) != len(decisions) or not decisions:
            raise ValueError('actor policy logits must contain one row per decision')
        index_rows.append(torch.tensor(logit_rows, device=device, dtype=torch.long))
        decision_counts.append(len(decisions))

    projection_forward, execution_groups = actor_lm_head_execution_plan(
        lm_head,
        selected_hidden_states,
    )
    decision_by_lm_row = {}
    output_index = 0
    for indices, decisions in zip(index_rows, decision_rows):
        for lm_row, decision in zip(indices.detach().cpu().tolist(), decisions):
            lm_row = int(lm_row)
            if lm_row in decision_by_lm_row:
                raise RuntimeError(f'actor policy LM-head row {lm_row} is duplicated')
            decision_by_lm_row[lm_row] = (output_index, decision)
            output_index += 1

    if policy_logits_observer is not None:
        # Numerical gates need the exact full rows once, but production has no
        # observer and never materializes this vocabulary-sized diagnostic.
        diagnostic_logits = {}
        with torch.no_grad():
            for group in execution_groups:
                requested = [lm_row for lm_row in group if lm_row in decision_by_lm_row]
                if not requested:
                    continue
                indices = torch.tensor(group, device=device, dtype=torch.long)
                group_logits = projection_forward(
                    selected_hidden_states[0].index_select(0, indices).unsqueeze(1)
                )[:, 0, :]
                for group_offset, lm_row in enumerate(group):
                    if lm_row in decision_by_lm_row:
                        diagnostic_logits[lm_row] = group_logits[group_offset]
            for row_index, (indices, decisions) in enumerate(zip(index_rows, decision_rows)):
                policy_logits_observer(
                    row_index,
                    decisions,
                    torch.stack([
                        diagnostic_logits[int(lm_row)]
                        for lm_row in indices.detach().cpu().tolist()
                    ]).detach(),
                )

    chunks = []
    chunk_groups = []
    chunk_decisions = 0
    for group in execution_groups:
        group_entries = [
            (group_offset, *decision_by_lm_row[lm_row])
            for group_offset, lm_row in enumerate(group)
            if lm_row in decision_by_lm_row
        ]
        if not group_entries:
            continue
        if chunk_groups and chunk_decisions + len(group_entries) > checkpoint_decisions:
            chunks.append(chunk_groups)
            chunk_groups = []
            chunk_decisions = 0
        chunk_groups.append((group, group_entries))
        chunk_decisions += len(group_entries)
    if chunk_groups:
        chunks.append(chunk_groups)

    def score_chunk(chunk_hidden, group_specs):
        scores = []
        cursor = 0
        if actor_fused_lm_head_input_backward_enabled(lm_head):
            # Preserve every one/two-row production projection in forward.
            # The custom backward avoids retaining the frozen projection graph
            # while preserving those same physical GEMM shapes exactly. Blocks
            # are capped at eight rows to bound recomputed vocabulary logits.
            fused_blocks = []
            block = []
            block_rows = 0
            for group_spec in group_specs:
                group_size = len(group_spec[0])
                if block and block_rows + group_size > 8:
                    fused_blocks.append(block)
                    block = []
                    block_rows = 0
                block.append(group_spec)
                block_rows += group_size
            if block:
                fused_blocks.append(block)

            for fused_block in fused_blocks:
                group_sizes = tuple(len(group) for group, _ in fused_block)
                fused_rows = sum(group_sizes)
                block_logits = actor_fused_lm_head_projection(
                    chunk_hidden[cursor:cursor + fused_rows],
                    lm_head,
                    projection_forward,
                    group_sizes,
                )
                block_cursor = 0
                for (group, entries), group_size in zip(
                        fused_block, group_sizes):
                    group_logits = block_logits[
                        block_cursor:block_cursor + group_size
                    ]
                    block_cursor += group_size
                    for group_offset, _, decision in entries:
                        scores.append(policy_sampler.selected_log_prob(
                            group_logits[group_offset],
                            token_id=decision.token_id,
                            stream=decision.route,
                            history=decision.history,
                        ))
                cursor += fused_rows
        else:
            for group, entries in group_specs:
                group_size = len(group)
                group_logits = projection_forward(
                    chunk_hidden[cursor:cursor + group_size].unsqueeze(1)
                )[:, 0, :]
                cursor += group_size
                for group_offset, _, decision in entries:
                    scores.append(policy_sampler.selected_log_prob(
                        group_logits[group_offset],
                        token_id=decision.token_id,
                        stream=decision.route,
                        history=decision.history,
                    ))
        if cursor != int(chunk_hidden.shape[0]) or not scores:
            raise RuntimeError('actor LM-head checkpoint chunk has inconsistent rows')
        return torch.stack(scores)

    needs_grad = torch.is_grad_enabled() and (
        selected_hidden_states.requires_grad
        or any(parameter.requires_grad for parameter in lm_head.parameters())
    )
    scored_chunks = []
    scored_order = []
    for group_specs in chunks:
        hidden_rows = [lm_row for group, _ in group_specs for lm_row in group]
        hidden_indices = torch.tensor(hidden_rows, device=device, dtype=torch.long)
        chunk_hidden = selected_hidden_states[0].index_select(0, hidden_indices)
        chunk_order = [output for _, entries in group_specs for _, output, _ in entries]

        def checkpointed_score(hidden, specs=group_specs):
            return score_chunk(hidden, specs)

        scores = (activation_checkpoint(
            checkpointed_score,
            chunk_hidden,
            use_reentrant=False,
            preserve_rng_state=False,
        )
                  if needs_grad else checkpointed_score(chunk_hidden))
        scored_chunks.append(scores)
        scored_order.extend(chunk_order)

    if len(scored_order) != output_index or sorted(scored_order) != list(range(output_index)):
        raise RuntimeError('actor LM-head checkpoint groups do not cover policy rows once')
    flat_logps = torch.cat(scored_chunks)
    restore_order = torch.argsort(torch.tensor(scored_order, device=device, dtype=torch.long))
    flat_logps = flat_logps.index_select(0, restore_order)
    return list(torch.split(flat_logps, decision_counts))


def _actor_logps_from_full_logits(row_logits, decisions, policy_sampler):
    """Unoptimized full-logit reference used by exact A/B gates."""
    if row_logits.ndim != 2 or row_logits.shape[0] != len(decisions):
        raise ValueError('actor policy logits must contain one row per decision')
    if not decisions:
        raise ValueError('actor replay contains no sampled policy decisions')

    def selected_logps(logit_rows):
        return torch.stack([
            policy_sampler.selected_log_prob(
                logit_rows[index],
                token_id=decision.token_id,
                stream=decision.route,
                history=decision.history,
            ) for index, decision in enumerate(decisions)
        ])

    return selected_logps(row_logits)


def actor_logps_for_trajectory(model, template, trajectory, policy_sampler, *, assistant_role_id: int, device):
    """Backward-compatible one-query wrapper around packed actor replay."""
    return actor_logps_for_trajectories(
        model,
        template,
        [trajectory],
        policy_sampler,
        assistant_role_id=assistant_role_id,
        device=device,
    )[0]


def _actor_microbatch_plan(trajectories, *, max_trajectories: int, packed_token_budget: int):
    """Return ordered ``(start, stop, token_count)`` actor microbatches."""
    if max_trajectories < 1 or packed_token_budget < 0:
        raise ValueError('invalid actor microbatch limits')
    plan = []
    start = 0
    while start < len(trajectories):
        stop = start
        packed_tokens = 0
        while stop < len(trajectories) and stop - start < max_trajectories:
            token_count = len(trajectories[stop].physical_ids)
            if token_count < 1:
                raise ValueError('actor trajectory must contain physical tokens')
            if (stop > start and packed_token_budget > 0
                    and packed_tokens + token_count > packed_token_budget):
                break
            packed_tokens += token_count
            stop += 1
        plan.append((start, stop, packed_tokens))
        start = stop
    return plan


def _select_trainable_parameters(model):
    parameters = []
    names = []
    for name, parameter in model.named_parameters():
        trainable = any(part in name for part in TRAINABLE_PARTS)
        parameter.requires_grad_(trainable)
        if trainable:
            names.append(name)
            parameters.append(parameter)
    if not parameters:
        raise RuntimeError('no dual-LoRA parameters found in checkpoint model')
    leaked = [name for name, parameter in model.named_parameters() if parameter.requires_grad and name not in names]
    if leaked:
        raise RuntimeError(f'unexpected trainable parameters: {leaked[:5]}')
    return names, parameters


def _load_named_safetensors(checkpoint: str, names, parameters):
    """Load only ``names`` from a single-file or sharded checkpoint."""
    from safetensors import safe_open

    checkpoint = Path(checkpoint).expanduser().resolve()
    index_path = checkpoint / 'model.safetensors.index.json'
    single_path = checkpoint / 'model.safetensors'
    if index_path.is_file():
        with index_path.open(encoding='utf-8') as handle:
            weight_map = json.load(handle).get('weight_map')
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError(f'invalid safetensors index: {index_path}')
    elif single_path.is_file():
        with safe_open(single_path, framework='pt', device='cpu') as handle:
            weight_map = {name: single_path.name for name in handle.keys()}
    else:
        raise RuntimeError(f'checkpoint has no safetensors weights: {checkpoint}')

    missing = [name for name in names if name not in weight_map]
    if missing:
        raise RuntimeError('SFT reference checkpoint lacks trainable tensors: '
                           f'{missing[:5]} (missing={len(missing)})')
    tensors = [None] * len(names)
    indices_by_shard = {}
    for index, name in enumerate(names):
        indices_by_shard.setdefault(weight_map[name], []).append(index)
    for shard_name, indices in sorted(indices_by_shard.items()):
        shard_path = checkpoint / shard_name
        if not shard_path.is_file():
            raise RuntimeError(f'missing SFT reference shard: {shard_path}')
        with safe_open(shard_path, framework='pt', device='cpu') as handle:
            for index in indices:
                name = names[index]
                source = handle.get_tensor(name)
                parameter = parameters[index]
                if tuple(source.shape) != tuple(parameter.shape):
                    raise RuntimeError('SFT reference tensor shape mismatch for '
                                       f'{name}: {tuple(source.shape)} != '
                                       f'{tuple(parameter.shape)}')
                tensors[index] = source.to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
    if any(tensor is None for tensor in tensors):
        raise RuntimeError('SFT reference tensor loading was incomplete')
    return tensors


class FrozenTrainableReference:
    """Frozen SFT dual-LoRA snapshot with fail-closed in-place activation.

    Only the trainable dual-LoRA tensors differ between actor and reference;
    all base/vision weights are frozen. Reusing the model therefore avoids a
    second 4B allocation while preserving an exact SFT-reference forward.
    """

    def __init__(self, names, parameters, tensors):
        if not (len(names) == len(parameters) == len(tensors)):
            raise ValueError('reference name/parameter/tensor count mismatch')
        self.names = tuple(names)
        self.parameters = tuple(parameters)
        self.tensors = tuple(tensor.detach() for tensor in tensors)
        self._live = tuple(torch.empty_like(parameter) for parameter in parameters)
        self.sha256 = _exact_named_tensor_digest(self.names, self.tensors)

    @classmethod
    def create(cls, names, parameters, *, source_checkpoint: Optional[str]):
        tensors = (
            [parameter.detach().clone() for parameter in parameters]
            if source_checkpoint is None
            else _load_named_safetensors(source_checkpoint, names, parameters)
        )
        return cls(names, parameters, tensors)

    @contextmanager
    def activated(self):
        with torch.no_grad():
            for live, parameter in zip(self._live, self.parameters):
                live.copy_(parameter)
            for reference, parameter in zip(self.tensors, self.parameters):
                parameter.copy_(reference)
        try:
            yield
        finally:
            with torch.no_grad():
                for live, parameter in zip(self._live, self.parameters):
                    parameter.copy_(live)


@torch.no_grad()
def _reference_logps_for_actor_plan(
    reference,
    model,
    template,
    trajectories,
    plan,
    policy_sampler,
    *,
    assistant_role_id: int,
    device,
    reuse_frozen_visual_features: bool,
    defer_actor_lm_head: bool,
):
    """Replay trajectories under the frozen SFT LoRA snapshot in plan order."""
    rows = []
    try:
        with reference.activated():
            for start, stop, _ in plan:
                replay = actor_logps_for_trajectories(
                    model,
                    template,
                    trajectories[start:stop],
                    policy_sampler,
                    assistant_role_id=assistant_role_id,
                    device=device,
                    reuse_frozen_visual_features=reuse_frozen_visual_features,
                    defer_actor_lm_head=defer_actor_lm_head,
                )
                rows.extend(row[0].detach().clone() for row in replay)
                del replay
                clear_dual_ctx()
                clear_step_mask_context()
    finally:
        clear_dual_ctx()
        clear_step_mask_context()
    if len(rows) != len(trajectories):
        raise RuntimeError('SFT reference replay returned the wrong trajectory count')
    return rows


def _disable_dropout(model) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0


def _disable_vision_gradient_checkpointing(model) -> int:
    """Frozen vision blocks never need actor activation checkpointing."""
    visual = getattr(getattr(model, 'model', None), 'visual', None)
    blocks = list(getattr(visual, 'blocks', ()))
    if visual is not None and hasattr(visual, 'gradient_checkpointing'):
        visual.gradient_checkpointing = False
    for block in blocks:
        block.gradient_checkpointing = False
    if any(bool(getattr(block, 'gradient_checkpointing', False)) for block in blocks):
        raise RuntimeError('frozen vision gradient checkpointing could not be disabled')
    return len(blocks)


def _configure_gradient_checkpointing(model, args) -> dict:
    """Apply the requested Qwen3.5 decoder checkpoint policy fail-closed."""
    text_model = getattr(getattr(model, 'model', None), 'language_model', None)
    layers = list(getattr(text_model, 'layers', ()))
    if not layers:
        raise RuntimeError('gradient checkpointing requires Qwen3.5 decoder layers')
    if not args.gradient_checkpointing:
        for layer in layers:
            layer.gradient_checkpointing = False
        vision_blocks = _disable_vision_gradient_checkpointing(model)
        return {
            'policy': 'disabled',
            'checkpointed_decoder_layers': 0,
            'checkpointed_decoder_layer_indices': [],
            'decoder_layers': len(layers),
            'checkpointed_vision_blocks': 0,
            'vision_blocks': vision_blocks,
        }

    # Actor replay disables every dropout module before the first forward, so
    # saving/restoring CUDA RNG state around dozens of deterministic
    # checkpoints is pure synchronization overhead.
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            'use_reentrant': False,
            'preserve_rng_state': False,
        })
    vision_blocks = _disable_vision_gradient_checkpointing(model)
    linear_indices = [
        index for index, layer in enumerate(layers)
        if getattr(layer, 'layer_type', None) == 'linear_attention'
    ]
    if len(linear_indices) == len(layers):
        raise RuntimeError('selective checkpoint policies require full-attention decoder layers')
    if args.gradient_checkpointing_policy == 'all':
        expected = list(range(len(layers)))
    else:
        stride = {
            'linear_only': 1,
            'linear_alternating': 2,
            'linear_sparse': 3,
        }[args.gradient_checkpointing_policy]
        expected = linear_indices[::stride]
    expected_set = set(expected)
    for index, layer in enumerate(layers):
        layer.gradient_checkpointing = index in expected_set
    checkpointed = [index for index, layer in enumerate(layers) if layer.gradient_checkpointing]
    if checkpointed != expected:
        raise RuntimeError('decoder gradient checkpoint policy was not applied exactly: '
                           f'actual={checkpointed} expected={expected}')
    return {
        'policy': args.gradient_checkpointing_policy,
        'checkpointed_decoder_layers': len(checkpointed),
        'checkpointed_decoder_layer_indices': checkpointed,
        'decoder_layers': len(layers),
        'checkpointed_vision_blocks': 0,
        'vision_blocks': vision_blocks,
    }


def _gradient_family_norms(model) -> dict[str, float]:
    squared = {part: torch.zeros((), device=next(model.parameters()).device) for part in TRAINABLE_PARTS}
    leaked = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            if parameter.grad is not None:
                leaked.append(name)
            continue
        family = next((part for part in TRAINABLE_PARTS if part in name), None)
        if family is None:
            leaked.append(name)
            continue
        if parameter.grad is not None:
            squared[family] = (squared[family] + parameter.grad.detach().float().square().sum())
    if leaked:
        raise RuntimeError(f'gradient escaped dual-LoRA trainable set: {leaked[:5]}')
    return {family: float(value.sqrt().item()) for family, value in squared.items()}


def _assert_frozen_parameters_have_no_grad(model) -> None:
    """Reject gradient leakage without reducing every trainable tensor."""
    leaked = [
        name for name, parameter in model.named_parameters()
        if not parameter.requires_grad and parameter.grad is not None
    ]
    if leaked:
        raise RuntimeError(f'gradient escaped dual-LoRA trainable set: {leaked[:5]}')


def _synchronize_fp32_gradients(
    parameters,
    *,
    world_size: int,
    local_gradients=None,
    bucket_mib: int = 64,
):
    """Sum globally normalized gradients without wrapping the model in DDP."""
    parameters = list(parameters)
    reduction_dtype = torch.float32
    if local_gradients is None:
        local_gradients = [
            torch.zeros_like(parameter, dtype=torch.float32) if parameter.grad is None else parameter.grad.detach().to(
                dtype=torch.float32) for parameter in parameters
        ]
    else:
        local_gradients = list(local_gradients)
        if len(local_gradients) != len(parameters):
            raise ValueError('local gradient/parameter count mismatch')
        reduction_dtype = (
            torch.float64 if any(gradient.dtype == torch.float64 for gradient in local_gradients) else torch.float32)
        normalized = []
        for index, (parameter, gradient) in enumerate(zip(parameters, local_gradients)):
            if tuple(gradient.shape) != tuple(parameter.shape):
                raise ValueError(f'local gradient shape mismatch at {index}: '
                                 f'{tuple(gradient.shape)} != {tuple(parameter.shape)}')
            normalized.append(gradient.detach().to(device=parameter.device, dtype=reduction_dtype))
        local_gradients = normalized
    if world_size == 1:
        return [gradient.to(dtype=torch.float32) for gradient in local_gradients]
    bucket_numel = max(1, bucket_mib * 1024 * 1024 // torch.empty((), dtype=reduction_dtype).element_size())
    synchronized = [None] * len(parameters)
    pending = []
    pending_numel = 0

    def flush_bucket():
        nonlocal pending, pending_numel
        if not pending:
            return
        flat = torch.cat([gradient.reshape(-1) for _, gradient, _ in pending])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        cursor = 0
        for parameter_index, gradient, shape in pending:
            next_cursor = cursor + gradient.numel()
            synchronized[parameter_index] = (flat[cursor:next_cursor].view(shape).to(dtype=torch.float32))
            cursor = next_cursor
        if cursor != flat.numel():
            raise RuntimeError('gradient reduction bucket split is inconsistent')
        pending = []
        pending_numel = 0

    for index, (parameter, gradient) in enumerate(zip(parameters, local_gradients)):
        if pending and pending_numel + gradient.numel() > bucket_numel:
            flush_bucket()
        pending.append((index, gradient, tuple(parameter.shape)))
        pending_numel += gradient.numel()
    flush_bucket()
    if any(gradient is None for gradient in synchronized):
        raise RuntimeError('not every trainable gradient was synchronized')
    return synchronized


def _clip_fp32_gradients(gradients, max_norm: float) -> float:
    squared = torch.stack([gradient.square().sum() for gradient in gradients]).sum()
    norm = float(squared.sqrt().item())
    if not np.isfinite(norm):
        raise RuntimeError(f'non-finite gradient norm: {norm}')
    scale = min(1.0, float(max_norm) / (norm + 1e-6))
    if scale < 1.0:
        for gradient in gradients:
            gradient.mul_(scale)
    return norm


def _apply_optimizer_step(optimizer, gradients, *, global_has_advantage: bool) -> bool:
    """Apply AdamW only when the caller has a non-zero learning signal."""
    if not global_has_advantage:
        optimizer.last_model_update_norm = 0.0
        if any(torch.count_nonzero(gradient).item() for gradient in gradients):
            raise RuntimeError('zero global update signal produced a non-zero gradient')
        # Calling AdamW with explicit zero gradients is not a no-op: it
        # advances moment/bias-correction state and can apply weight decay.
        # Leave both optimizer state and model/master weights untouched.
        return False
    optimizer.step(master_gradients=gradients)
    return True


def _global_advantage_flag(advantages, *, device: str, distributed: bool) -> bool:
    """Report learning-signal presence independently of update execution."""
    flag = torch.tensor(
        int(any(abs(advantage) > 0 for advantage in advantages)),
        device=device,
        dtype=torch.int32,
    )
    if distributed:
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def _gradient_family_norms_from_fp32(names: list[str], gradients) -> dict[str, float]:
    if len(names) != len(gradients):
        raise ValueError('gradient/name count mismatch')
    squared = {family: torch.zeros((), device=gradients[0].device) for family in TRAINABLE_PARTS}
    for name, gradient in zip(names, gradients):
        family = next((part for part in TRAINABLE_PARTS if part in name), None)
        if family is None:
            raise RuntimeError(f'gradient escaped dual-LoRA trainable set: {name}')
        squared[family] = squared[family] + gradient.square().sum()
    return {family: float(value.sqrt().item()) for family, value in squared.items()}


def _exact_named_tensor_digest(names, tensors) -> str:
    if len(names) != len(tensors):
        raise ValueError('tensor/name count mismatch')
    digest = hashlib.sha256()
    for name, tensor in zip(names, tensors):
        value = tensor.detach().contiguous()
        digest.update(name.encode('utf-8'))
        digest.update(str(tuple(value.shape)).encode('ascii'))
        digest.update(str(value.dtype).encode('ascii'))
        digest.update(value.view(torch.uint8).cpu().numpy().tobytes(order='C'))
    return digest.hexdigest()


def _triton_autotuner_snapshot() -> list[dict]:
    """Return the exact FLA Triton configs selected in this process."""
    try:
        from triton.runtime.autotuner import Autotuner
    except ImportError:
        return []

    rows = []
    seen = set()
    for module_name, module in sorted(sys.modules.items()):
        if not module_name.startswith('fla.') or module is None:
            continue
        for attribute_name, value in sorted(vars(module).items()):
            tuner = (value if isinstance(value, Autotuner) else getattr(value, 'fn', None))
            if not isinstance(tuner, Autotuner):
                continue
            identity = id(tuner)
            if identity in seen or not tuner.cache:
                continue
            seen.add(identity)
            selections = []
            for cache_key, config in sorted(tuner.cache.items(), key=lambda item: repr(item[0])):
                selections.append({
                    'cache_key': repr(cache_key),
                    'kwargs': {key: repr(item)
                               for key, item in sorted(config.kwargs.items())},
                    'num_warps': config.num_warps,
                    'num_stages': config.num_stages,
                    'num_ctas': config.num_ctas,
                })
            rows.append({
                'kernel': f'{module_name}.{attribute_name}',
                'selections': selections,
            })
    return rows


def _exact_trainable_digest(names, parameters) -> str:
    return _exact_named_tensor_digest(names, parameters)


def _assert_exact_trainable_rank_equality(names, parameters, *, world_size: int) -> str:
    local_digest = _exact_trainable_digest(names, parameters)
    if world_size == 1:
        return local_digest
    digests = [None] * world_size
    dist.all_gather_object(digests, local_digest)
    if len(set(digests)) != 1:
        raise RuntimeError(f'post-update trainable tensors differ across ranks: {digests}')
    return local_digest


def _checkpoint_due(args, completed_step: int) -> bool:
    if args.dry_run_rollout:
        return False
    periodic = (args.save_every > 0 and completed_step % args.save_every == 0)
    final = completed_step == args.max_steps and args.save_final
    return periodic or final


def _trainable_digest_due(args, completed_step: int) -> bool:
    exact_check = (not args.dry_run_rollout and args.distributed_exact_check_every > 0
                   and completed_step % args.distributed_exact_check_every == 0)
    return _checkpoint_due(args, completed_step) or exact_check


def _copy_first_existing(candidates, destination: Path) -> bool:
    for candidate in candidates:
        candidate = Path(candidate)
        if candidate.is_file():
            shutil.copy2(candidate, destination)
            return True
    return False


def _capture_rng_state() -> dict:
    return {
        'python_rng_state': random.getstate(),
        'numpy_rng_state': np.random.get_state(),
        'torch_rng_state': torch.get_rng_state(),
        'cuda_rng_state': (torch.cuda.get_rng_state() if torch.cuda.is_available() else None),
    }


def save_checkpoint(sm,
                    optimizer,
                    *,
                    output_dir: Path,
                    step: int,
                    args,
                    trainable_names: list[str],
                    rng_states_by_rank=None) -> Path:
    final = output_dir / f'checkpoint-{step}'
    temporary = output_dir / f'.checkpoint-{step}.incomplete'
    if final.exists() or temporary.exists():
        raise FileExistsError(f'refusing to overwrite checkpoint path: {final}')
    temporary.mkdir(parents=True)
    sm.model.save_pretrained(
        temporary,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    # StreamModel wraps the HF image processor with an in-memory cache.  That
    # wrapper is an inference optimization, not a serializable checkpoint
    # component; temporarily expose the original processor while saving.
    image_processor = getattr(sm.processor, 'image_processor', None)
    inner_image_processor = getattr(image_processor, '_inner', None)
    if inner_image_processor is not None:
        sm.processor.image_processor = inner_image_processor
    try:
        sm.processor.save_pretrained(temporary)
    finally:
        if inner_image_processor is not None:
            sm.processor.image_processor = image_processor
    source_model = Path(args.resume_from_checkpoint or args.model)
    # ``processor.save_pretrained`` in the currently pinned transformers
    # version writes processor_config.json but omits the Qwen image
    # preprocessor descriptor.  Without it ``get_processor`` silently falls
    # back to a bare tokenizer and a saved RL checkpoint cannot be resumed or
    # inferred.  Preserve the authoritative multimodal descriptor verbatim.
    if not _copy_first_existing(
        [source_model / 'preprocessor_config.json', source_model.parent / 'preprocessor_config.json'],
            temporary / 'preprocessor_config.json',
    ):
        raise FileNotFoundError(f'source checkpoint lacks preprocessor_config.json: ' f'{source_model}')
    copied_contract = _copy_first_existing(
        [source_model / 'dual_infer_contract.json', source_model.parent / 'dual_infer_contract.json'],
        temporary / 'dual_infer_contract.json',
    )
    contract_path = temporary / 'dual_infer_contract.json'
    if not copied_contract:
        contract = {
            key: value
            for key, value in sm.dual_runtime_spec.items() if key not in {'manifest_path', 'step_mask_window_source'}
        }
    else:
        contract = json.loads(contract_path.read_text(encoding='utf-8'))
    contract.update({
        'contract_version': max(int(contract.get('contract_version', 1)), 2),
        'rl_algorithm': 'dual_stream_grpo',
        'rl_step': step,
        'step_mask_window': args.window,
        'rl_trainable_parameters_regex': '|'.join(TRAINABLE_PARTS),
    })
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    if not _copy_first_existing(
        [source_model / 'args.json', source_model.parent / 'args.json'],
            temporary / 'args.json',
    ):
        (temporary / 'args.json').write_text(
            json.dumps({
                'trainable_parameters_regex': '|'.join(TRAINABLE_PARTS),
                'rlhf_type': 'grpo',
            }, indent=2) + '\n',
            encoding='utf-8')
    if rng_states_by_rank is None:
        rng_states_by_rank = [_capture_rng_state()]
    if len(rng_states_by_rank) != int(args.world_size):
        raise ValueError('RNG rank-state count does not match distributed world size')
    state = {
        'step': step,
        'optimizer': optimizer.state_dict(),
        'distributed_world_size': int(args.world_size),
        'rng_states_by_rank': rng_states_by_rank,
        'trainable_names': trainable_names,
    }
    torch.save(state, temporary / 'rl_trainer_state.pt')
    (temporary / 'rl_train_args.json').write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    # Keep the generic progress file consumed by the production checkpoint
    # verifier, but identify this as an RL step rather than pretending it is
    # the source SFT trainer state.  The optimizer/RNG payload remains in the
    # dedicated binary file above.
    (temporary / 'trainer_state.json').write_text(
        json.dumps(
            {
                'global_step': step,
                'rl_global_step': step,
                'training_phase': 'dual_stream_grpo',
                'distributed_world_size': int(args.world_size),
                'source_model': str(args.model),
            },
            ensure_ascii=False,
            indent=2) + '\n',
        encoding='utf-8',
    )
    (temporary / 'checkpoint_complete.json').write_text(
        json.dumps({
            'step': step,
            'complete': True,
            'distributed_world_size': int(args.world_size),
        }, indent=2) + '\n',
        encoding='utf-8',
    )
    temporary.rename(final)
    return final


def _validate_resume_argument_contract(path: Optional[str], args) -> None:
    """Fail closed if resume would change policy/update semantics."""
    if path is None:
        return
    args_path = Path(path) / 'rl_train_args.json'
    if not args_path.is_file():
        raise FileNotFoundError(f'resume checkpoint lacks {args_path.name}: {path}')
    saved = json.loads(args_path.read_text(encoding='utf-8'))
    if not isinstance(saved, dict):
        raise RuntimeError(f'resume {args_path.name} is not a JSON object: {path}')
    # Contract-12 checkpoints predate the CLI knob but used this exact
    # hard-coded value.
    saved.setdefault('max_sequence_length', 32768)
    saved.setdefault('actor_rank_rebalance', ACTOR_RANK_REBALANCE_NONE)
    saved.setdefault('actor_packed_token_budget', 0)
    if 'actor_fused_frozen_linear_backward' not in saved:
        # Compatibility with checkpoints written while the experimental
        # switch had an overly broad name. That implementation already fused
        # frozen Linear derivatives only in accepted production runs.
        saved['actor_fused_frozen_linear_backward'] = saved.get(
            'actor_fused_frozen_backward', False)
    saved.setdefault('actor_fused_trainable_linear_input_backward', False)
    saved.setdefault('actor_segmented_rmsnorm_backward', False)
    saved.setdefault('actor_fused_gated_rmsnorm_backward', False)
    saved.setdefault('actor_fused_lm_head_input_backward', False)
    if 'actor_backward_numerics_version' not in saved:
        if saved['actor_fused_lm_head_input_backward']:
            raise RuntimeError(
                'resume checkpoint predates exact physical-group LM-head '
                'backward; restart this run from its SFT checkpoint'
            )
        saved['actor_backward_numerics_version'] = (
            ACTOR_BACKWARD_NUMERICS_VERSION
        )
    # Checkpoints written before the timing-ablation CLI used the complete
    # positive pool and hard-window F1. These defaults are an exact semantic
    # migration, not a relaxed resume contract.
    saved.setdefault('timing_data_variant', FULL_POSITIVE_DATA)
    saved.setdefault('timing_epoch_steps', 0)
    saved.setdefault('timing_reward_mode', HARD_WINDOW_F1)
    saved.setdefault('silent_rate_loss_weight', 0.0)
    saved.setdefault('action_outside_loss_weight', 0.0)
    saved.setdefault('action_coverage_loss_weight', 0.0)
    saved.setdefault('action_count_loss_weight', 0.0)
    if 'sft_reference_model' not in saved:
        if float(saved.get('kl_coef', 0.0)) != 0.0:
            raise RuntimeError('KL-enabled resume checkpoint lacks its SFT reference model contract')
        saved['sft_reference_model'] = None
    saved.setdefault('policy_decision_scope', POLICY_DECISION_ALL)
    saved.setdefault(
        'actor_flex_kernel_options',
        INFERENCE_ALIGNED_FLEX_KERNEL_OPTIONS,
    )
    contract_keys = list(RESUME_CONTRACT_KEYS)
    contract_keys.extend(('timing_train_sha256', 'timing_manifest_sha256'))
    missing = [key for key in contract_keys if key not in saved]
    if missing:
        raise RuntimeError('resume checkpoint lacks numerical-contract arguments: ' f'{missing}')
    mismatches = {
        key: {
            'saved': saved[key],
            'current': getattr(args, key)
        }
        for key in contract_keys if saved[key] != getattr(args, key)
    }
    if mismatches:
        raise RuntimeError('resume numerical contract differs from checkpoint: '
                           + json.dumps(mismatches, ensure_ascii=False, sort_keys=True))


def _validate_sft_reference_digest(path: Optional[str], actual: Optional[str]) -> None:
    """Reject a KL resume if its frozen SFT reference changed in place."""
    if path is None or actual is None:
        return
    args_path = Path(path) / 'rl_train_args.json'
    saved = json.loads(args_path.read_text(encoding='utf-8'))
    expected = saved.get('sft_reference_trainable_sha256')
    if expected is None:
        raise RuntimeError('KL-enabled resume checkpoint lacks its SFT reference digest')
    if expected != actual:
        raise RuntimeError('SFT reference trainable tensors changed since checkpoint: '
                           f'{expected} != {actual}')


def _restore_training_state(path: Optional[str],
                            optimizer,
                            *,
                            trainable_names,
                            rank: int = 0,
                            world_size: int = 1) -> int:
    if path is None:
        return 0
    checkpoint_path = Path(path)
    completion_path = checkpoint_path / 'checkpoint_complete.json'
    if not completion_path.is_file():
        raise FileNotFoundError('resume checkpoint lacks checkpoint_complete.json: ' f'{checkpoint_path}')
    completion = json.loads(completion_path.read_text(encoding='utf-8'))
    if completion.get('complete') is not True:
        raise RuntimeError(f'resume checkpoint is not marked complete: {checkpoint_path}')

    state_path = checkpoint_path / 'rl_trainer_state.pt'
    if not state_path.is_file():
        raise FileNotFoundError(f'resume checkpoint lacks {state_path.name}: {path}')
    state = torch.load(state_path, map_location='cpu', weights_only=False)
    state_step = int(state['step'])
    if int(completion.get('step', -1)) != state_step:
        raise RuntimeError('resume completion/state step mismatch: ' f'{completion.get("step")} != {state_step}')
    marker_world_size = int(completion.get('distributed_world_size', world_size))
    if marker_world_size != world_size:
        raise RuntimeError('resume completion WORLD_SIZE differs from current run: '
                           f'{marker_world_size} != {world_size}')
    saved_names = state.get('trainable_names')
    expected_names = list(trainable_names)
    if saved_names != expected_names:
        saved_names = saved_names or []
        mismatch = next((index for index, pair in enumerate(zip(saved_names, expected_names)) if pair[0] != pair[1]),
                        min(len(saved_names), len(expected_names)))
        raise RuntimeError('resume dual-LoRA parameter order differs from checkpoint: '
                           f'first_mismatch={mismatch} '
                           f'saved_count={len(saved_names)} '
                           f'current_count={len(expected_names)}')
    optimizer.load_state_dict(state['optimizer'])
    rng_states = state.get('rng_states_by_rank')
    if rng_states is None:
        if world_size != 1:
            raise RuntimeError('legacy single-rank checkpoint cannot resume with ' f'WORLD_SIZE={world_size}')
        rng_state = state
    else:
        saved_world_size = int(state.get('distributed_world_size', len(rng_states)))
        if saved_world_size != world_size or len(rng_states) != world_size:
            raise RuntimeError('resume WORLD_SIZE differs from checkpoint: ' f'{world_size} != {saved_world_size}')
        rng_state = rng_states[rank]
    random.setstate(rng_state['python_rng_state'])
    np.random.set_state(rng_state['numpy_rng_state'])
    torch.set_rng_state(rng_state['torch_rng_state'])
    if torch.cuda.is_available():
        if rng_state.get('cuda_rng_state') is not None:
            torch.cuda.set_rng_state(rng_state['cuda_rng_state'])
        elif rng_state.get('cuda_rng_state_all'):
            torch.cuda.set_rng_state_all(rng_state['cuda_rng_state_all'])
    return state_step


def _mean(values):
    return sum(values) / max(len(values), 1)


def _synchronized_time(device: str) -> float:
    """Return a wall-clock timestamp after prior CUDA work is complete."""
    torch.cuda.synchronize(torch.device(device))
    return time.perf_counter()


@contextmanager
def _cuda_profile_region(enabled: bool, events: list) -> None:
    """Record one default-stream CUDA region without adding synchronization."""
    if not enabled:
        yield
        return
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    try:
        yield
    finally:
        end.record()
        events.append((start, end))


def _cuda_profile_elapsed_s(events: list) -> float:
    """Sum completed CUDA-event intervals in seconds."""
    return sum(start.elapsed_time(end) for start, end in events) / 1000.0


def _aggregate_rank_records(records: list[dict]) -> dict:
    if not records:
        raise ValueError('cannot aggregate an empty rank record list')
    records = sorted(records, key=lambda row: row['rank'])
    world_size = int(records[0]['world_size'])
    uniform_fields = (
        'step',
        'world_size',
        'deterministic_algorithms',
        'query_batch_size',
        'local_query_batch_size',
        'group_size',
        'dry_run_rollout',
    )
    for key in uniform_fields:
        values = {row.get(key) for row in records}
        if None in values:
            raise RuntimeError(f'rank record lacks {key}')
        if len(values) != 1:
            raise RuntimeError(f'ranks disagree on {key}: {values}')
    if len(records) != world_size:
        raise RuntimeError(f'got {len(records)} rank records for WORLD_SIZE={world_size}')
    if [row['rank'] for row in records] != list(range(world_size)):
        raise RuntimeError('rank records are not a complete ordered rank set')
    query_batch_size = int(records[0]['query_batch_size'])
    local_query_batch_size = int(records[0]['local_query_batch_size'])
    group_size = int(records[0]['group_size'])
    if query_batch_size != local_query_batch_size * world_size:
        raise RuntimeError('global/local query_bs does not match WORLD_SIZE: '
                           f'{query_batch_size} != '
                           f'{local_query_batch_size} * {world_size}')
    prompt_slots_by_rank = []
    for row in records:
        slots = row.get('global_prompt_slots')
        if slots is None:
            start, stop = row['global_prompt_slot_range']
            slots = list(range(start, stop))
        slots = [int(slot) for slot in slots]
        if len(slots) != local_query_batch_size or len(set(slots)) != len(slots):
            raise RuntimeError('rank prompt slots differ from local query_bs')
        prompt_slots_by_rank.append(slots)
    all_prompt_slots = [slot for slots in prompt_slots_by_rank for slot in slots]
    if sorted(all_prompt_slots) != list(range(query_batch_size)):
        raise RuntimeError('rank query shards do not cover contiguous global query_bs')

    case_by_prompt_slot = {}
    rollout_fingerprints = []
    for row, prompt_slots in zip(records, prompt_slots_by_rank):
        if len(row['selected_case_ids']) != local_query_batch_size:
            raise RuntimeError('rank selected-case count differs from local query_bs')
        expected_trajectories = local_query_batch_size * group_size
        if int(row['trajectory_count']) != expected_trajectories:
            raise RuntimeError('rank trajectory count differs from local query_bs * G')
        fingerprints = row.get('rollout_fingerprints')
        if (not isinstance(fingerprints, list) or len(fingerprints) != expected_trajectories):
            raise RuntimeError('rank rollout-fingerprint count differs from trajectories')
        expected_cases = dict(zip(prompt_slots, row['selected_case_ids']))
        expected_keys = {(prompt_slot, group_index)
                         for prompt_slot in prompt_slots for group_index in range(group_size)}
        actual_keys = {(int(item['prompt_slot']), int(item['group_index'])) for item in fingerprints}
        if actual_keys != expected_keys:
            raise RuntimeError('rank rollout fingerprints do not cover query_bs * G')
        for item in fingerprints:
            prompt_slot = int(item['prompt_slot'])
            if int(item['case_id']) != int(expected_cases[prompt_slot]):
                raise RuntimeError('rollout fingerprint case/prompt mapping differs')
            for digest_key in (
                    'physical_ids_sha256',
                    'rollout_policy_trace_sha256',
            ):
                digest = item.get(digest_key)
                if not isinstance(digest, str) or len(digest) != 64:
                    raise RuntimeError(f'rollout fingerprint lacks {digest_key}')
        rollout_fingerprints.extend(fingerprints)
        for prompt_slot, case_id in expected_cases.items():
            if prompt_slot in case_by_prompt_slot:
                raise RuntimeError('duplicate global prompt slot across rank records')
            case_by_prompt_slot[prompt_slot] = int(case_id)
    selected_case_ids = [case_by_prompt_slot[slot] for slot in range(query_batch_size)]
    if len(set(selected_case_ids)) != query_batch_size:
        raise RuntimeError('global query_bs does not contain distinct case ids')

    actor_rebalance_rows = [row.get('actor_rank_rebalance') for row in records]
    if any(not isinstance(info, dict) for info in actor_rebalance_rows):
        raise RuntimeError('rank record lacks actor rank-rebalance contract')
    actor_rebalance_modes = {info.get('mode') for info in actor_rebalance_rows}
    actor_rebalance_effective = {info.get('effective') for info in actor_rebalance_rows}
    if len(actor_rebalance_modes) != 1 or next(iter(actor_rebalance_modes)) not in ACTOR_RANK_REBALANCE_CHOICES:
        raise RuntimeError(f'ranks disagree on actor rank-rebalance mode: {actor_rebalance_modes}')
    if len(actor_rebalance_effective) != 1 or not actor_rebalance_effective <= {True, False}:
        raise RuntimeError('ranks disagree on whether actor rebalance was effective')
    actor_rebalance_mode = next(iter(actor_rebalance_modes))
    actor_rebalance_is_effective = next(iter(actor_rebalance_effective))
    actor_rebalance_skip_reasons = {info.get('skip_reason') for info in actor_rebalance_rows}
    if len(actor_rebalance_skip_reasons) != 1:
        raise RuntimeError('ranks disagree on the actor rebalance skip reason')
    actor_rebalance_skip_reason = next(iter(actor_rebalance_skip_reasons))
    current_load_rows = [info.get('current_sampled_decision_loads') for info in actor_rebalance_rows]
    planned_load_rows = [info.get('planned_sampled_decision_loads') for info in actor_rebalance_rows]
    predicted_gain_rows = [info.get('predicted_max_load_relative_gain') for info in actor_rebalance_rows]
    if (any(value != current_load_rows[0] for value in current_load_rows[1:])
            or any(value != planned_load_rows[0] for value in planned_load_rows[1:])
            or any(value != predicted_gain_rows[0] for value in predicted_gain_rows[1:])):
        raise RuntimeError('ranks derived different actor rebalance planning diagnostics')
    current_actor_decision_loads = current_load_rows[0]
    planned_actor_decision_loads = planned_load_rows[0]
    predicted_actor_gain = predicted_gain_rows[0]
    actor_prompt_slots_by_rank = []
    for rank, (info, rollout_slots) in enumerate(zip(actor_rebalance_rows, prompt_slots_by_rank)):
        if [int(slot) for slot in info.get('rollout_prompt_slots', [])] != rollout_slots:
            raise RuntimeError(f'rank {rank} actor record changed rollout ownership')
        actor_slots = [int(slot) for slot in info.get('actor_prompt_slots', [])]
        actor_case_ids = [int(case_id) for case_id in info.get('actor_case_ids', [])]
        if len(actor_slots) != local_query_batch_size or len(set(actor_slots)) != len(actor_slots):
            raise RuntimeError(f'rank {rank} actor slots differ from local query capacity')
        if len(actor_case_ids) != local_query_batch_size:
            raise RuntimeError(f'rank {rank} actor case count differs from local query capacity')
        expected_actor_case_ids = [case_by_prompt_slot[slot] for slot in actor_slots]
        if actor_case_ids != expected_actor_case_ids:
            raise RuntimeError(f'rank {rank} actor case/prompt mapping differs from rollout')
        expected_moved = len(set(rollout_slots) - set(actor_slots))
        expected_incoming = len(set(actor_slots) - set(rollout_slots))
        if (int(info.get('moved_out_query_count', -1)) != expected_moved
                or int(info.get('incoming_query_count', -1)) != expected_incoming):
            raise RuntimeError(f'rank {rank} actor movement counts are inconsistent')
        actor_prompt_slots_by_rank.append(actor_slots)
    if sorted(slot for slots in actor_prompt_slots_by_rank for slot in slots) != list(range(query_batch_size)):
        raise RuntimeError('actor rank plan does not cover global query_bs exactly once')

    actor_decision_loads = None
    if actor_rebalance_is_effective:
        if actor_rebalance_mode == ACTOR_RANK_REBALANCE_NONE or world_size == 1:
            raise RuntimeError('actor rebalance cannot be effective for none or WORLD_SIZE=1')
        load_rows = [info.get('sampled_decision_loads') for info in actor_rebalance_rows]
        if any(not isinstance(loads, list) or len(loads) != world_size for loads in load_rows):
            raise RuntimeError('actor rebalance lacks one planned load per rank')
        if any(loads != load_rows[0] for loads in load_rows[1:]):
            raise RuntimeError('ranks derived different actor sampled-decision loads')
        actor_decision_loads = [int(value) for value in load_rows[0]]
        if any(value < 1 for value in actor_decision_loads):
            raise RuntimeError('actor sampled-decision loads must be positive')
        for rank, info in enumerate(actor_rebalance_rows):
            if int(info.get('local_sampled_decision_load', -1)) != actor_decision_loads[rank]:
                raise RuntimeError(f'rank {rank} local actor load differs from the global plan')
    else:
        if actor_prompt_slots_by_rank != prompt_slots_by_rank:
            raise RuntimeError('disabled actor rebalance changed rank ownership')
        if any(info.get('sampled_decision_loads') is not None for info in actor_rebalance_rows):
            raise RuntimeError('disabled actor rebalance unexpectedly recorded a global plan')
    rollout_fingerprints.sort(key=lambda item: (int(item['prompt_slot']), int(item['group_index'])))
    digests = {
        row['post_update_trainable_sha256']
        for row in records if row['post_update_trainable_sha256'] is not None
    }
    if len(digests) > 1:
        raise RuntimeError(f'rank records contain different parameter digests: {digests}')
    gradient_digests = {
        row['pre_clip_gradient_sha256']
        for row in records if row['pre_clip_gradient_sha256'] is not None
    }
    if len(gradient_digests) > 1:
        raise RuntimeError('rank records contain different synchronized gradient digests: ' f'{gradient_digests}')
    update_flags = {(
        bool(row['global_has_advantage']),
        bool(row.get('global_has_silent_rate_signal', False)),
        bool(row.get('global_has_action_timing_signal', False)),
        bool(row.get('global_has_kl_signal', False)),
        bool(row.get('global_has_update_signal', row['global_has_advantage'])),
        bool(row['optimizer_step_applied']),
    )
                    for row in records}
    if len(update_flags) != 1:
        raise RuntimeError(f'ranks disagree on optimizer-step state: {update_flags}')
    actor_contracts = {(
        row.get('actor_linear_mode'),
        row.get('actor_replay_contract_version'),
    )
                       for row in records}
    if any(mode is None or version is None for mode, version in actor_contracts):
        raise RuntimeError('rank record lacks actor replay contract')
    if len(actor_contracts) != 1:
        raise RuntimeError(f'ranks disagree on actor replay contracts: {actor_contracts}')
    actor_linear_mode, actor_replay_contract_version = next(iter(actor_contracts))
    expected_actor_contract = ('inference_aligned', ACTOR_REPLAY_CONTRACT_VERSION)
    if (actor_linear_mode, actor_replay_contract_version) != (expected_actor_contract):
        raise RuntimeError('unsupported actor replay contract in rank records: '
                           f'{(actor_linear_mode, actor_replay_contract_version)}')
    rollout_query_modes = {row.get('rollout_query_batch_mode') for row in records}
    if None in rollout_query_modes:
        raise RuntimeError('rank record lacks rollout query-batch mode')
    if len(rollout_query_modes) != 1:
        raise RuntimeError('ranks disagree on rollout query-batch mode: ' f'{rollout_query_modes}')
    rollout_query_batch_mode = next(iter(rollout_query_modes))
    rank_shardings = {row.get('rank_sharding', 'contiguous_v0') for row in records}
    if len(rank_shardings) != 1:
        raise RuntimeError(f'ranks disagree on rank sharding: {rank_shardings}')
    rank_sharding = next(iter(rank_shardings))
    batch_schedules = {row.get('batch_schedule', 'random_v0') for row in records}
    if len(batch_schedules) != 1:
        raise RuntimeError(f'ranks disagree on batch schedule: {batch_schedules}')
    batch_schedule = next(iter(batch_schedules))
    visual_reuse_rows = [row.get('frozen_visual_feature_reuse', {}) for row in records]
    visual_reuse_enabled = {bool(row.get('enabled', False)) for row in visual_reuse_rows}
    if len(visual_reuse_enabled) != 1:
        raise RuntimeError('ranks disagree on frozen visual feature reuse')
    visual_reuse_summary = {
        'enabled': next(iter(visual_reuse_enabled)),
        'unique_images': sum(int(row.get('unique_images', 0)) for row in visual_reuse_rows),
        'trajectory_image_uses': sum(int(row.get('trajectory_image_uses', 0)) for row in visual_reuse_rows),
        'reused_image_uses': sum(int(row.get('reused_image_uses', 0)) for row in visual_reuse_rows),
        'actor_image_uses': sum(int(row.get('actor_image_uses', 0)) for row in visual_reuse_rows),
    }

    run_contract = records[0].get('run_contract')
    if run_contract is None or any(row.get('run_contract') != run_contract for row in records):
        raise RuntimeError('ranks disagree on the RL run contract')
    actor_batch_sizes = {
        int(row.get('actor_trajectory_microbatch_size',
                    run_contract.get('actor_trajectory_microbatch_size', 1)))
        for row in records
    }
    actor_token_budgets = {
        int(row.get('actor_packed_token_budget',
                    run_contract.get('actor_packed_token_budget', 0)))
        for row in records
    }
    if len(actor_batch_sizes) != 1 or len(actor_token_budgets) != 1:
        raise RuntimeError('ranks disagree on actor microbatch limits')
    actor_batch_size = next(iter(actor_batch_sizes))
    actor_token_budget = next(iter(actor_token_budgets))

    trajectory_count = sum(row['trajectory_count'] for row in records)
    group_diagnostics = [item for row in records for item in row.get('group_diagnostics', [])]
    if len(group_diagnostics) != query_batch_size:
        raise RuntimeError('rank group diagnostics do not cover global query_bs')
    if len({int(row['prompt_slot']) for row in group_diagnostics}) != query_batch_size:
        raise RuntimeError('rank group diagnostics contain duplicate prompt slots')
    group_diagnostics.sort(key=lambda row: int(row['prompt_slot']))
    group_sampling = _summarize_group_diagnostics(group_diagnostics)

    def weighted_mean(key):
        return sum(row[key] * row['trajectory_count'] for row in records) / trajectory_count

    def weighted_mean_optional(key, *, fallback=None, default=0.0):
        return sum(
            row.get(key, row[fallback] if fallback is not None else default)
            * row['trajectory_count'] for row in records
        ) / trajectory_count

    return {
        'step':
        records[0]['step'],
        'world_size':
        world_size,
        'deterministic_algorithms':
        records[0]['deterministic_algorithms'],
        'serialized_triton_autotune':
        records[0].get('serialized_triton_autotune', False),
        'triton_cache_dir':
        records[0].get('triton_cache_dir'),
        'triton_autotune_lock':
        records[0].get('triton_autotune_lock'),
        'triton_autotune_lock_scope':
        records[0].get('triton_autotune_lock_scope'),
        'global_has_advantage':
        records[0]['global_has_advantage'],
        'global_has_silent_rate_signal':
        records[0].get('global_has_silent_rate_signal', False),
        'global_has_action_timing_signal':
        records[0].get('global_has_action_timing_signal', False),
        'global_has_kl_signal':
        records[0].get('global_has_kl_signal', False),
        'global_has_update_signal':
        records[0].get('global_has_update_signal', records[0]['global_has_advantage']),
        'optimizer_step_applied':
        records[0]['optimizer_step_applied'],
        'actor_linear_mode':
        actor_linear_mode,
        'actor_rank_rebalance': {
            'mode': actor_rebalance_mode,
            'effective': actor_rebalance_is_effective,
            'skip_reason': actor_rebalance_skip_reason,
            'rollout_prompt_slots_by_rank': prompt_slots_by_rank,
            'actor_prompt_slots_by_rank': actor_prompt_slots_by_rank,
            'sampled_decision_loads': actor_decision_loads,
            'current_sampled_decision_loads': current_actor_decision_loads,
            'planned_sampled_decision_loads': planned_actor_decision_loads,
            'predicted_max_load_relative_gain': predicted_actor_gain,
            'minimum_relative_gain': actor_rebalance_rows[0].get('minimum_relative_gain'),
            'moved_query_count': sum(
                int(info['moved_out_query_count']) for info in actor_rebalance_rows),
            'rebuilt_image_count': sum(
                int(info['rebuilt_image_count']) for info in actor_rebalance_rows),
        },
        'actor_replay_contract_version':
        actor_replay_contract_version,
        'rollout_query_batch_mode':
        rollout_query_batch_mode,
        'rank_sharding':
        rank_sharding,
        'batch_schedule':
        batch_schedule,
        'frozen_visual_feature_reuse':
        visual_reuse_summary,
        'run_contract':
        run_contract,
        'query_batch_size':
        query_batch_size,
        'local_query_batch_size':
        records[0]['local_query_batch_size'],
        'group_size':
        records[0]['group_size'],
        'trajectory_count':
        trajectory_count,
        'selected_case_ids':
        selected_case_ids,
        'reward':
        weighted_mean('reward'),
        'reward_format':
        weighted_mean('reward_format'),
        'reward_timing':
        weighted_mean('reward_timing'),
        'rollout_fingerprints':
        rollout_fingerprints,
        'window_reward_metrics':
        _merge_timing_summaries([row.get('window_reward_metrics') for row in records]),
        'window_reward_metrics_by_source': {
            source:
            _merge_timing_summaries([row.get('window_reward_metrics_by_source', {}).get(source) for row in records])
            for source in sorted(
                {source
                 for row in records for source in row.get('window_reward_metrics_by_source', {})})
        },
        'loss':
        weighted_mean('loss'),
        'policy_loss':
        weighted_mean_optional('policy_loss', fallback='loss'),
        'kl':
        weighted_mean_optional('kl'),
        'kl_loss_weighted':
        weighted_mean_optional('kl_loss_weighted'),
        'silent_rate_loss':
        weighted_mean_optional('silent_rate_loss'),
        'silent_rate_loss_weighted':
        weighted_mean_optional('silent_rate_loss_weighted'),
        'action_outside_loss':
        weighted_mean_optional('action_outside_loss'),
        'action_outside_loss_weighted':
        weighted_mean_optional('action_outside_loss_weighted'),
        'action_coverage_loss':
        weighted_mean_optional('action_coverage_loss'),
        'action_coverage_loss_weighted':
        weighted_mean_optional('action_coverage_loss_weighted'),
        'action_count_loss':
        weighted_mean_optional('action_count_loss'),
        'action_count_loss_weighted':
        weighted_mean_optional('action_count_loss_weighted'),
        'action_timing_loss_weighted':
        weighted_mean_optional('action_timing_loss_weighted'),
        'action_expected_inside_speak_count':
        weighted_mean_optional('action_expected_inside_speak_count'),
        'action_expected_outside_speak_count':
        weighted_mean_optional('action_expected_outside_speak_count'),
        'action_mean_event_hit_probability':
        weighted_mean_optional('action_mean_event_hit_probability'),
        'expected_speak_count':
        weighted_mean_optional('expected_speak_count'),
        'target_speak_count':
        weighted_mean_optional('target_speak_count'),
        'expected_silent_rate':
        weighted_mean_optional('expected_silent_rate'),
        'target_silent_rate':
        weighted_mean_optional('target_silent_rate'),
        'policy_decision_scope':
        run_contract.get('policy_decision_scope', POLICY_DECISION_ALL),
        'loss_policy_decisions':
        sum(row.get('loss_policy_decisions', row.get('policy_decisions', 0))
            for row in records),
        'ratio':
        weighted_mean('ratio'),
        'clip_frac':
        weighted_mean('clip_frac'),
        'group_reward_std_nonzero':
        group_sampling['informative_group_count'] > 0,
        'group_sampling':
        group_sampling,
        'group_diagnostics':
        group_diagnostics,
        'max_logp_drift':
        max(row['max_logp_drift'] for row in records),
        'max_mean_logp_drift':
        max(row['max_mean_logp_drift'] for row in records),
        'max_logp_drift_by_route':
        {route: max(row['max_logp_drift_by_route'][route] for row in records)
         for route in ('B', 'A')},
        'max_logp_drift_by_kind':
        {kind: max(row['max_logp_drift_by_kind'][kind] for row in records)
         for kind in ('action', 'content')},
        'same_weight_importance_ratio_range': [
            min(row['same_weight_importance_ratio_range'][0] for row in records),
            max(row['same_weight_importance_ratio_range'][1] for row in records),
        ],
        'grad_norm':
        records[0]['grad_norm'],
        'model_parameter_update_norm':
        records[0].get('model_parameter_update_norm', 0.0),
        'grad_family_norms':
        records[0]['grad_family_norms'],
        'pre_clip_gradient_sha256': (next(iter(gradient_digests)) if gradient_digests else None),
        'post_update_trainable_sha256': (next(iter(digests)) if digests else None),
        'max_physical_tokens':
        max(row['max_physical_tokens'] for row in records),
        'rollout_peak_allocated_mib':
        max(row['rollout_peak_allocated_mib'] for row in records),
        'rollout_peak_reserved_mib':
        max(row.get('rollout_peak_reserved_mib', row['rollout_peak_allocated_mib']) for row in records),
        'actor_peak_allocated_mib':
        max(row['actor_peak_allocated_mib'] for row in records),
        'actor_peak_reserved_mib':
        max(row.get('actor_peak_reserved_mib', row['actor_peak_allocated_mib']) for row in records),
        'actor_trajectory_microbatch_size':
        actor_batch_size,
        'actor_packed_token_budget':
        actor_token_budget,
        'actor_microbatch_plan_by_rank':
        [row.get('actor_microbatch_plan', []) for row in records],
        'actor_effective_microbatch_size':
        _mean([row.get('actor_effective_microbatch_size', row.get('actor_trajectory_microbatch_size', 1))
               for row in records]),
        'actor_peak_packed_tokens':
        max(row.get('actor_peak_packed_tokens', row.get('max_physical_tokens', 0)) for row in records),
        'rollout_elapsed_s':
        max(row['rollout_elapsed_s'] for row in records),
        'reward_advantage_elapsed_s':
        max(row['reward_advantage_elapsed_s'] for row in records),
        'actor_rebalance_elapsed_s':
        max(row.get('actor_rebalance_elapsed_s', 0.0) for row in records),
        'actor_replay_backward_elapsed_s':
        max(row['actor_replay_backward_elapsed_s'] for row in records),
        'reference_replay_elapsed_s':
        max(row.get('reference_replay_elapsed_s', 0.0) for row in records),
        'actor_forward_elapsed_s':
        max(row.get('actor_forward_elapsed_s', 0.0) for row in records),
        'actor_backward_elapsed_s':
        max(row.get('actor_backward_elapsed_s', 0.0) for row in records),
        'actor_gradient_capture_elapsed_s':
        max(row.get('actor_gradient_capture_elapsed_s', 0.0) for row in records),
        'actor_other_elapsed_s':
        max(row.get('actor_other_elapsed_s', 0.0) for row in records),
        'gradient_update_elapsed_s':
        max(row['gradient_update_elapsed_s'] for row in records),
        'elapsed_s':
        max(row['elapsed_s'] for row in records),
        'dry_run_rollout':
        records[0]['dry_run_rollout'],
        'rank_summaries': [{
            key: row[key]
            for key in (
                'rank',
                'global_prompt_slots',
                'selected_case_ids',
                'selected_frame_counts',
                'estimated_work_units',
                'rollout_physical_token_lengths',
                'trajectory_count',
                'reward',
                'loss',
                'loss_policy_decisions',
                'max_logp_drift',
                'grad_norm',
                'model_parameter_update_norm',
                'rollout_peak_allocated_mib',
                'rollout_peak_reserved_mib',
                'actor_peak_allocated_mib',
                'actor_peak_reserved_mib',
                'actor_trajectory_microbatch_size',
                'actor_packed_token_budget',
                'actor_microbatch_plan',
                'actor_effective_microbatch_size',
                'actor_peak_packed_tokens',
                'rollout_elapsed_s',
                'reward_advantage_elapsed_s',
                'actor_rank_rebalance',
                'actor_rebalance_elapsed_s',
                'actor_replay_backward_elapsed_s',
                'reference_replay_elapsed_s',
                'actor_forward_elapsed_s',
                'actor_backward_elapsed_s',
                'actor_gradient_capture_elapsed_s',
                'actor_other_elapsed_s',
                'gradient_update_elapsed_s',
                'elapsed_s',
            )
            if key in row
        } for row in records],
    }


def _flatten_numeric(prefix: str, value, output: dict) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _flatten_numeric(f'{prefix}/{key}' if prefix else str(key), child, output)
    elif isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        number = float(value)
        if np.isfinite(number):
            output[prefix] = number


SWANLAB_RUN_MANIFEST = 'swanlab_run.json'


def _read_swanlab_run_manifest(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f'invalid SwanLab run manifest: {path}') from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f'SwanLab run manifest is not a JSON object: {path}')
    required = ('run_id', 'project', 'workspace', 'experiment_name')
    missing = [key for key in required if key not in payload]
    if missing:
        raise RuntimeError(f'SwanLab run manifest lacks fields {missing}: {path}')
    if not isinstance(payload['run_id'], str) or not payload['run_id'].strip():
        raise RuntimeError(f'SwanLab run manifest has an invalid run_id: {path}')
    return payload


def _write_swanlab_run_manifest(path: Path, payload: dict) -> None:
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    os.replace(temporary, path)


@contextmanager
def _wall_clock_timeout(seconds: float, description: str):
    """Interrupt a blocking rank-zero SDK call on POSIX.

    SwanLab 0.9.1 can wait indefinitely while its core process establishes an
    online run. In distributed training that used to let the other ranks enter
    rollout and later appear to hang in an unrelated NCCL collective. The
    trainer invokes this context only from Python's main thread on rank zero.
    """
    if seconds <= 0:
        raise ValueError('wall-clock timeout must be positive')
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('wall-clock timeout must run on the main thread')
    if not hasattr(signal, 'SIGALRM') or not hasattr(signal, 'setitimer'):
        raise RuntimeError('wall-clock timeout requires POSIX SIGALRM/setitimer')

    class WallClockTimeout(TimeoutError):
        pass

    def handle_timeout(_signum, _frame):
        raise WallClockTimeout(
            f'{description} exceeded its {seconds:g}s wall-clock timeout')

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()
    signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        previous_delay, previous_interval = previous_timer
        if previous_delay > 0:
            elapsed = time.monotonic() - started
            signal.setitimer(
                signal.ITIMER_REAL,
                max(1e-6, previous_delay - elapsed),
                previous_interval,
            )


class SwanLabLogger:
    """Small rank-zero adapter; the JSONL log remains authoritative."""

    def __init__(self, args, config: dict):
        self.module = None
        if not args.report_to_swanlab or args.rank != 0:
            return
        try:
            import swanlab
        except ImportError as exc:
            raise RuntimeError('SwanLab logging requested but swanlab is not installed') from exc
        output_dir = Path(args.output_dir)
        output_manifest = output_dir / SWANLAB_RUN_MANIFEST
        experiment_name = args.swanlab_exp_name or output_dir.name
        expected = {
            'project': args.swanlab_project,
            'workspace': args.swanlab_workspace,
            'experiment_name': experiment_name,
        }
        manifest = None
        manifest_path = None
        candidates = [output_manifest]
        if args.resume_from_checkpoint:
            candidates.append(Path(args.resume_from_checkpoint).parent / SWANLAB_RUN_MANIFEST)
        for candidate in candidates:
            if candidate.is_file():
                manifest = _read_swanlab_run_manifest(candidate)
                manifest_path = candidate
                break

        run_id = args.swanlab_run_id
        if manifest is not None:
            mismatches = {
                key: {'saved': manifest[key], 'current': value}
                for key, value in expected.items() if manifest[key] != value
            }
            if mismatches:
                raise RuntimeError(
                    'SwanLab run manifest differs from current logging contract: '
                    + json.dumps(mismatches, ensure_ascii=False, sort_keys=True))
            if run_id is not None and run_id != manifest['run_id']:
                raise RuntimeError(
                    f'--swanlab_run_id={run_id!r} differs from {manifest_path}: '
                    f'{manifest["run_id"]!r}')
            run_id = manifest['run_id']

        is_checkpoint_resume = bool(args.resume_from_checkpoint)
        if is_checkpoint_resume and not run_id:
            raise RuntimeError(
                'checkpoint resume has no persisted SwanLab run ID; set '
                'SWANLAB_RUN_ID once to the original run ID, then retry')
        if not is_checkpoint_resume and run_id:
            raise RuntimeError(
                'refusing to resume SwanLab without a model checkpoint; clear '
                f'{output_manifest} or resume a complete checkpoint')

        init_kwargs = {}
        if run_id:
            init_kwargs.update(id=run_id, resume='must')
        else:
            init_kwargs['resume'] = 'never'
        init_timeout = getattr(args, 'swanlab_init_timeout_seconds', 180)
        print(
            f'[dual-grpo] SwanLab init start timeout={init_timeout}s '
            f'resume={is_checkpoint_resume}',
            flush=True,
        )
        with _wall_clock_timeout(init_timeout, 'SwanLab initialization'):
            run = swanlab.init(
                project=args.swanlab_project,
                workspace=args.swanlab_workspace,
                experiment_name=experiment_name,
                config=config,
                logdir=str(output_dir / 'swanlab'),
                mode=args.swanlab_mode,
                **init_kwargs,
            )
        active_run_id = getattr(run, 'id', None)
        if not isinstance(active_run_id, str) or not active_run_id.strip():
            raise RuntimeError('SwanLab init did not return a valid run ID')
        if run_id and active_run_id != run_id:
            raise RuntimeError(
                f'SwanLab resumed unexpected run {active_run_id!r}; expected {run_id!r}')
        _write_swanlab_run_manifest(output_manifest, {
            **expected,
            'run_id': active_run_id,
        })
        print(
            f'[dual-grpo] SwanLab run_id={active_run_id} '
            f'resume={is_checkpoint_resume}',
            flush=True,
        )
        self.module = swanlab

    def log_train(self, record: dict) -> None:
        if self.module is None:
            return
        keys = (
            'reward',
            'reward_format',
            'reward_timing',
            'loss',
            'policy_loss',
            'kl',
            'kl_loss_weighted',
            'silent_rate_loss',
            'silent_rate_loss_weighted',
            'action_outside_loss',
            'action_outside_loss_weighted',
            'action_coverage_loss',
            'action_coverage_loss_weighted',
            'action_count_loss',
            'action_count_loss_weighted',
            'action_timing_loss_weighted',
            'action_expected_inside_speak_count',
            'action_expected_outside_speak_count',
            'action_mean_event_hit_probability',
            'expected_speak_count',
            'target_speak_count',
            'expected_silent_rate',
            'target_silent_rate',
            'loss_policy_decisions',
            'ratio',
            'clip_frac',
            'max_logp_drift',
            'max_mean_logp_drift',
            'grad_norm',
            'model_parameter_update_norm',
            'max_physical_tokens',
            'rollout_peak_allocated_mib',
            'rollout_peak_reserved_mib',
            'actor_peak_allocated_mib',
            'actor_peak_reserved_mib',
            'actor_effective_microbatch_size',
            'actor_peak_packed_tokens',
            'rollout_elapsed_s',
            'reward_advantage_elapsed_s',
            'actor_rebalance_elapsed_s',
            'actor_replay_backward_elapsed_s',
            'reference_replay_elapsed_s',
            'actor_forward_elapsed_s',
            'actor_backward_elapsed_s',
            'actor_gradient_capture_elapsed_s',
            'actor_other_elapsed_s',
            'gradient_update_elapsed_s',
            'elapsed_s',
        )
        metrics = {f'train/{key}': record[key] for key in keys if record.get(key) is not None}
        _flatten_numeric('train/grad_family_norms', record.get('grad_family_norms', {}), metrics)
        _flatten_numeric('train/window', record.get('window_reward_metrics', {}), metrics)
        _flatten_numeric('train/window_by_source', record.get('window_reward_metrics_by_source', {}), metrics)
        _flatten_numeric('train/group_sampling', record.get('group_sampling', {}), metrics)
        _flatten_numeric('train/actor_rank_rebalance', record.get('actor_rank_rebalance', {}), metrics)
        _flatten_numeric('train/frozen_visual_reuse', record.get('frozen_visual_feature_reuse', {}), metrics)
        metrics['train/frozen_visual_reuse/enabled'] = float(
            record.get('frozen_visual_feature_reuse', {}).get('enabled', False))
        metrics['train/optimizer_step_applied'] = float(record['optimizer_step_applied'])
        global_has_advantage = bool(record.get('global_has_advantage', False))
        metrics['train/global_has_advantage'] = float(global_has_advantage)
        metrics['train/global_has_silent_rate_signal'] = float(
            record.get('global_has_silent_rate_signal', False))
        metrics['train/global_has_action_timing_signal'] = float(
            record.get('global_has_action_timing_signal', False))
        metrics['train/global_has_kl_signal'] = float(
            record.get('global_has_kl_signal', False))
        metrics['train/global_has_update_signal'] = float(
            record.get('global_has_update_signal', global_has_advantage))
        metrics['train/group_reward_std_nonzero'] = float(record['group_reward_std_nonzero'])
        self.module.log(metrics, step=int(record['step']))

    def log_validation(self, summary: dict) -> None:
        if self.module is None:
            return
        metrics = {}
        _flatten_numeric('val', summary, metrics)
        self.module.log(metrics, step=int(summary['step']))

    def finish(self) -> None:
        if self.module is not None:
            self.module.finish()


def _initialize_swanlab(args, config: dict) -> SwanLabLogger:
    """Initialize on rank zero while every nonzero rank waits for its result."""
    local_logger = None
    if args.rank != 0:
        local_logger = SwanLabLogger(args, config)
    rank_zero_logger = _rank0_call_or_raise(
        args,
        'initialize SwanLab',
        lambda: SwanLabLogger(args, config),
    )
    logger = rank_zero_logger if args.rank == 0 else local_logger
    if logger is None:
        raise RuntimeError('SwanLab startup rendezvous returned no logger')
    return logger


def _configure_determinism(enabled: bool) -> None:
    if enabled:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)


def _configure_actor_flex_kernel_options(args) -> None:
    """Make actor full attention use the benchmark-inference kernel layout."""
    configured = os.environ.get('MSS_FLEX_KERNEL_OPTS', '').replace(' ', '')
    expected = INFERENCE_ALIGNED_FLEX_KERNEL_OPTIONS
    if configured and configured != expected:
        raise RuntimeError(
            'inference_aligned actor replay requires MSS_FLEX_KERNEL_OPTS='
            f'{expected!r}, got {configured!r}; different FlexAttention tiles '
            'can change BF16 policy logits')
    os.environ['MSS_FLEX_KERNEL_OPTS'] = expected


def main(argv=None):
    args = parse_args(argv)
    os.environ['MSS_DUAL_ACTOR_LINEAR_MODE'] = args.actor_linear_mode
    os.environ['MSS_DUAL_ACTOR_FUSED_FROZEN_LINEAR_BACKWARD'] = (
        '1' if args.actor_fused_frozen_linear_backward else '0'
    )
    os.environ['MSS_DUAL_ACTOR_FUSED_TRAINABLE_LINEAR_INPUT_BACKWARD'] = (
        '1' if args.actor_fused_trainable_linear_input_backward else '0'
    )
    os.environ['MSS_DUAL_ACTOR_SEGMENTED_RMSNORM_BACKWARD'] = (
        '1' if args.actor_segmented_rmsnorm_backward else '0'
    )
    os.environ['MSS_DUAL_ACTOR_FUSED_GATED_RMSNORM_BACKWARD'] = (
        '1' if args.actor_fused_gated_rmsnorm_backward else '0'
    )
    os.environ['MSS_DUAL_ACTOR_FUSED_LM_HEAD_INPUT_BACKWARD'] = (
        '1' if args.actor_fused_lm_head_input_backward else '0'
    )
    if (args.actor_fused_lm_head_input_backward
            and not args.defer_actor_lm_head):
        raise ValueError('actor_fused_lm_head_input_backward requires '
                         'defer_actor_lm_head')
    _configure_actor_flex_kernel_options(args)
    _configure_determinism(args.deterministic_algorithms)
    _init_distributed(args)
    cases, train_info = load_timing_cases(
        args.timing_train_jsonl,
        args.timing_manifest,
        max_frames=args.max_frames,
        video_root_overrides=args.video_root,
        indices=args.cases,
    )
    val_cases, val_info = load_timing_cases(
        args.timing_val_jsonl,
        args.timing_manifest,
        max_frames=None,
        video_root_overrides=args.video_root,
    )
    data_source = 'proactive_timing_v1'
    data_info = {'train': train_info, 'validation': val_info}
    args.timing_train_sha256 = train_info['sha256']
    args.timing_val_sha256 = val_info['sha256']
    args.timing_manifest_sha256 = train_info['manifest_sha256']
    if args.timing_data_variant != FULL_POSITIVE_DATA:
        pool_case_count = len(cases)
        cases, epoch_info = _materialize_timing_epoch(
            cases,
            epoch_steps=args.timing_epoch_steps,
            query_batch_size=args.query_batch_size,
            seed=args.seed,
            variant=args.timing_data_variant,
        )
        data_info['train'] = {
            **train_info,
            'pool_cases': pool_case_count,
            'materialized_epoch': epoch_info,
        }
        expected_steps = args.timing_epoch_steps * args.num_train_epochs
        if args.max_steps is not None and args.max_steps != expected_steps:
            raise ValueError('materialized timing epoch requires max_steps == '
                             f'timing_epoch_steps * num_train_epochs ({expected_steps}), '
                             f'got {args.max_steps}')
        args.max_steps = expected_steps
    elif args.max_steps is None:
        steps_per_epoch = (len(cases) + args.query_batch_size - 1) // args.query_batch_size
        args.max_steps = steps_per_epoch * args.num_train_epochs
    if len(cases) < args.query_batch_size:
        raise ValueError(f'training data has {len(cases)} cases for query_bs={args.query_batch_size}')
    _validate_resume_argument_contract(args.resume_from_checkpoint, args)
    if not torch.cuda.is_available() or not str(args.device).startswith('cuda'):
        raise RuntimeError('dual-stream GRPO requires a CUDA device')
    autotune_runtime = _install_serialized_triton_autotune(args)
    if autotune_runtime['enabled']:
        _rank_print(
            args, 'serialized Triton autotune cache='
            f'{autotune_runtime["cache_dir"]} lock='
            f'{autotune_runtime["lock_path"]} '
            f'scope={autotune_runtime["lock_scope"]}')
    process_seed = args.seed + args.rank * 10_000_019
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed(process_seed)
    os.environ.setdefault('STEP_FAST_ENCODE', '1')
    configured_window = os.environ.get('STEP_MASK_WINDOW')
    if configured_window is not None:
        try:
            configured_window_value = int(configured_window)
        except ValueError as exc:
            raise RuntimeError(f'invalid STEP_MASK_WINDOW={configured_window!r}') from exc
        if configured_window_value != args.window:
            raise RuntimeError(f'STEP_MASK_WINDOW={configured_window_value} conflicts with '
                               f'--window={args.window}; rollout/actor masks must agree')
    os.environ['STEP_MASK_WINDOW'] = str(args.window)

    from examples.infer.proactive_eval.checkpoint_utils import checkpoint_has_dual_weights
    from examples.infer.proactive_eval.runner import StreamModel

    effective_model = args.resume_from_checkpoint or args.model
    if not checkpoint_has_dual_weights(effective_model):
        raise RuntimeError(f'checkpoint is not detected as dual-branch: {effective_model}')
    stream_kwargs = dict(
        model_path=effective_model,
        device=args.device,
        window=args.window,
        warmup=args.warmup,
    )
    if args.system_prompt is not None:
        stream_kwargs['system_prompt_path'] = args.system_prompt
    sm = StreamModel(**stream_kwargs)
    if not sm.dual_branch:
        raise RuntimeError('StreamModel did not load dual-branch modules')
    model = sm.model
    _disable_dropout(model)
    trainable_names, trainable_parameters = _select_trainable_parameters(model)
    sft_reference = None
    if args.kl_coef > 0:
        # A fresh actor is already the SFT checkpoint. A resumed actor must
        # reload the original SFT LoRA tensors rather than snapshotting its
        # updated RL weights.
        reference_checkpoint = args.model if args.resume_from_checkpoint else None
        sft_reference = FrozenTrainableReference.create(
            trainable_names,
            trainable_parameters,
            source_checkpoint=reference_checkpoint,
        )
        args.sft_reference_trainable_sha256 = sft_reference.sha256
        _validate_sft_reference_digest(
            args.resume_from_checkpoint,
            args.sft_reference_trainable_sha256,
        )
    else:
        args.sft_reference_trainable_sha256 = None
    if args.reuse_frozen_visual_features:
        if args.actor_linear_mode != 'inference_aligned':
            raise RuntimeError('frozen visual reuse requires inference_aligned actor replay')
        visual = getattr(getattr(model, 'model', None), 'visual', None)
        if visual is None or any(parameter.requires_grad for parameter in visual.parameters()):
            raise RuntimeError('frozen visual reuse requires a present, fully frozen visual tower')
        if not getattr(visual, '_mss_actor_per_image_patched', False):
            raise RuntimeError('frozen visual reuse requires the actor visual replay patch')
    checkpoint_runtime = _configure_gradient_checkpointing(model, args)
    optimizer = FP32MasterAdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=True,
    )
    start_step = _restore_training_state(
        args.resume_from_checkpoint,
        optimizer,
        trainable_names=trainable_names,
        rank=args.rank,
        world_size=args.world_size,
    )
    if start_step >= args.max_steps:
        raise ValueError(f'resume step {start_step} must be smaller than ' f'--max_steps {args.max_steps}')

    initial_trainable_digest = None
    if args.distributed:
        initial_trainable_digest = _assert_exact_trainable_rank_equality(
            trainable_names,
            trainable_parameters,
            world_size=args.world_size,
        )
        _rank_print(
            args,
            f'initial trainable SHA256={initial_trainable_digest}',
        )

    output_dir = Path(args.output_dir)
    _rank0_call_or_raise(
        args,
        'create the RL output directory',
        lambda: output_dir.mkdir(parents=True, exist_ok=True),
    )
    aggregate_log_path = (Path(args.log_path) if args.log_path else output_dir / 'train.jsonl')
    _rank0_call_or_raise(
        args,
        'create the RL log directory',
        lambda: aggregate_log_path.parent.mkdir(parents=True, exist_ok=True),
    )
    log_path = aggregate_log_path
    if args.distributed:
        log_path = aggregate_log_path.with_name(f'{aggregate_log_path.stem}.rank{args.rank}'
                                                f'{aggregate_log_path.suffix}')
    train_template = get_template(
        sm.processor,
        default_system=sm.system_prompt,
        template_type='qwen3_5',
        use_chat_template=True,
        enable_thinking=False,
        response_prefix='',
        add_non_thinking_prefix=sm.template_add_non_thinking_prefix,
        loss_scale=sm.template_loss_scale,
        padding_free=True,
        max_length=args.max_sequence_length,
    )
    train_template.set_mode('train')
    policy_sampler = DualBranchSampler(
        sm.tokenizer,
        max_response_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=None,
    )
    reward_weights = DualRewardWeights(
        format=args.reward_format,
        timing=args.reward_timing,
        sigma=args.reward_sigma,
        timing_mode=args.timing_reward_mode,
    )
    run_contract = {
        'model': str(Path(effective_model).expanduser().resolve()),
        'sft_reference_trainable_sha256': args.sft_reference_trainable_sha256,
        'data': data_info,
        'gradient_checkpointing_runtime': checkpoint_runtime,
        **{key: getattr(args, key)
           for key in RESUME_CONTRACT_KEYS},
    }
    n_trainable = sum(parameter.numel() for parameter in trainable_parameters)
    print(
        f'[dual-grpo] model={effective_model}\n'
        f'[dual-grpo] data_source={data_source}\n'
        f'[dual-grpo] train_cases={len(cases)} val_cases={len(val_cases)} max_steps={args.max_steps}\n'
        f'[dual-grpo] query_bs={args.query_batch_size} G={args.group_size} '
        f'rollout_trajectory_microbatch='
        f'{args.rollout_trajectory_microbatch_size or args.local_query_batch_size} '
        f'reuse_frozen_visual_features={args.reuse_frozen_visual_features} '
        f'defer_actor_lm_head={args.defer_actor_lm_head} '
        f'actor_fused_frozen_linear_backward={args.actor_fused_frozen_linear_backward} '
        f'actor_fused_trainable_linear_input_backward='
        f'{args.actor_fused_trainable_linear_input_backward} '
        f'actor_segmented_rmsnorm_backward={args.actor_segmented_rmsnorm_backward} '
        f'actor_fused_gated_rmsnorm_backward='
        f'{args.actor_fused_gated_rmsnorm_backward} '
        f'actor_fused_lm_head_input_backward='
        f'{args.actor_fused_lm_head_input_backward} '
        f'rollout_query_mode={args.rollout_query_batch_mode} '
        f'actor_trajectory_microbatch={args.actor_trajectory_microbatch_size or 1} '
        f'actor_packed_token_budget={args.actor_packed_token_budget} '
        f'batch_schedule={args.batch_schedule} rank_sharding={args.rank_sharding}\n'
        f'[dual-grpo] timing_data_variant={args.timing_data_variant} '
        f'timing_epoch_steps={args.timing_epoch_steps} '
        f'timing_reward_mode={args.timing_reward_mode} '
        f'timing_sigma={args.reward_sigma} '
        f'silent_rate_loss_weight={args.silent_rate_loss_weight} '
        f'action_loss_weights=({args.action_outside_loss_weight},'
        f'{args.action_coverage_loss_weight},{args.action_count_loss_weight}) '
        f'kl_coef={args.kl_coef} '
        f'policy_decision_scope={args.policy_decision_scope}\n'
        f'[dual-grpo] gradient_checkpointing={checkpoint_runtime}\n'
        f'[dual-grpo] swanlab={args.report_to_swanlab} mode={args.swanlab_mode} '
        f'project={args.swanlab_project} workspace={args.swanlab_workspace or "<personal>"}\n'
        f'[dual-grpo] trainable={n_trainable:,} tensors={len(trainable_names)} '
        f'(dual LoRA only)',
        flush=True,
    )
    swanlab_config = {
        **run_contract,
        'trainer_args': vars(args),
        'world_size': args.world_size,
        'train_case_count': len(cases),
        'validation_case_count': len(val_cases),
        'trainable_parameters': n_trainable,
    }
    # Rank zero performs the only external initialization. All other ranks
    # rendezvous here, so they cannot waste a full rollout and later blame the
    # first content/actor collective when SwanLab itself is stalled.
    swanlab_logger = _initialize_swanlab(args, swanlab_config)
    _rank_print(args, 'startup rendezvous complete', all_ranks=True)
    if val_cases and args.eval_at_start:
        summary = run_timing_validation(sm, val_cases, args=args, step=start_step, output_dir=output_dir)
        if args.rank == 0:
            swanlab_logger.log_validation(summary)
            print(
                f'[val step {start_step}] mean_f1={summary["overall"]["mean_f1"]:.4f} '
                f'macro_f1={summary["overall"]["macro_f1"]:.4f}',
                flush=True)

    with ExitStack() as stack:
        if args.rank == 0:
            stack.callback(swanlab_logger.finish)
        log_handle = stack.enter_context(open(log_path, 'a', encoding='utf-8'))
        aggregate_log_handle = log_handle
        if args.distributed and args.rank == 0:
            aggregate_log_handle = stack.enter_context(open(aggregate_log_path, 'a', encoding='utf-8'))
        for step in range(start_step, args.max_steps):
            step_start = _synchronized_time(args.device)
            torch.cuda.reset_peak_memory_stats(torch.device(args.device))
            step_base_allocated = torch.cuda.memory_allocated(torch.device(args.device))
            step_base_reserved = torch.cuda.memory_reserved(torch.device(args.device))
            global_selected_cases = _training_batch(
                cases,
                step,
                args.query_batch_size,
                args.seed,
                args.batch_schedule,
            )
            global_work_units = [_case_estimated_work(case) for case in global_selected_cases]
            prompt_slot_shards = rank_prompt_slot_shards(
                global_work_units,
                args.world_size,
                args.rank_sharding,
            )
            planned_rank_loads = shard_loads(prompt_slot_shards, global_work_units)
            _rank_print(
                args,
                f'[step {step + 1}] planned rank work={planned_rank_loads} '
                f'max/min={max(planned_rank_loads) / min(planned_rank_loads):.3f}',
            )
            local_prompt_slots = prompt_slot_shards[args.rank]
            selected_cases = [global_selected_cases[slot] for slot in local_prompt_slots]
            selected_frame_counts = [_case_frame_count(case) for case in selected_cases]
            estimated_work_units = sum(global_work_units[slot] for slot in local_prompt_slots)
            _rank_print(
                args,
                f'[step {step + 1}] shard={args.rank_sharding} '
                f'prompt_slots={local_prompt_slots} '
                f'case_ids={[case.row_index for case in selected_cases]} '
                f'frames={selected_frame_counts} work={estimated_work_units}',
                all_ranks=True,
            )
            tasks = build_rollout_tasks(
                selected_cases,
                group_size=args.group_size,
                seed=args.seed,
                step=step,
                prompt_slots=local_prompt_slots,
            )

            rollout_start = _synchronized_time(args.device)
            model.eval()
            rollout_visual_reuse = {}
            trajectories = rollout_trajectories(
                sm,
                tasks,
                args=args,
                stats=rollout_visual_reuse,
            )
            rollout_elapsed_s = (_synchronized_time(args.device) - rollout_start)
            rollout_physical_token_lengths = [len(trajectory.physical_ids) for trajectory in trajectories]
            _rank_print(
                args,
                f'[step {step + 1}] rollout complete t={rollout_elapsed_s:.1f}s '
                f'physical_tokens={rollout_physical_token_lengths}',
                all_ranks=True,
            )
            rollout_peak_allocated = torch.cuda.max_memory_allocated(torch.device(args.device))
            rollout_peak_reserved = torch.cuda.max_memory_reserved(torch.device(args.device))
            reward_advantage_start = time.perf_counter()
            reward_keys = (
                'total',
                'format',
                'timing',
                'speak_count_mul',
                'shaped_timing',
            )
            for trajectory in trajectories:
                trajectory.reward = aggregate_dual_reward(
                    trajectory.frames,
                    trajectory.case.gt,
                    reward_weights,
                )
                invalid = [
                    key for key in reward_keys
                    if key not in trajectory.reward or not np.isfinite(float(trajectory.reward[key]))
                ]
                if invalid:
                    raise RuntimeError('dual reward has missing/non-finite components for '
                                       f'case={trajectory.case.row_index}: {invalid}')

            advantages = _advantages_by_query(
                trajectories,
                prompt_slots=local_prompt_slots,
                group_size=args.group_size,
            )
            reward_advantage_elapsed_s = (time.perf_counter() - reward_advantage_start)

            actor_rebalance_elapsed_s = 0.0
            rebalance_requested = actor_rebalance_enabled(
                args.actor_rank_rebalance,
                args.world_size,
            )
            if rebalance_requested:
                actor_rebalance_start = _synchronized_time(args.device)
            actor_trajectories, actor_advantages, actor_rebalance = _rebalance_actor_trajectories(
                sm,
                trajectories,
                advantages,
                global_selected_cases=global_selected_cases,
                local_prompt_slots=local_prompt_slots,
                args=args,
            )
            if rebalance_requested:
                actor_rebalance_elapsed_s = (_synchronized_time(args.device) - actor_rebalance_start)
            if actor_rebalance['effective']:
                _rank_print(
                    args,
                    f'[step {step + 1}] actor rebalance '
                    f'rollout_slots={local_prompt_slots} '
                    f'actor_slots={actor_rebalance["actor_prompt_slots"]} '
                    f'loads={actor_rebalance["sampled_decision_loads"]} '
                    f't={actor_rebalance_elapsed_s:.1f}s',
                    all_ranks=True,
                )
            elif rebalance_requested:
                _rank_print(
                    args,
                    f'[step {step + 1}] actor rebalance skipped '
                    f'gain={actor_rebalance["predicted_max_load_relative_gain"]:.4f} '
                    f'minimum={actor_rebalance["minimum_relative_gain"]:.4f} '
                    f't={actor_rebalance_elapsed_s:.1f}s',
                    all_ranks=True,
                )

            actor_start = _synchronized_time(args.device)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(torch.device(args.device))
            actor_base_allocated = torch.cuda.memory_allocated(torch.device(args.device))
            actor_base_reserved = torch.cuda.memory_reserved(torch.device(args.device))
            model.train()
            _disable_dropout(model)
            train_template.register_post_encode_hook([model])
            metric_rows = []
            max_logp_drift = 0.0
            max_logp_drift_by_route = {'B': 0.0, 'A': 0.0}
            max_logp_drift_by_kind = {'action': 0.0, 'content': 0.0}
            max_mean_logp_drift = 0.0
            importance_ratio_min = float('inf')
            importance_ratio_max = 0.0
            max_tokens_seen = 0
            alignment_rows = []
            max_actor_isolation_logp_drift = 0.0
            max_mean_actor_isolation_logp_drift = 0.0
            actor_batch_size = (args.actor_trajectory_microbatch_size or 1)
            actor_microbatch_plan = _actor_microbatch_plan(
                actor_trajectories,
                max_trajectories=actor_batch_size,
                packed_token_budget=args.actor_packed_token_budget,
            )
            micro_gradient_diagnostics = []
            local_high_precision_gradients = None
            actor_forward_events = []
            actor_backward_events = []
            actor_gradient_capture_events = []
            reference_replay_elapsed_s = 0.0
            reference_logp_rows = None
            if not args.dry_run_rollout:
                # A parameter's .grad has the parameter dtype (BF16 here).
                # Accumulating multiple trajectory backwards in that buffer
                # makes the result depend on the local rank/chunk partition.
                # Capture every microbatch gradient in FP64 immediately.
                # FP64 preserves the sum of this small set of FP32-equivalent
                # micro-gradients across different rank reduction trees.
                local_high_precision_gradients = [
                    torch.zeros_like(parameter, dtype=torch.float64) for parameter in trainable_parameters
                ]
            try:
                if sft_reference is not None:
                    reference_start = _synchronized_time(args.device)
                    reference_logp_rows = _reference_logps_for_actor_plan(
                        sft_reference,
                        model,
                        train_template,
                        actor_trajectories,
                        actor_microbatch_plan,
                        policy_sampler,
                        assistant_role_id=sm.assistant_role_id,
                        device=args.device,
                        reuse_frozen_visual_features=args.reuse_frozen_visual_features,
                        defer_actor_lm_head=args.defer_actor_lm_head,
                    )
                    reference_replay_elapsed_s = (
                        _synchronized_time(args.device) - reference_start
                    )
                for chunk_start, chunk_stop, _ in actor_microbatch_plan:
                    chunk = actor_trajectories[chunk_start:chunk_stop]
                    chunk_advantages = actor_advantages[chunk_start:chunk_stop]
                    isolated_replay_rows = None
                    if args.validate_actor_isolation and len(chunk) > 1:
                        isolated_replay_rows = []
                        with torch.no_grad():
                            for trajectory in chunk:
                                with _cuda_profile_region(
                                        args.profile_actor_phases,
                                        actor_forward_events,
                                ):
                                    isolated_replay_rows.append(
                                        actor_logps_for_trajectories(
                                            model,
                                            train_template,
                                            [trajectory],
                                            policy_sampler,
                                            assistant_role_id=sm.assistant_role_id,
                                            device=args.device,
                                            reuse_frozen_visual_features=args.reuse_frozen_visual_features,
                                            defer_actor_lm_head=args.defer_actor_lm_head,
                                        )[0])
                    # Keep the packed forward last. Gradient-checkpoint
                    # recomputation reads module-level mask/fork metadata and
                    # must see this exact packed context during backward.
                    with torch.set_grad_enabled(not args.dry_run_rollout):
                        with _cuda_profile_region(
                                args.profile_actor_phases,
                                actor_forward_events,
                        ):
                            replay_rows = actor_logps_for_trajectories(
                                model,
                                train_template,
                                chunk,
                                policy_sampler,
                                assistant_role_id=sm.assistant_role_id,
                                device=args.device,
                                reuse_frozen_visual_features=args.reuse_frozen_visual_features,
                                defer_actor_lm_head=args.defer_actor_lm_head,
                            )
                    chunk_losses = []
                    for row_in_chunk, (trajectory, replay,
                                       advantage) in enumerate(zip(chunk, replay_rows, chunk_advantages)):
                        (actor_logps, rollout_logps, sequence_length, alignment) = replay
                        alignment.update({
                            'case_id': trajectory.case.row_index,
                            'prompt_slot': trajectory.prompt_slot,
                            'group_index': trajectory.group_index,
                            'actor_chunk_start': chunk_start,
                            'reward_total': float(trajectory.reward['total']),
                            'advantage': float(advantage),
                        })
                        if isolated_replay_rows is not None:
                            isolated_logps = isolated_replay_rows[row_in_chunk][0].detach()
                            if actor_logps.shape != isolated_logps.shape:
                                raise RuntimeError('packed/isolated actor decision count '
                                                   f'mismatch: {tuple(actor_logps.shape)} != '
                                                   f'{tuple(isolated_logps.shape)}')
                            isolation_drift = (actor_logps.detach() - isolated_logps).abs()
                            isolation_worst = int(isolation_drift.argmax().item())
                            isolation_decisions = _policy_decisions(trajectory, policy_sampler)
                            isolation_decision = isolation_decisions[isolation_worst]
                            isolation_summary = {
                                'max_abs_logp_drift': float(isolation_drift.max().item()),
                                'mean_abs_logp_drift': float(isolation_drift.mean().item()),
                                'worst_policy_decision_index': isolation_worst,
                                'route': isolation_decision.route,
                                'frame_index': isolation_decision.frame_index,
                                'token_index': isolation_decision.token_index,
                                'token_id': isolation_decision.token_id,
                                'target_position': isolation_decision.position,
                                'prediction_position': (isolation_decision.position - 1),
                                'packed_logp': float(actor_logps.detach()[isolation_worst].item()),
                                'isolated_logp': float(isolated_logps[isolation_worst].item()),
                            }
                            alignment['packed_vs_isolated_actor'] = (isolation_summary)
                            max_actor_isolation_logp_drift = max(
                                max_actor_isolation_logp_drift,
                                isolation_summary['max_abs_logp_drift'],
                            )
                            max_mean_actor_isolation_logp_drift = max(
                                max_mean_actor_isolation_logp_drift,
                                isolation_summary['mean_abs_logp_drift'],
                            )
                            if (isolation_summary['max_abs_logp_drift'] > args.max_actor_isolation_logp_drift):
                                raise RuntimeError('packed actor differs from one-trajectory '
                                                   'actor reference: diagnostic='
                                                   f'{json.dumps(isolation_summary, sort_keys=True)} '
                                                   f'limit={args.max_actor_isolation_logp_drift}; '
                                                   'refusing update')
                        drift = float((actor_logps.detach() - rollout_logps).abs().max().item())
                        max_logp_drift = max(max_logp_drift, drift)
                        for route in ('B', 'A'):
                            max_logp_drift_by_route[route] = max(
                                max_logp_drift_by_route[route],
                                alignment['max_logp_drift_by_route'][route],
                            )
                        for kind in ('action', 'content'):
                            max_logp_drift_by_kind[kind] = max(
                                max_logp_drift_by_kind[kind],
                                alignment['max_logp_drift_by_kind'][kind],
                            )
                        max_mean_logp_drift = max(
                            max_mean_logp_drift,
                            alignment['mean_logp_drift'],
                        )
                        importance_ratio_min = min(
                            importance_ratio_min,
                            alignment['same_weight_importance_ratio_min'],
                        )
                        importance_ratio_max = max(
                            importance_ratio_max,
                            alignment['same_weight_importance_ratio_max'],
                        )
                        max_tokens_seen = max(max_tokens_seen, sequence_length)
                        alignment_rows.append(alignment)
                        alignment_diagnostic = json.dumps(
                            {
                                'case_id': alignment['case_id'],
                                'prompt_slot': alignment['prompt_slot'],
                                'group_index': alignment['group_index'],
                                'mean_logp_drift': alignment['mean_logp_drift'],
                                'max_logp_drift_by_route': (alignment['max_logp_drift_by_route']),
                                'max_logp_drift_by_kind': (alignment['max_logp_drift_by_kind']),
                                'worst': alignment['worst_logp_drift'],
                            },
                            sort_keys=True)
                        if drift > args.max_logp_drift:
                            raise RuntimeError('rollout/training-layout logp gate failed: '
                                               f'max_abs={drift:.6f} > '
                                               f'{args.max_logp_drift:.6f}; diagnostic='
                                               f'{alignment_diagnostic}; refusing update')
                        if (alignment['mean_logp_drift'] > args.max_mean_logp_drift):
                            raise RuntimeError('rollout/training-layout mean logp gate '
                                               f'failed: mean_abs='
                                               f'{alignment["mean_logp_drift"]:.6f} > '
                                               f'{args.max_mean_logp_drift:.6f}; '
                                               f'diagnostic={alignment_diagnostic}; '
                                               'refusing update')
                        loss_indices = torch.as_tensor(
                            _policy_loss_decision_indices(
                                trajectory,
                                policy_sampler,
                                args.policy_decision_scope,
                            ),
                            dtype=torch.long,
                            device=actor_logps.device,
                        )
                        loss_actor_logps = actor_logps.index_select(0, loss_indices)
                        loss_rollout_logps = rollout_logps.index_select(0, loss_indices)
                        if reference_logp_rows is None:
                            loss_reference_logps = loss_actor_logps.detach()
                        else:
                            reference_logps = reference_logp_rows[
                                chunk_start + row_in_chunk
                            ]
                            if reference_logps.shape != actor_logps.shape:
                                raise RuntimeError(
                                    'SFT reference/actor decision count mismatch: '
                                    f'{tuple(reference_logps.shape)} != '
                                    f'{tuple(actor_logps.shape)}'
                                )
                            loss_reference_logps = reference_logps.index_select(
                                0, loss_indices
                            )
                        alignment['policy_decision_scope'] = args.policy_decision_scope
                        alignment['loss_policy_decisions'] = int(loss_indices.numel())
                        metrics = grpo_per_token_loss(
                            actor_logps=loss_actor_logps,
                            rollout_logps=loss_rollout_logps,
                            ref_logps=loss_reference_logps,
                            advantage=advantage,
                            clip_eps=args.clip_eps,
                            kl_coef=args.kl_coef,
                        )
                        grpo_loss = metrics['loss']
                        policy_loss = metrics['pg_loss']
                        weighted_kl_loss = metrics['kl'] * args.kl_coef
                        action_loss_enabled = any(weight > 0 for weight in (
                            args.action_outside_loss_weight,
                            args.action_coverage_loss_weight,
                            args.action_count_loss_weight,
                        ))
                        calibration_logps = (
                            actor_logps if args.silent_rate_loss_weight > 0 or action_loss_enabled
                            else actor_logps.detach()
                        )
                        decisions = _policy_decisions(trajectory, policy_sampler)
                        silent_rate_loss, silent_rate_diagnostics = expected_silent_rate_loss(
                            calibration_logps,
                            decisions,
                            speak_token_id=policy_sampler.ids['<speak>'],
                            silent_token_id=policy_sampler.silent_id,
                            target_speak_count=_target_speak_count(trajectory),
                            frame_count=len(trajectory.frames),
                        )
                        weighted_silent_rate_loss = (
                            silent_rate_loss * args.silent_rate_loss_weight
                        )
                        zero = calibration_logps.sum() * 0.0
                        action_losses = {
                            'action_outside_loss': zero,
                            'action_coverage_loss': zero,
                            'action_count_loss': zero,
                        }
                        action_diagnostics = {
                            'action_expected_inside_speak_count': 0.0,
                            'action_expected_outside_speak_count': 0.0,
                            'action_mean_event_hit_probability': 0.0,
                            'action_event_region_count': 0.0,
                            'action_target_event_count': 0.0,
                            'action_outside_frame_count': 0.0,
                        }
                        if action_loss_enabled:
                            frame_times = [
                                float(frame.get('t', frame.get('frame_idx', frame_index)))
                                for frame_index, frame in enumerate(trajectory.frames)
                            ]
                            action_losses, action_diagnostics = action_timing_auxiliary_loss(
                                calibration_logps,
                                decisions,
                                speak_token_id=policy_sampler.ids['<speak>'],
                                silent_token_id=policy_sampler.silent_id,
                                frame_times=frame_times,
                                gt_windows=trajectory.case.gt.get('gt_windows'),
                            )
                        weighted_action_outside_loss = (
                            action_losses['action_outside_loss']
                            * args.action_outside_loss_weight
                        )
                        weighted_action_coverage_loss = (
                            action_losses['action_coverage_loss']
                            * args.action_coverage_loss_weight
                        )
                        weighted_action_count_loss = (
                            action_losses['action_count_loss']
                            * args.action_count_loss_weight
                        )
                        weighted_action_timing_loss = (
                            weighted_action_outside_loss
                            + weighted_action_coverage_loss
                            + weighted_action_count_loss
                        )
                        metrics['policy_loss'] = policy_loss
                        metrics['kl_loss_weighted'] = weighted_kl_loss
                        metrics['silent_rate_loss'] = silent_rate_loss
                        metrics['silent_rate_loss_weighted'] = weighted_silent_rate_loss
                        metrics.update(action_losses)
                        metrics['action_outside_loss_weighted'] = weighted_action_outside_loss
                        metrics['action_coverage_loss_weighted'] = weighted_action_coverage_loss
                        metrics['action_count_loss_weighted'] = weighted_action_count_loss
                        metrics['action_timing_loss_weighted'] = weighted_action_timing_loss
                        for key, value in silent_rate_diagnostics.items():
                            metrics[key] = actor_logps.new_tensor(value)
                        for key, value in action_diagnostics.items():
                            metrics[key] = actor_logps.new_tensor(value)
                        metrics['loss'] = (
                            grpo_loss
                            + weighted_silent_rate_loss
                            + weighted_action_timing_loss
                        )
                        required_metric_keys = ('loss', 'ratio', 'clip_frac')
                        missing_metric_keys = [key for key in required_metric_keys if key not in metrics]
                        nonfinite_metric_keys = [
                            key for key, value in metrics.items() if not isinstance(value, torch.Tensor)
                            or value.numel() != 1 or not bool(torch.isfinite(value.detach()).item())
                        ]
                        if missing_metric_keys or nonfinite_metric_keys:
                            raise RuntimeError('GRPO loss produced an invalid scalar: '
                                               f'missing={missing_metric_keys} '
                                               f'nonfinite={nonfinite_metric_keys}')
                        clip_fraction = float(metrics['clip_frac'].detach().item())
                        if clip_fraction != 0.0:
                            raise RuntimeError('same-weight rollout/replay importance ratio '
                                               'already enters the GRPO clip interval before '
                                               'any optimizer update; refusing a numerically '
                                               f'biased step (clip_frac={clip_fraction:.6f}); '
                                               f'diagnostic={alignment_diagnostic}')
                        if not args.dry_run_rollout:
                            chunk_losses.append(
                                _global_mean_loss_contribution(
                                    metrics['loss'],
                                    query_batch_size=args.query_batch_size,
                                    group_size=args.group_size,
                                ))
                        metric_rows.append({key: float(value.detach().item()) for key, value in metrics.items()})
                    if chunk_losses:
                        with _cuda_profile_region(
                                args.profile_actor_phases,
                                actor_backward_events,
                        ):
                            torch.stack(chunk_losses).sum().backward()
                        with _cuda_profile_region(
                                args.profile_actor_phases,
                                actor_gradient_capture_events,
                        ):
                            # Check that no frozen/base parameter received a
                            # gradient before clearing the BF16 buffers.
                            _assert_frozen_parameters_have_no_grad(model)
                            if args.debug_micro_gradient_digests:
                                micro_family_norms = _gradient_family_norms(model)
                                micro_gradients = [
                                    torch.zeros_like(parameter, dtype=torch.float32)
                                    if parameter.grad is None else parameter.grad.detach().float()
                                    for parameter in trainable_parameters
                                ]
                                micro_gradient_diagnostics.append({
                                    'actor_chunk_start':
                                    chunk_start,
                                    'trajectory_keys': [(
                                        trajectory.case.row_index,
                                        trajectory.prompt_slot,
                                        trajectory.group_index,
                                    ) for trajectory in chunk],
                                    'advantages': [float(value) for value in chunk_advantages],
                                    'gradient_sha256':
                                    _exact_named_tensor_digest(trainable_names, micro_gradients),
                                    'grad_family_norms':
                                    micro_family_norms,
                                })
                                del micro_gradients
                            for accumulator, parameter in zip(
                                    local_high_precision_gradients,
                                    trainable_parameters,
                            ):
                                if parameter.grad is not None:
                                    accumulator.add_(parameter.grad.detach().double())
                            optimizer.zero_grad(set_to_none=True)
                    del replay_rows, isolated_replay_rows, chunk_losses
                    del actor_logps, rollout_logps, metrics, replay
                    clear_dual_ctx()
                    clear_step_mask_context()
            finally:
                train_template.remove_post_encode_hook()
                clear_dual_ctx()
                clear_step_mask_context()
            actor_replay_backward_elapsed_s = (_synchronized_time(args.device) - actor_start)
            actor_forward_elapsed_s = _cuda_profile_elapsed_s(actor_forward_events)
            actor_backward_elapsed_s = _cuda_profile_elapsed_s(actor_backward_events)
            actor_gradient_capture_elapsed_s = _cuda_profile_elapsed_s(actor_gradient_capture_events)
            actor_other_elapsed_s = max(
                0.0,
                actor_replay_backward_elapsed_s - actor_forward_elapsed_s - actor_backward_elapsed_s
                - actor_gradient_capture_elapsed_s,
            )
            _rank_print(
                args,
                f'[step {step + 1}] actor replay/backward complete '
                f't={actor_replay_backward_elapsed_s:.1f}s '
                f'forward={actor_forward_elapsed_s:.1f}s '
                f'backward={actor_backward_elapsed_s:.1f}s '
                f'grad_capture={actor_gradient_capture_elapsed_s:.1f}s '
                f'other={actor_other_elapsed_s:.1f}s; '
                f'entering gradient synchronization',
                all_ranks=True,
            )

            gradient_update_start = _synchronized_time(args.device)
            gradient_digest = None
            parameter_digest = None
            global_has_advantage = _global_advantage_flag(
                actor_advantages,
                device=args.device,
                distributed=args.distributed,
            )
            global_has_silent_rate_signal = _global_advantage_flag(
                [
                    row['silent_rate_loss_weighted']
                    for row in metric_rows
                ],
                device=args.device,
                distributed=args.distributed,
            )
            global_has_action_timing_signal = _global_advantage_flag(
                [row['action_timing_loss_weighted'] for row in metric_rows],
                device=args.device,
                distributed=args.distributed,
            )
            global_has_kl_signal = _global_advantage_flag(
                [row['kl_loss_weighted'] for row in metric_rows],
                device=args.device,
                distributed=args.distributed,
            )
            global_has_update_signal = (
                global_has_advantage
                or global_has_silent_rate_signal
                or global_has_action_timing_signal
                or global_has_kl_signal
            )
            optimizer_step_applied = False
            if args.dry_run_rollout:
                grad_norm = 0.0
                model_parameter_update_norm = 0.0
                grad_family_norms = {family: 0.0 for family in TRAINABLE_PARTS}
            else:
                synchronized_gradients = _synchronize_fp32_gradients(
                    trainable_parameters,
                    world_size=args.world_size,
                    local_gradients=local_high_precision_gradients,
                )
                gradient_digest = _exact_named_tensor_digest(trainable_names, synchronized_gradients)
                grad_family_norms = _gradient_family_norms_from_fp32(trainable_names, synchronized_gradients)
                if (global_has_update_signal and any(value <= 0 for value in grad_family_norms.values())):
                    missing_families = [family for family, value in grad_family_norms.items() if value <= 0]
                    raise RuntimeError('non-zero policy update signal did not reach every '
                                       'dual-LoRA family; refusing optimizer step: '
                                       f'{missing_families}')
                grad_norm = _clip_fp32_gradients(synchronized_gradients, args.max_grad_norm)
                optimizer_step_applied = _apply_optimizer_step(
                    optimizer,
                    synchronized_gradients,
                    global_has_advantage=global_has_update_signal,
                )
                model_parameter_update_norm = optimizer.last_model_update_norm
                del synchronized_gradients, local_high_precision_gradients
            optimizer.zero_grad(set_to_none=True)
            model.eval()
            completed_step = step + 1
            should_save = _checkpoint_due(args, completed_step)
            if _trainable_digest_due(args, completed_step):
                parameter_digest = _assert_exact_trainable_rank_equality(
                    trainable_names,
                    trainable_parameters,
                    world_size=args.world_size,
                )
            gradient_update_elapsed_s = (_synchronized_time(args.device) - gradient_update_start)

            rewards = [trajectory.reward for trajectory in trajectories]
            rollout_fingerprints = [
                trajectory_rollout_fingerprint(
                    trajectory,
                    include_trace=args.log_rollout_policy_trace,
                ) for trajectory in trajectories
            ]
            group_sampling, group_diagnostics = _group_sampling_diagnostics(
                trajectories,
                advantages,
                rollout_fingerprints,
            )
            actor_peak_allocated = torch.cuda.max_memory_allocated(torch.device(args.device))
            actor_peak_reserved = torch.cuda.max_memory_reserved(torch.device(args.device))
            elapsed = time.perf_counter() - step_start
            record = {
                'step':
                step + 1,
                'rank':
                args.rank,
                'world_size':
                args.world_size,
                'deterministic_algorithms':
                args.deterministic_algorithms,
                'run_contract':
                run_contract,
                'serialized_triton_autotune':
                autotune_runtime['enabled'],
                'triton_cache_dir':
                autotune_runtime['cache_dir'],
                'triton_autotune_lock':
                autotune_runtime['lock_path'],
                'triton_autotune_lock_scope': (autotune_runtime['lock_scope']),
                'global_has_advantage':
                global_has_advantage,
                'global_has_silent_rate_signal':
                global_has_silent_rate_signal,
                'global_has_action_timing_signal':
                global_has_action_timing_signal,
                'global_has_kl_signal':
                global_has_kl_signal,
                'global_has_update_signal':
                global_has_update_signal,
                'optimizer_step_applied':
                optimizer_step_applied,
                'actor_linear_mode':
                args.actor_linear_mode,
                'actor_replay_contract_version':
                args.actor_replay_contract_version,
                'gradient_checkpointing_runtime':
                checkpoint_runtime,
                'query_batch_size':
                args.query_batch_size,
                'local_query_batch_size':
                args.local_query_batch_size,
                'rank_sharding':
                args.rank_sharding,
                'batch_schedule':
                args.batch_schedule,
                'global_prompt_slots':
                local_prompt_slots,
                'selected_case_ids': [case.row_index for case in selected_cases],
                'selected_frame_counts':
                selected_frame_counts,
                'estimated_work_units':
                estimated_work_units,
                'rollout_physical_token_lengths':
                rollout_physical_token_lengths,
                'rollout_trajectory_microbatch_size': (args.rollout_trajectory_microbatch_size
                                                       or args.local_query_batch_size),
                'rollout_query_batch_mode':
                args.rollout_query_batch_mode,
                'frozen_visual_feature_reuse': {
                    **rollout_visual_reuse,
                    'actor_image_uses': sum(len(row.image_embeds) for row in actor_trajectories),
                },
                'actor_rank_rebalance':
                actor_rebalance,
                'actor_trajectory_microbatch_size':
                actor_batch_size,
                'actor_packed_token_budget':
                args.actor_packed_token_budget,
                'actor_microbatch_plan': [{
                    'trajectory_count': stop - start,
                    'physical_token_count': token_count,
                } for start, stop, token_count in actor_microbatch_plan],
                'actor_effective_microbatch_size':
                (len(actor_trajectories) / len(actor_microbatch_plan)),
                'actor_peak_packed_tokens':
                max(token_count for _, _, token_count in actor_microbatch_plan),
                'group_size':
                args.group_size,
                'trajectory_count':
                len(trajectories),
                'rollout_fingerprints':
                rollout_fingerprints,
                'global_trajectory_count': (args.query_batch_size * args.group_size),
                'data_source':
                data_source,
                'reward':
                _mean([row['total'] for row in rewards]),
                'reward_format':
                _mean([row['format'] for row in rewards]),
                'reward_timing':
                _mean([row['shaped_timing'] for row in rewards]),
                'window_reward_metrics':
                (_timing_metric_summary(rewards) if all('f1' in row for row in rewards) else None),
                'window_reward_metrics_by_source': {
                    source: _timing_metric_summary([
                        reward for trajectory, reward in zip(trajectories, rewards)
                        if trajectory.case.meta.get('source') == source
                    ])
                    for source in sorted({
                        trajectory.case.meta.get('source')
                        for trajectory in trajectories if trajectory.case.meta.get('source') is not None
                    })
                },
                'group_reward_std_nonzero':
                group_sampling['informative_group_count'] > 0,
                'group_sampling':
                group_sampling,
                'group_diagnostics':
                group_diagnostics,
                'loss':
                _mean([row['loss'] for row in metric_rows]),
                'policy_loss':
                _mean([row['policy_loss'] for row in metric_rows]),
                'kl':
                _mean([row['kl'] for row in metric_rows]),
                'kl_loss_weighted':
                _mean([row['kl_loss_weighted'] for row in metric_rows]),
                'silent_rate_loss':
                _mean([row['silent_rate_loss'] for row in metric_rows]),
                'silent_rate_loss_weighted':
                _mean([row['silent_rate_loss_weighted'] for row in metric_rows]),
                'action_outside_loss':
                _mean([row['action_outside_loss'] for row in metric_rows]),
                'action_outside_loss_weighted':
                _mean([row['action_outside_loss_weighted'] for row in metric_rows]),
                'action_coverage_loss':
                _mean([row['action_coverage_loss'] for row in metric_rows]),
                'action_coverage_loss_weighted':
                _mean([row['action_coverage_loss_weighted'] for row in metric_rows]),
                'action_count_loss':
                _mean([row['action_count_loss'] for row in metric_rows]),
                'action_count_loss_weighted':
                _mean([row['action_count_loss_weighted'] for row in metric_rows]),
                'action_timing_loss_weighted':
                _mean([row['action_timing_loss_weighted'] for row in metric_rows]),
                'action_expected_inside_speak_count':
                _mean([row['action_expected_inside_speak_count'] for row in metric_rows]),
                'action_expected_outside_speak_count':
                _mean([row['action_expected_outside_speak_count'] for row in metric_rows]),
                'action_mean_event_hit_probability':
                _mean([row['action_mean_event_hit_probability'] for row in metric_rows]),
                'expected_speak_count':
                _mean([row['expected_speak_count'] for row in metric_rows]),
                'target_speak_count':
                _mean([row['target_speak_count'] for row in metric_rows]),
                'expected_silent_rate':
                _mean([row['expected_silent_rate'] for row in metric_rows]),
                'target_silent_rate':
                _mean([row['target_silent_rate'] for row in metric_rows]),
                'ratio':
                _mean([row['ratio'] for row in metric_rows]),
                'clip_frac':
                _mean([row['clip_frac'] for row in metric_rows]),
                'max_logp_drift':
                max_logp_drift,
                'max_logp_drift_by_route':
                max_logp_drift_by_route,
                'max_logp_drift_by_kind':
                max_logp_drift_by_kind,
                'max_mean_logp_drift':
                max_mean_logp_drift,
                'actor_isolation_validation_enabled': (args.validate_actor_isolation),
                'max_actor_isolation_logp_drift': (max_actor_isolation_logp_drift),
                'max_mean_actor_isolation_logp_drift': (max_mean_actor_isolation_logp_drift),
                'same_weight_importance_ratio_range': [importance_ratio_min, importance_ratio_max],
                'policy_decisions_by_route':
                {route: sum(row['policy_decisions_by_route'][route] for row in alignment_rows)
                 for route in ('B', 'A')},
                'sampled_decisions':
                sum(row['sampled_decisions'] for row in alignment_rows),
                'policy_decisions':
                sum(row['policy_decisions'] for row in alignment_rows),
                'policy_decision_scope':
                args.policy_decision_scope,
                'loss_policy_decisions':
                sum(row['loss_policy_decisions'] for row in alignment_rows),
                'structural_decisions_excluded':
                sum(row['structural_decisions_excluded'] for row in alignment_rows),
                'loss_target_position_range': [
                    min(row['target_position_min'] for row in alignment_rows),
                    max(row['target_position_max'] for row in alignment_rows),
                ],
                'loss_prediction_position_range': [
                    min(row['prediction_position_min'] for row in alignment_rows),
                    max(row['prediction_position_max'] for row in alignment_rows),
                ],
                'packed_loss_target_position_range': [
                    min(row['packed_target_position_min'] for row in alignment_rows),
                    max(row['packed_target_position_max'] for row in alignment_rows),
                ],
                'packed_loss_prediction_position_range': [
                    min(row['packed_prediction_position_min'] for row in alignment_rows),
                    max(row['packed_prediction_position_max'] for row in alignment_rows),
                ],
                'actor_alignment':
                alignment_rows,
                'micro_gradient_diagnostics':
                (micro_gradient_diagnostics if args.debug_micro_gradient_digests else None),
                'triton_autotuner_selections':
                (_triton_autotuner_snapshot() if args.debug_micro_gradient_digests else None),
                'grad_norm':
                grad_norm,
                'model_parameter_update_norm':
                model_parameter_update_norm,
                'grad_family_norms':
                grad_family_norms,
                'pre_clip_gradient_sha256':
                gradient_digest,
                'post_update_trainable_sha256':
                parameter_digest,
                'max_physical_tokens':
                max_tokens_seen,
                'rollout_peak_allocated_mib': (rollout_peak_allocated / (1024**2)),
                'rollout_peak_delta_mib': ((rollout_peak_allocated - step_base_allocated) / (1024**2)),
                'rollout_peak_reserved_mib': (rollout_peak_reserved / (1024**2)),
                'rollout_peak_reserved_delta_mib': ((rollout_peak_reserved - step_base_reserved) / (1024**2)),
                'actor_peak_allocated_mib': (actor_peak_allocated / (1024**2)),
                'actor_peak_delta_mib': ((actor_peak_allocated - actor_base_allocated) / (1024**2)),
                'actor_peak_reserved_mib': (actor_peak_reserved / (1024**2)),
                'actor_peak_reserved_delta_mib': ((actor_peak_reserved - actor_base_reserved) / (1024**2)),
                'rollout_elapsed_s':
                rollout_elapsed_s,
                'reward_advantage_elapsed_s': (reward_advantage_elapsed_s),
                'actor_rebalance_elapsed_s':
                actor_rebalance_elapsed_s,
                'actor_replay_backward_elapsed_s': (actor_replay_backward_elapsed_s),
                'reference_replay_elapsed_s':
                reference_replay_elapsed_s,
                'actor_forward_elapsed_s':
                actor_forward_elapsed_s,
                'actor_backward_elapsed_s':
                actor_backward_elapsed_s,
                'actor_gradient_capture_elapsed_s':
                actor_gradient_capture_elapsed_s,
                'actor_other_elapsed_s':
                actor_other_elapsed_s,
                'gradient_update_elapsed_s':
                gradient_update_elapsed_s,
                'elapsed_s':
                elapsed,
                'dry_run_rollout':
                args.dry_run_rollout,
            }
            log_handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            log_handle.flush()
            if args.distributed:
                rank_records = [None] * args.world_size
                dist.all_gather_object(rank_records, record)
                aggregate_record = _aggregate_rank_records(rank_records)
                if args.rank == 0:
                    aggregate_log_handle.write(json.dumps(aggregate_record, ensure_ascii=False) + '\n')
                    aggregate_log_handle.flush()
            else:
                aggregate_record = record
            if args.rank == 0:
                swanlab_logger.log_train(aggregate_record)
                print(
                    f'[step {step + 1}] '
                    f'reward={aggregate_record["reward"]:.4f} '
                    f'loss={aggregate_record["loss"]:.4f} '
                    f'ratio={aggregate_record["ratio"]:.4f} '
                    f'logp_drift='
                    f'{aggregate_record["max_logp_drift"]:.6f} '
                    f'groups={aggregate_record["group_sampling"]["informative_group_count"]}/'
                    f'{aggregate_record["group_sampling"]["group_count"]} '
                    f'reward_std={aggregate_record["group_sampling"]["mean_reward_std"]:.4f} '
                    f'|adv|={aggregate_record["group_sampling"]["mean_abs_advantage"]:.4f} '
                    f'all_silent={aggregate_record["group_sampling"]["all_silent_group_count"]}/'
                    f'{aggregate_record["group_sampling"]["silent_observed_group_count"]} '
                    f'grad={aggregate_record["grad_norm"]:.4f} '
                    f'update={aggregate_record["model_parameter_update_norm"]:.4f} '
                    f'tokens<={aggregate_record["max_physical_tokens"]} '
                    f't={aggregate_record["elapsed_s"]:.1f}s',
                    flush=True,
                )

            if should_save:
                checkpoint_start = time.perf_counter()
                local_rng_state = _capture_rng_state()
                if args.distributed:
                    rng_states_by_rank = [None] * args.world_size
                    dist.all_gather_object(rng_states_by_rank, local_rng_state)
                else:
                    rng_states_by_rank = [local_rng_state]

                def save_rank_zero_checkpoint():
                    return save_checkpoint(
                        sm,
                        optimizer,
                        output_dir=output_dir,
                        step=step + 1,
                        args=args,
                        trainable_names=trainable_names,
                        rng_states_by_rank=rng_states_by_rank,
                    )

                checkpoint = _rank0_call_or_raise(
                    args,
                    f'save checkpoint-{step + 1}',
                    save_rank_zero_checkpoint,
                )
                if args.rank == 0:
                    print(
                        f'[dual-grpo] saved {checkpoint} '
                        f't={time.perf_counter() - checkpoint_start:.1f}s',
                        flush=True,
                    )

            del actor_trajectories, actor_advantages, trajectories, advantages
            _clear_image_processor_cache(sm)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            completed_step = step + 1
            validate_now = bool(val_cases) and (completed_step == args.max_steps or
                                                (args.eval_every > 0 and completed_step % args.eval_every == 0))
            if validate_now and not args.dry_run_rollout:
                summary = run_timing_validation(sm, val_cases, args=args, step=completed_step, output_dir=output_dir)
                if args.rank == 0:
                    swanlab_logger.log_validation(summary)
                    print(
                        f'[val step {completed_step}] '
                        f'mean_f1={summary["overall"]["mean_f1"]:.4f} '
                        f'macro_f1={summary["overall"]["macro_f1"]:.4f}',
                        flush=True)
            if args.dry_run_rollout:
                if args.rank == 0:
                    print('[dual-grpo] dry-run gate complete; ' 'no update performed')
                break

    if args.distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    try:
        main()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
