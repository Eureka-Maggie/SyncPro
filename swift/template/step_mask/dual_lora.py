"""Dual-LoRA FFN for parallel branch routing (inner vs. speak).

Each transformer decoder layer's MLP has three Linear layers (gate_proj,
up_proj, down_proj forming a SwiGLU FFN). Under dual-branch training we wrap
each of these Linears with a :class:`DualLoraLinear` that carries TWO
independent LoRA adapters (A and B) alongside the frozen base weight, and
routes them per-token based on masks stashed in a contextvar:

* ``LoRA_A`` — active on tokens in the ``<speak>`` branch span (``route_A``)
* ``LoRA_B`` — active on tokens in the ``<inner>`` branch span (``route_B``)
* Neither  — everything else goes through the base weight only

The route masks are computed once per forward from :class:`StepMaskMeta`
and stored in :data:`_dual_lora_ctx`. All 32 * 3 = 96 DualLoraLinears in
a Qwen3.5-4B model read from the same contextvar.

Activation is gated by ``MSS_DUAL_BRANCH`` env var (1 = on). LoRA rank /
alpha / dropout come from ``MSS_LORA_RANK`` / ``MSS_LORA_ALPHA`` /
``MSS_LORA_DROPOUT`` env vars.
"""

from __future__ import annotations

import logging
import math
import os
import torch
import torch.nn as nn
from typing import Optional

log = logging.getLogger(__name__)

# Env-var toggles.
_ENV_ENABLE = 'MSS_DUAL_BRANCH'
_ENV_RANK = 'MSS_LORA_RANK'
_ENV_ALPHA = 'MSS_LORA_ALPHA'
_ENV_DROPOUT = 'MSS_LORA_DROPOUT'
_ENV_DEBUG = 'MSS_DUAL_BRANCH_DEBUG'

# One-shot debug counter: prints per-forward dispatch summary for the first
# N calls, then goes silent. Meant to make it easy to eyeball that LoRA_A/B
# are actually being applied on the right tokens.
_DEBUG_MAX_CALLS = 5
_debug_call_count = 0
_position_debug_call_count = 0


def _debug_enabled() -> bool:
    return os.environ.get(_ENV_DEBUG, '').strip() not in ('', '0', 'false', 'False')


def env_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, '').strip() not in ('', '0', 'false', 'False')


def env_rank() -> int:
    raw = os.environ.get(_ENV_RANK, '64').strip()
    return int(raw) if raw else 64


def env_alpha() -> float:
    raw = os.environ.get(_ENV_ALPHA, '128').strip()
    return float(raw) if raw else 128.0


def env_dropout() -> float:
    raw = os.environ.get(_ENV_DROPOUT, '0.05').strip()
    return float(raw) if raw else 0.05


# Module-level dict carrying the current forward's route masks. Populated by
# the ``pre_forward_hook`` wrapper and read by every DualLoraLinear.forward().
#
# Values (populated when dual-branch is active):
#   {
#       'route_A': torch.BoolTensor [L]  or  [B, L]  (True at A-span tokens)
#       'route_B': torch.BoolTensor [L]  or  [B, L]
#   }
# Empty dict when we're outside a dual-branch forward.
#
# NOTE: uses a plain Python dict rather than contextvars.ContextVar because
# torch.utils.checkpoint does NOT preserve contextvars across the recompute
# boundary -- if the ctx var is cleared/re-populated between forward and
# backward's recomputed forward, the dual-branch code path diverges and
# CheckpointError fires ("A different number of tensors was saved"). The
# module-level dict survives across all Python-level state changes within
# a single training step, resolving that mismatch. For multi-threaded
# scenarios the trainer is single-thread here (DDP puts each rank in its
# own process); if we ever need multi-threaded inference, add a per-thread
# dict keyed by thread id.
_dual_lora_ctx: dict = {}


def get_current_route_masks():
    """Return the A/B route masks used by the current model forward."""
    return _dual_lora_ctx.get('route_A'), _dual_lora_ctx.get('route_B')


class DualLoraLinear(nn.Module):
    """A frozen Linear wrapped with two independent LoRA branches (A and B).

    Forward:
        out = base(x)
        if any route_A token: out += lora_A(x) * route_A_mask
        if any route_B token: out += lora_B(x) * route_B_mask

    ``route_A``/``route_B`` are pulled from :data:`_dual_lora_ctx`. When the
    contextvar is empty (or both routes have zero True entries), the module
    is a bit-identity to the frozen base Linear -- this is the code path we
    fall through on for inference prefill (image tokens) or for legacy
    single-branch data.
    """

    def __init__(self, base_linear: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base_linear,
                          nn.Linear), (f'DualLoraLinear expects nn.Linear, got {type(base_linear).__name__}')
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rank = rank
        self.scaling = alpha / rank

        # Frozen base weight. We keep the module reference (rather than the
        # raw weight tensor) so that any pre-existing hooks / dtypes /
        # device placement carry through.
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        # LoRA_A: down (in_features -> rank) then up (rank -> out_features).
        # Standard PEFT LoRA init: down = kaiming, up = zeros, so the initial
        # LoRA contribution is exactly zero and the module is a no-op at init.
        base_dtype = base_linear.weight.dtype
        base_device = base_linear.weight.device
        self.lora_A_down = nn.Linear(self.in_features, rank, bias=False, dtype=base_dtype, device=base_device)
        self.lora_A_up = nn.Linear(rank, self.out_features, bias=False, dtype=base_dtype, device=base_device)
        self.lora_B_down = nn.Linear(self.in_features, rank, bias=False, dtype=base_dtype, device=base_device)
        self.lora_B_up = nn.Linear(rank, self.out_features, bias=False, dtype=base_dtype, device=base_device)
        nn.init.kaiming_uniform_(self.lora_A_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_A_up.weight)
        nn.init.kaiming_uniform_(self.lora_B_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_up.weight)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _apply_lora(self, x, down, up):
        """Standard LoRA: up @ down @ dropout(x), scaled."""
        return up(down(self.dropout(x))) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)

        if not _dual_lora_ctx:
            return base_out

        route_A = _dual_lora_ctx.get('route_A')  # None, [L], or [B, L]
        route_B = _dual_lora_ctx.get('route_B')

        # If neither mask has any True token, skip both LoRA branches
        # entirely -- big savings during e.g. image-only prefill.
        # The route tensors are shared by all 96 wrapped projections in one
        # model forward.  Re-running ``any().item()`` here would introduce two
        # device-to-host synchronizations per projection (192 sync points per
        # token in Qwen3.5-4B).  Producers therefore cache these two Python
        # booleans once per forward.  Keep the tensor fallback for older test
        # and third-party callers that populate only route_A/route_B.
        cached_has_A = _dual_lora_ctx.get('has_A')
        cached_has_B = _dual_lora_ctx.get('has_B')
        sync_presence = bool(_dual_lora_ctx.get('sync_route_presence_reference', False))
        has_A = (
            bool(cached_has_A)
            if cached_has_A is not None and not sync_presence else route_A is not None and bool(route_A.any().item()))
        has_B = (
            bool(cached_has_B)
            if cached_has_B is not None and not sync_presence else route_B is not None and bool(route_B.any().item()))
        if not has_A and not has_B:
            return base_out

        # Broadcast route mask to base_out shape.
        # base_out: [B, L, out_features] (typical) or [L, out_features] (bs=1
        # padding_free packed with squeezed batch). Handle both.
        def _broadcast(mask):
            if mask is None:
                return None
            if base_out.dim() == 3:
                if mask.dim() == 1:
                    m = mask.view(1, -1, 1)
                elif mask.dim() == 2:
                    m = mask.view(mask.size(0), mask.size(1), 1)
                else:
                    raise ValueError(f'unexpected route mask ndim: {mask.dim()}')
            elif base_out.dim() == 2:
                if mask.dim() == 1:
                    m = mask.view(-1, 1)
                elif mask.dim() == 2:
                    assert mask.size(0) == 1, (f'2D base_out expects [L, D] but mask is [B={mask.size(0)}, L]; '
                                               'either squeeze batch first or use 3D base_out.')
                    m = mask[0].view(-1, 1)
                else:
                    raise ValueError(f'unexpected route mask ndim: {mask.dim()}')
            else:
                raise ValueError(f'unexpected base_out ndim: {base_out.dim()}')
            return m.to(base_out.dtype)

        if has_A:
            a_out = self._apply_lora(x, self.lora_A_down, self.lora_A_up)
            mask_A = _broadcast(route_A)
            base_out = base_out + a_out * mask_A
        if has_B:
            b_out = self._apply_lora(x, self.lora_B_down, self.lora_B_up)
            mask_B = _broadcast(route_B)
            base_out = base_out + b_out * mask_B

        return base_out

    def extra_repr(self) -> str:
        return (f'in={self.in_features}, out={self.out_features}, '
                f'rank={self.rank}, scaling={self.scaling:.3f}')


def _find_text_mlp_modules(model):
    """Iterate over (parent, attr_name, mlp_module) for every text-side MLP
    that should receive DualLoraLinear injection.

    We deliberately match the class name ``Qwen3_5MLP`` and exclude
    ``Qwen3_5VisionMLP`` so vision encoder FFNs stay frozen (they aren't
    LoRA'd in the single-branch setup either -- ``--freeze_vit true`` is on).
    """
    # For each named_module, we need to know its parent to be able to
    # ``setattr(parent, name, new_module)``.
    for parent_name, parent_module in model.named_modules():
        for child_name, child_module in parent_module.named_children():
            cls_name = type(child_module).__name__
            if cls_name == 'Qwen3_5MLP':
                yield parent_module, child_name, child_module


_dual_lora_patched: dict = {'installed': False}


def inject_dual_lora_ffn(model, *, rank: int, alpha: float, dropout: float) -> int:
    """Walk the model and replace each text-side MLP's three Linears with
    DualLoraLinear wrappers. Returns the number of MLPs replaced.

    Idempotent: safe to call multiple times.
    """
    if _dual_lora_patched.get('installed'):
        log.warning('[dual_lora] already injected, skipping')
        return 0

    n_replaced = 0
    for parent, child_name, mlp in _find_text_mlp_modules(model):
        # Wrap gate_proj / up_proj / down_proj.
        for linear_name in ('gate_proj', 'up_proj', 'down_proj'):
            base_linear = getattr(mlp, linear_name)
            wrapped = DualLoraLinear(base_linear, rank=rank, alpha=alpha, dropout=dropout)
            setattr(mlp, linear_name, wrapped)
        n_replaced += 1

    _dual_lora_patched.update(installed=True, n_replaced=n_replaced)
    log.info(f'[dual_lora] injected DualLoraLinear into {n_replaced} MLPs '
             f'(rank={rank}, alpha={alpha}, dropout={dropout}); '
             f'total LoRA params ~= {n_replaced * 3 * 2} adapters')
    return n_replaced


def compute_route_masks(input_ids, assistant_role_id: int, device):
    """Run build_step_mask_meta on input_ids and return (route_A, route_B)
    torch bool tensors on ``device``.

    ``input_ids`` shape: [B, L] or [L]. Return shape matches input.
    """
    from .meta import build_step_mask_meta

    if input_ids.dim() == 2:
        # Batched: compute per-sample meta and stack.
        route_A_list, route_B_list = [], []
        for b in range(input_ids.size(0)):
            ids_np = input_ids[b].detach().cpu().numpy()
            meta = build_step_mask_meta(ids_np, assistant_role_id=assistant_role_id)
            route_A_list.append(torch.from_numpy(meta.route_A))
            route_B_list.append(torch.from_numpy(meta.route_B))
        route_A = torch.stack(route_A_list, dim=0).to(device=device, dtype=torch.bool)
        route_B = torch.stack(route_B_list, dim=0).to(device=device, dtype=torch.bool)
    elif input_ids.dim() == 1:
        ids_np = input_ids.detach().cpu().numpy()
        meta = build_step_mask_meta(ids_np, assistant_role_id=assistant_role_id)
        route_A = torch.from_numpy(meta.route_A).to(device=device, dtype=torch.bool)
        route_B = torch.from_numpy(meta.route_B).to(device=device, dtype=torch.bool)
    else:
        raise ValueError(f'input_ids must be 1D or 2D, got {input_ids.dim()}D')

    return route_A, route_B


def set_ctx_from_input_ids(input_ids, assistant_role_id: int, cu_seqlens=None) -> Optional[dict]:
    """Populate branch routes and packed-sample boundaries for one forward."""
    from .linear_fork import set_linear_fork_ctx
    from .meta import build_step_mask_meta

    global _debug_call_count
    device = input_ids.device

    if input_ids.dim() == 2:
        if input_ids.size(0) != 1:
            raise ValueError(f'dual-branch padding-free input must have physical batch size 1, '
                             f'got {tuple(input_ids.shape)}')
        ids_np = input_ids[0].detach().cpu().numpy()
    elif input_ids.dim() == 1:
        ids_np = input_ids.detach().cpu().numpy()
    else:
        raise ValueError(f'input_ids must be 1D or 2D, got {input_ids.dim()}D')

    meta = build_step_mask_meta(ids_np, assistant_role_id=assistant_role_id)
    if not meta.is_dual_branch:
        # LoRA routes are empty, but linear recurrence must still reset at
        # every packed logical sample.
        _dual_lora_ctx.clear()
        set_linear_fork_ctx(
            frame_starts=[],
            b_spans=[],
            a_spans=[],
            cu_seqlens=cu_seqlens,
            seq_len=input_ids.shape[-1],
            device=device,
        )
        return _dual_lora_ctx

    route_A = torch.from_numpy(meta.route_A).to(device=device, dtype=torch.bool)
    route_B = torch.from_numpy(meta.route_B).to(device=device, dtype=torch.bool)
    if input_ids.dim() == 2:
        route_A = route_A.unsqueeze(0)
        route_B = route_B.unsqueeze(0)
    # Metadata is still on CPU here; count routes before copying it to CUDA so
    # setup itself does not add a device synchronization.
    n_A = int(meta.route_A.sum())
    n_B = int(meta.route_B.sum())

    # In-place updates survive gradient-checkpoint recomputation.
    _dual_lora_ctx.clear()
    _dual_lora_ctx['route_A'] = route_A
    _dual_lora_ctx['route_B'] = route_B
    _dual_lora_ctx['has_A'] = n_A > 0
    _dual_lora_ctx['has_B'] = n_B > 0
    set_linear_fork_ctx(
        frame_starts=meta.frame_start_idx,
        b_spans=meta.b_spans,
        a_spans=meta.a_spans,
        cu_seqlens=cu_seqlens,
        seq_len=input_ids.shape[-1],
        device=device,
    )

    if _debug_enabled() and _debug_call_count < _DEBUG_MAX_CALLS:
        rank = int(os.environ.get('RANK', '0') or 0)
        if rank == 0:
            length = input_ids.shape[-1]
            n_samples = (int(cu_seqlens.numel() - 1) if isinstance(cu_seqlens, torch.Tensor) else 1)
            print(
                f'[dual_lora][call#{_debug_call_count}] '
                f'input_ids.shape={tuple(input_ids.shape)} '
                f'n_samples={n_samples} n_route_A={n_A} n_route_B={n_B} '
                f'ratio_A={n_A / length:.3f} ratio_B={n_B / length:.3f} '
                f'n_frames={len(meta.frame_start_idx)} '
                f'n_b_spans={len(meta.b_spans)} n_a_spans={len(meta.a_spans)}',
                flush=True,
            )
        _debug_call_count += 1

    return _dual_lora_ctx


def clear_ctx() -> None:
    from .linear_fork import clear_linear_fork_ctx
    _dual_lora_ctx.clear()
    clear_linear_fork_ctx()


_hook_patched: dict = {'installed': False}


def patch_pre_forward_hook_for_dual_lora(assistant_role_id: int) -> None:
    """Wrap swift's ``Template.pre_forward_hook`` so that on every forward,
    :func:`set_ctx_from_input_ids` fires to populate the dual_lora contextvar
    from the current batch's ``input_ids``.

    Idempotent. Chains with any pre-existing wrapper (e.g., step_mask's).
    """
    if _hook_patched.get('installed'):
        return

    from swift.template import base as _tpl_base
    from .position_overlap import apply_overlap_to_position_ids_tensor
    from .position_overlap import env_enabled as pos_overlap_enabled

    original_hook = _tpl_base.Template.pre_forward_hook

    def wrapped(self, model, args, kwargs):
        global _position_debug_call_count
        input_ids = kwargs.get('input_ids')
        if input_ids is None:
            clear_ctx()
            return original_hook(self, model, args, kwargs)

        try:
            set_ctx_from_input_ids(
                input_ids,
                assistant_role_id=assistant_role_id,
                cu_seqlens=kwargs.get('cu_seq_lens_q'),
            )
            # Optional position overlap keeps B/A branch positions parallel.
            if pos_overlap_enabled():
                from .linear_fork import _linear_fork_ctx

                lf = _linear_fork_ctx
                if lf and lf.get('b_spans') and kwargs.get('position_ids') is not None:
                    position_ids_before = kwargs['position_ids']
                    position_ids_after = apply_overlap_to_position_ids_tensor(
                        position_ids_before,
                        lf['b_spans'],
                        lf['a_spans'],
                        sample_spans=lf['sample_spans'],
                    )
                    kwargs['position_ids'] = position_ids_after
                    if (_debug_enabled() and _position_debug_call_count < _DEBUG_MAX_CALLS
                            and int(os.environ.get('RANK', '0') or 0) == 0 and position_ids_after.dim() == 3
                            and position_ids_after.shape[0] in (3, 4)):
                        mrope_start = 0 if position_ids_after.shape[0] == 3 else 1
                        before = position_ids_before[mrope_start:]
                        after = position_ids_after[mrope_start:]
                        shift = before - after
                        all_axis_shift = bool(torch.equal(shift[0], shift[1]) and torch.equal(shift[0], shift[2]))
                        b_starts = torch.tensor(
                            [span[0] for span in lf['b_spans']],
                            device=after.device,
                        )
                        a_starts = torch.tensor(
                            [span[0] for span in lf['a_spans']],
                            device=after.device,
                        )
                        branch_starts_equal = bool(torch.equal(after[:, 0, b_starts], after[:, 0, a_starts]))

                        b_start, b_end = lf['b_spans'][0]
                        a_start, a_end = lf['a_spans'][0]
                        sample_end = next(end for start, end in lf['sample_spans'] if start <= b_start < end)
                        next_after = None
                        next_expected = None
                        next_ok = True
                        if a_end < sample_end:
                            next_after_t = after[:, 0, a_end]
                            next_expected_t = (after[:, 0, b_start] + max(b_end - b_start, a_end - a_start))
                            next_ok = bool(torch.equal(next_after_t, next_expected_t))
                            next_after = next_after_t.detach().cpu().tolist()
                            next_expected = next_expected_t.detach().cpu().tolist()

                        if not (all_axis_shift and branch_starts_equal and next_ok):
                            raise RuntimeError('three-axis MRoPE overlap contract failed: '
                                               f'all_axis_shift={all_axis_shift} '
                                               f'branch_starts_equal={branch_starts_equal} '
                                               f'next_ok={next_ok}')
                        print(
                            f'[position_overlap][call#{_position_debug_call_count}] '
                            f'shape={tuple(position_ids_after.shape)} '
                            f'all_axis_shift={all_axis_shift} '
                            f'branch_starts_equal={branch_starts_equal} '
                            f'next_ok={next_ok} '
                            f'B_before={before[:, 0, b_start].detach().cpu().tolist()} '
                            f'A_before={before[:, 0, a_start].detach().cpu().tolist()} '
                            f'B_after={after[:, 0, b_start].detach().cpu().tolist()} '
                            f'A_after={after[:, 0, a_start].detach().cpu().tolist()} '
                            f'next_after={next_after} next_expected={next_expected}',
                            flush=True,
                        )
                        _position_debug_call_count += 1
        except Exception as exc:
            clear_ctx()
            raise RuntimeError('dual-branch metadata or packed-boundary construction failed') from exc
        return original_hook(self, model, args, kwargs)

    _tpl_base.Template.pre_forward_hook = wrapped
    _hook_patched.update(installed=True, original_hook=original_hook)
    log.info('[dual_lora] wrapped Template.pre_forward_hook to populate route masks')


def enable_dual_lora(model,
                     assistant_role_id: int,
                     *,
                     rank: Optional[int] = None,
                     alpha: Optional[float] = None,
                     dropout: Optional[float] = None,
                     inject_linear_fork: bool = True) -> None:
    """Idempotent one-shot activation: read env-var config, inject DualLoraFFN
    into every text MLP, wire the pre_forward_hook to populate the route
    contextvar, and (unless disabled) install the linear-attn fork wrapper
    so per-frame fork+merge runs during forward.

    Call once, right after the model is loaded.
    """
    if not env_enabled():
        return
    if _dual_lora_patched.get('installed'):
        return

    rank = rank if rank is not None else env_rank()
    alpha = alpha if alpha is not None else env_alpha()
    dropout = dropout if dropout is not None else env_dropout()

    n_ffn = inject_dual_lora_ffn(model, rank=rank, alpha=alpha, dropout=dropout)
    patch_pre_forward_hook_for_dual_lora(assistant_role_id=assistant_role_id)

    n_linear = 0
    if inject_linear_fork:
        from .linear_fork import get_merge_mode, inject_dual_branch_linear_attn
        merge_mode = get_merge_mode()
        n_linear = inject_dual_branch_linear_attn(model, merge_mode=merge_mode)

    log.info(f'[dual_lora] enabled: injected into {n_ffn} MLPs '
             f'(rank={rank}, alpha={alpha}, dropout={dropout}); '
             f'linear-attn fork patched on {n_linear} layers')
