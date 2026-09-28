# Copyright (c) ModelScope Contributors. All rights reserved.
# Part of the implementation is borrowed from huggingface/transformers.
import inspect
import math
import os
import torch
from contextlib import contextmanager, nullcontext
from peft import PeftModel
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import Seq2SeqTrainer as HfSeq2SeqTrainer
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.utils import is_peft_available
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine
from swift.metrics import SumMetric
from swift.sequence_parallel import sequence_parallel
from swift.utils import HfConfigFactory, JsonlWriter, Serializer, gc_collect, get_logger, unwrap_model_for_generation
from .arguments import Seq2SeqTrainingArguments
from .mixin import DataLoaderMixin, SwiftMixin
from .utils import per_token_loss_func, per_token_loss_func_sp

logger = get_logger()

_DUAL_ACTION_MEAN_METRIC_KEYS = (
    'special/B/inner_recall',
    'special/B/inner_precision',
    'special/B/inner_fpr',
    'special/B/silent_acc',
    'special/B/illegal_top1_rate',
    'special/B/cross_speak_rate',
    'special/B/restricted_binary_acc',
    'special/B/restricted_binary_recall',
    'special/B/restricted_binary_precision',
    'special/B/restricted_binary_fpr',
    'special/B/restricted_binary_tie_rate',
    'special/B/inner_margin_on_inner',
    'special/B/inner_margin_on_silent',
    'special/A/speak_recall',
    'special/A/speak_precision',
    'special/A/speak_fpr',
    'special/A/silent_acc',
    'special/A/illegal_top1_rate',
    'special/A/cross_inner_rate',
    'special/A/restricted_binary_acc',
    'special/A/restricted_binary_recall',
    'special/A/restricted_binary_precision',
    'special/A/restricted_binary_fpr',
    'special/A/restricted_binary_tie_rate',
    'special/A/speak_margin_on_speak',
    'special/A/speak_margin_on_silent',
    'special/A/first_silent_fpr',
    'special/A/repeat_silent_fpr',
    'special/A/first_silent_margin',
    'special/A/repeat_silent_margin',
    'content/B/token_acc',
    'content/A/token_acc',
)

_DUAL_ACTION_COUNT_METRIC_KEYS = (
    'special/B/inner_count',
    'special/B/silent_count',
    'special/B/pred_inner_count',
    'special/B/restricted_pred_inner_count',
    'special/A/speak_count',
    'special/A/silent_count',
    'special/A/pred_speak_count',
    'special/A/restricted_pred_speak_count',
    'content/B/token_count',
    'content/A/token_count',
)


class Seq2SeqTrainer(SwiftMixin, DataLoaderMixin, HfSeq2SeqTrainer):
    args: Seq2SeqTrainingArguments

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = True  # fix transformers>=4.46.2
        if self.args.predict_with_generate:
            self.infer_engine = TransformersEngine(
                self.model, template=self.template, max_batch_size=self.args.per_device_eval_batch_size)
        self.jsonl_writer = JsonlWriter(os.path.join(self.args.output_dir, 'predict.jsonl'))
        self._init_dual_action_metrics()

    @staticmethod
    def _repeat_silent_focal_config() -> Optional[Tuple[float, float, int, int, Optional[float]]]:
        """Return the configuration for the opt-in repeat-silent loss.

        An unset, empty, or zero lambda preserves the historical loss path.
        The remaining variables are read only when the feature is active so
        unrelated stale variables cannot affect existing jobs.
        """
        raw_lambda = os.environ.get('REPEAT_SILENT_FOCAL_LAMBDA', '').strip()
        if not raw_lambda:
            return None
        try:
            focal_lambda = float(raw_lambda)
        except ValueError as exc:
            raise ValueError('REPEAT_SILENT_FOCAL_LAMBDA must be a finite non-negative float') from exc
        if not math.isfinite(focal_lambda) or focal_lambda < 0:
            raise ValueError('REPEAT_SILENT_FOCAL_LAMBDA must be a finite non-negative float')
        if focal_lambda == 0:
            return None

        raw_gamma = os.environ.get('REPEAT_SILENT_FOCAL_GAMMA', '2.0').strip()
        try:
            gamma = float(raw_gamma)
        except ValueError as exc:
            raise ValueError('REPEAT_SILENT_FOCAL_GAMMA must be a finite non-negative float') from exc
        if not math.isfinite(gamma) or gamma < 0:
            raise ValueError('REPEAT_SILENT_FOCAL_GAMMA must be a finite non-negative float')

        def non_negative_int(name: str) -> int:
            raw = os.environ.get(name, '0').strip()
            try:
                value = int(raw)
            except ValueError as exc:
                raise ValueError(f'{name} must be a non-negative integer') from exc
            if value < 0:
                raise ValueError(f'{name} must be a non-negative integer')
            return value

        delay_steps = non_negative_int('REPEAT_SILENT_FOCAL_DELAY_STEPS')
        ramp_steps = non_negative_int('REPEAT_SILENT_FOCAL_RAMP_STEPS')

        raw_min_prob = os.environ.get('REPEAT_SILENT_FOCAL_MIN_SPEAK_PROB', '').strip()
        min_speak_prob = None
        if raw_min_prob:
            try:
                min_speak_prob = float(raw_min_prob)
            except ValueError as exc:
                raise ValueError(
                    'REPEAT_SILENT_FOCAL_MIN_SPEAK_PROB must be finite and in [0, 1)') from exc
            if not math.isfinite(min_speak_prob) or not 0 <= min_speak_prob < 1:
                raise ValueError('REPEAT_SILENT_FOCAL_MIN_SPEAK_PROB must be finite and in [0, 1)')
        return focal_lambda, gamma, delay_steps, ramp_steps, min_speak_prob

    @staticmethod
    def _a_temporal_rank_config() -> Optional[Tuple[float, float, int, int]]:
        """Return the opt-in A-branch local action-ranking configuration.

        The auxiliary objective compares each ground-truth ``<speak>`` action
        with nearby ground-truth ``<silent>`` actions from the same packed
        sample.  An unset, empty, or zero lambda preserves the historical loss
        path and deliberately ignores the remaining variables.
        """
        raw_lambda = os.environ.get('A_TEMPORAL_RANK_LAMBDA', '').strip()
        if not raw_lambda:
            return None
        try:
            rank_lambda = float(raw_lambda)
        except ValueError as exc:
            raise ValueError('A_TEMPORAL_RANK_LAMBDA must be a finite non-negative float') from exc
        if not math.isfinite(rank_lambda) or rank_lambda < 0:
            raise ValueError('A_TEMPORAL_RANK_LAMBDA must be a finite non-negative float')
        if rank_lambda == 0:
            return None

        raw_margin = os.environ.get('A_TEMPORAL_RANK_MARGIN', '0.5').strip()
        try:
            margin = float(raw_margin)
        except ValueError as exc:
            raise ValueError('A_TEMPORAL_RANK_MARGIN must be a finite non-negative float') from exc
        if not math.isfinite(margin) or margin < 0:
            raise ValueError('A_TEMPORAL_RANK_MARGIN must be a finite non-negative float')

        def positive_int(name: str, default: str) -> int:
            raw = os.environ.get(name, default).strip()
            try:
                value = int(raw)
            except ValueError as exc:
                raise ValueError(f'{name} must be a positive integer') from exc
            if value <= 0:
                raise ValueError(f'{name} must be a positive integer')
            return value

        radius = positive_int('A_TEMPORAL_RANK_RADIUS', '4')
        hard_negatives = positive_int('A_TEMPORAL_RANK_HARD_NEGATIVES', '1')
        return rank_lambda, margin, radius, hard_negatives

    @staticmethod
    def _dual_action_loss_mode() -> str:
        """Return the opt-in dual action-token objective.

        ``restricted_binary`` defines the policy at the first B/A action token
        on the same legal support used by dual RL rollout: B chooses between
        ``<silent>`` and ``<inner>``, while A chooses between ``<silent>`` and
        ``<speak>``.  The historical full-vocabulary objective remains the
        default for every existing training entry point.
        """
        mode = os.environ.get('MSS_DUAL_ACTION_LOSS_MODE', 'full_vocab').strip().lower()
        allowed = {'full_vocab', 'restricted_binary'}
        if mode not in allowed:
            raise ValueError('MSS_DUAL_ACTION_LOSS_MODE must be one of '
                             f'{sorted(allowed)}, got {mode!r}')
        return mode

    def _get_special_weight_table(self, names, weights):
        cache_key = '_swt_' + ','.join(f'{n}:{w}' for n, w in zip(names, weights))
        if not hasattr(self, cache_key):
            tok = getattr(self.template.processor, 'tokenizer', self.template.processor)
            ids = [tok.convert_tokens_to_ids(t) for t in names]
            setattr(self, cache_key, (ids, list(weights)))
        return getattr(self, cache_key)

    def _get_dual_action_token_ids(self):
        cache_key = '_dual_action_token_ids'
        if not hasattr(self, cache_key):
            tok = getattr(self.template.processor, 'tokenizer', self.template.processor)
            names = set(self.args.special_token_names or [])
            if not {'<inner>', '<speak>', '<silent>'}.issubset(names):
                ids = {}
            else:
                ids = {
                    'inner': tok.convert_tokens_to_ids('<inner>'),
                    'speak': tok.convert_tokens_to_ids('<speak>'),
                    'silent': tok.convert_tokens_to_ids('<silent>'),
                }
                if any(v is None or v < 0 for v in ids.values()):
                    ids = {}
            setattr(self, cache_key, ids)
        return getattr(self, cache_key)

    def _get_dual_metric_structural_token_ids(self):
        cache_key = '_dual_metric_structural_token_ids'
        if not hasattr(self, cache_key):
            token_ids = self._get_dual_action_token_ids()
            if not token_ids:
                structural_ids = ()
            else:
                tok = getattr(self.template.processor, 'tokenizer', self.template.processor)
                structural_ids = set(token_ids.values())
                for token in ('</inner>', '</speak>', '<|im_end|>'):
                    token_id = tok.convert_tokens_to_ids(token)
                    if token_id is not None and token_id >= 0 and tok.convert_ids_to_tokens(token_id) == token:
                        structural_ids.add(int(token_id))
                structural_ids = tuple(sorted(structural_ids))
            setattr(self, cache_key, structural_ids)
        return getattr(self, cache_key)

    def _get_dual_metric_terminal_newline_id(self):
        cache_key = '_dual_metric_terminal_newline_id'
        if not hasattr(self, cache_key):
            tok = getattr(self.template.processor, 'tokenizer', self.template.processor)
            newline_ids = tok.encode('\n', add_special_tokens=False)
            newline_id = int(newline_ids[0]) if len(newline_ids) == 1 else None
            setattr(self, cache_key, newline_id)
        return getattr(self, cache_key)

    def _maybe_replace_dual_action_loss(
        self,
        token_loss: torch.Tensor,
        outputs,
        labels: torch.Tensor,
        logits_to_keep=None,
    ) -> torch.Tensor:
        """Replace only dual action rows with route-specific binary CE.

        All content and structural targets retain their full-vocabulary loss.
        Loss weights are deliberately applied by the caller *after* this
        replacement, so the existing special-token and FIRST_SILENT_WEIGHT
        semantics remain byte-for-byte identical.
        """
        if self._dual_action_loss_mode() == 'full_vocab':
            return token_loss

        from swift.template.step_mask.dual_lora import env_enabled, get_current_route_masks

        if not env_enabled():
            raise RuntimeError('restricted_binary dual action loss requires MSS_DUAL_BRANCH=1')
        if self.task_type != 'causal_lm':
            raise RuntimeError('restricted_binary dual action loss requires task_type=causal_lm')
        if self.template.sequence_parallel_size > 1:
            raise RuntimeError('restricted_binary dual action loss does not support sequence parallelism')
        if outputs.logits is None or torch.is_floating_point(labels):
            raise RuntimeError('restricted_binary dual action loss requires integer labels and vocabulary logits')

        token_ids = self._get_dual_action_token_ids()
        if not token_ids:
            raise RuntimeError('restricted_binary dual action loss requires '
                               '<inner>, <silent>, and <speak> in special_token_names')

        logits = outputs.logits
        expected_shape = logits.shape[:-1]
        if tuple(labels.shape) != tuple(expected_shape):
            raise RuntimeError('restricted_binary dual action loss label/logit shape mismatch: '
                               f'labels={tuple(labels.shape)} logits={tuple(logits.shape)}')
        if token_loss.numel() != labels.numel():
            raise RuntimeError('restricted_binary dual action loss token-loss shape mismatch: '
                               f'loss={tuple(token_loss.shape)} labels={tuple(labels.shape)}')

        shifted_labels = torch.full_like(labels, -100)
        shifted_labels[..., :-1] = labels[..., 1:]
        shifted_labels = shifted_labels.to(logits.device)
        route_A, route_B = get_current_route_masks()
        route_A = self._route_for_retained_logits(route_A, logits_to_keep, expected_shape, logits.device)
        route_B = self._route_for_retained_logits(route_B, logits_to_keep, expected_shape, logits.device)
        if route_A is None or route_B is None:
            raise RuntimeError('restricted_binary dual action loss route masks are unavailable or '
                               'do not align with retained logits')
        if (route_A & route_B).any():
            raise RuntimeError('restricted_binary dual action loss route_A and route_B overlap')

        inner_id = token_ids['inner']
        speak_id = token_ids['speak']
        silent_id = token_ids['silent']
        action_target = ((shifted_labels == inner_id) | (shifted_labels == speak_id)
                         | (shifted_labels == silent_id))
        b_slot = route_B & ((shifted_labels == inner_id) | (shifted_labels == silent_id))
        a_slot = route_A & ((shifted_labels == speak_id) | (shifted_labels == silent_id))
        unrouted = action_target & ~(a_slot | b_slot)
        wrong_branch = ((route_B & (shifted_labels == speak_id))
                        | (route_A & (shifted_labels == inner_id)))
        if unrouted.any() or wrong_branch.any():
            raise RuntimeError('restricted_binary dual action loss contract failed: '
                               f'unrouted_action_targets={int(unrouted.sum().item())}, '
                               f'wrong_branch_action_targets={int(wrong_branch.sum().item())}')
        if not (a_slot | b_slot).any():
            return token_loss

        replaced = token_loss.reshape(expected_shape).clone()

        def replace_branch(slot: torch.Tensor, positive_id: int) -> None:
            if not slot.any():
                return
            pair_logits = torch.stack(
                (logits[..., silent_id], logits[..., positive_id]),
                dim=-1,
            ).float()
            binary_target = (shifted_labels == positive_id).long()
            branch_loss = F.cross_entropy(pair_logits[slot], binary_target[slot], reduction='none')
            if self.args.enable_dft_loss:
                branch_loss = branch_loss * torch.exp(-branch_loss.detach())
            replaced[slot] = branch_loss.to(replaced.dtype)

        replace_branch(b_slot, inner_id)
        replace_branch(a_slot, speak_id)
        logger.info_once('[dual_action_loss] mode=restricted_binary '
                         'B=<silent>|<inner> A=<silent>|<speak>; '
                         'content_and_structure=full_vocab')
        return replaced.reshape_as(token_loss)

    def _init_dual_action_metrics(self) -> None:
        from swift.template.step_mask.dual_lora import env_enabled

        if not env_enabled() or self.task_type != 'causal_lm' or not self._get_dual_action_token_ids():
            return
        for mode in ('train', 'eval'):
            metrics = self.custom_metrics[mode]
            # Pre-register on every rank so distributed key synchronization
            # never mixes MeanMetric and SumMetric for the same count key.
            for key in _DUAL_ACTION_MEAN_METRIC_KEYS:
                metrics[key]
            for key in _DUAL_ACTION_COUNT_METRIC_KEYS:
                metrics[key] = SumMetric(device=self.args.device)
            if self._repeat_silent_focal_config() is not None:
                metrics['special/A/repeat_silent_focal_loss']
                metrics['special/A/repeat_silent_focal_active_fraction']
                metrics['special/A/repeat_silent_focal_effective_lambda']
                metrics['special/A/repeat_silent_focal_count'] = SumMetric(device=self.args.device)
                metrics['special/A/repeat_silent_focal_active_count'] = SumMetric(device=self.args.device)
            if self._a_temporal_rank_config() is not None:
                metrics['special/A/temporal_rank_loss']
                metrics['special/A/temporal_rank_violation_rate']
                metrics['special/A/temporal_rank_margin_gap']
                metrics['special/A/temporal_rank_positive_count'] = SumMetric(device=self.args.device)
                metrics['special/A/temporal_rank_pair_count'] = SumMetric(device=self.args.device)

    @staticmethod
    def _route_for_retained_logits(route, logits_to_keep, expected_shape, device):
        if route is None:
            return None
        if route.dim() == 1:
            route = route.unsqueeze(0)
        if route.dim() != len(expected_shape):
            return None

        if isinstance(logits_to_keep, torch.Tensor):
            selector = logits_to_keep.detach()
            if selector.dtype == torch.bool:
                selector = selector.reshape(-1).to(route.device)
                if selector.numel() != route.shape[-1]:
                    return None
                route = route[..., selector]
            elif selector.numel() == 1:
                keep = int(selector.item())
                route = route if keep == 0 else route[..., -keep:]
            else:
                selector = selector.reshape(-1).to(device=route.device, dtype=torch.long)
                route = route.index_select(-1, selector)
        elif isinstance(logits_to_keep, int):
            route = route if logits_to_keep == 0 else route[..., -logits_to_keep:]
        elif logits_to_keep is not None:
            return None

        if tuple(route.shape) != tuple(expected_shape):
            return None
        return route.to(device=device, dtype=torch.bool)

    @staticmethod
    def _sample_ids_for_retained_logits(position_ids, logits_to_keep, expected_shape, device):
        """Map retained logit rows to their original packed-sample ids."""
        if position_ids is None:
            # Non-packed batches are still safely separated by their batch row.
            return torch.arange(expected_shape[0], device=device).unsqueeze(-1).expand(expected_shape)
        pos = position_ids
        if pos.dim() == 3:
            pos = pos[-1]
        if pos.dim() != len(expected_shape):
            return None

        starts = pos == 0
        starts = starts.clone()
        starts[..., 0] = True
        sample_ids = starts.long().cumsum(dim=-1) - 1

        if isinstance(logits_to_keep, torch.Tensor):
            selector = logits_to_keep.detach()
            if selector.dtype == torch.bool:
                selector = selector.reshape(-1).to(sample_ids.device)
                if selector.numel() != sample_ids.shape[-1]:
                    return None
                sample_ids = sample_ids[..., selector]
            elif selector.numel() == 1:
                keep = int(selector.item())
                sample_ids = sample_ids if keep == 0 else sample_ids[..., -keep:]
            else:
                selector = selector.reshape(-1).to(device=sample_ids.device, dtype=torch.long)
                sample_ids = sample_ids.index_select(-1, selector)
        elif isinstance(logits_to_keep, int):
            sample_ids = sample_ids if logits_to_keep == 0 else sample_ids[..., -logits_to_keep:]
        elif logits_to_keep is not None:
            return None

        if tuple(sample_ids.shape) != tuple(expected_shape):
            return None
        # Make ids globally unique across batch rows.
        stride = sample_ids.shape[-1] + 1
        batch_offset = torch.arange(expected_shape[0], device=sample_ids.device).unsqueeze(-1) * stride
        return (sample_ids + batch_offset).to(device=device, dtype=torch.long)

    @staticmethod
    def _route_terminal_sources(route):
        """Mark the last routed source token in each contiguous branch span."""
        if route is None:
            return None
        next_is_routed = torch.zeros_like(route, dtype=torch.bool)
        next_is_routed[..., :-1] = route[..., 1:].to(dtype=torch.bool)
        return route.to(dtype=torch.bool) & ~next_is_routed

    @staticmethod
    def _update_mean_metric(metrics, key, values) -> None:
        if values.numel() > 0:
            metrics[key].update(values.detach().float())

    @staticmethod
    def _update_count_metric(metrics, key, mask) -> None:
        metrics[key].update(int(mask.sum().item()))

    def _get_assistant_role_id(self):
        cache_key = '_assistant_role_id'
        if not hasattr(self, cache_key):
            tok = getattr(self.template.processor, 'tokenizer', self.template.processor)
            asst_ids = tok.encode('assistant', add_special_tokens=False)
            if len(asst_ids) != 1:
                raise ValueError(f"expected 'assistant' to be one token, got {asst_ids}")
            setattr(self, cache_key, int(asst_ids[0]))
        return getattr(self, cache_key)

    @torch.no_grad()
    def _update_dual_action_token_metrics(
        self,
        outputs,
        labels,
        logits_to_keep=None,
        first_silent_source_mask=None,
    ) -> None:
        """Track branch-aware action decisions at the same logits used by loss.

        Branch B is evaluated only on its ``<inner>``/``<silent>`` decision
        positions and branch A only on its ``<speak>``/``<silent>`` positions.
        The branch identity comes from the exact route masks used by DualLoRA,
        not from the neighboring serialized branch tokens.

        When ``first_silent_source_mask`` is available, A-branch ground-truth
        silence is split into the first silence of a run and all later silence.
        Bucket FPR uses the full-vocabulary top-1 prediction; bucket margin is
        ``logit(<speak>) - logit(<silent>)``.
        """
        from swift.template.step_mask.dual_lora import env_enabled

        if not env_enabled() or self.task_type != 'causal_lm' or outputs.logits is None or labels is None:
            return
        if torch.is_floating_point(labels):
            return

        token_ids = self._get_dual_action_token_ids()
        if not token_ids:
            return

        logits = outputs.logits.detach()
        preds = logits.argmax(dim=-1)
        shifted_labels = torch.full_like(labels.detach(), -100)
        shifted_labels[..., :-1] = labels.detach()[..., 1:]
        shifted_labels = shifted_labels.to(preds.device)
        if preds.shape != shifted_labels.shape:
            return

        from swift.template.step_mask.dual_lora import get_current_route_masks

        route_A, route_B = get_current_route_masks()
        # Detect span ends before logits_to_keep removes non-contiguous source
        # positions. The A span's final source predicts its terminal newline.
        route_A_terminal = self._route_terminal_sources(route_A)
        route_A = self._route_for_retained_logits(route_A, logits_to_keep, preds.shape, preds.device)
        route_B = self._route_for_retained_logits(route_B, logits_to_keep, preds.shape, preds.device)
        route_A_terminal = self._route_for_retained_logits(route_A_terminal, logits_to_keep, preds.shape, preds.device)
        if route_A is None or route_B is None or route_A_terminal is None:
            logger.warning_once(
                '[dual_action_metrics] route masks are unavailable or do not align with retained logits; '
                'branch-aware metrics were skipped.')
            return
        if (route_A & route_B).any():
            raise RuntimeError('dual action metric contract failed: route_A and route_B overlap')

        inner_id = token_ids['inner']
        speak_id = token_ids['speak']
        silent_id = token_ids['silent']
        action_target = ((shifted_labels == inner_id) | (shifted_labels == speak_id) | (shifted_labels == silent_id))
        if not action_target.any():
            return

        b_slot = route_B & ((shifted_labels == inner_id) | (shifted_labels == silent_id))
        a_slot = route_A & ((shifted_labels == speak_id) | (shifted_labels == silent_id))
        unrouted = action_target & ~(b_slot | a_slot)
        wrong_branch = (route_B & (shifted_labels == speak_id)) | (route_A & (shifted_labels == inner_id))
        if unrouted.any() or wrong_branch.any():
            raise RuntimeError('dual action metric contract failed: '
                               f'unrouted_action_targets={int(unrouted.sum().item())}, '
                               f'wrong_branch_action_targets={int(wrong_branch.sum().item())}')

        retained_first_silent = None
        if first_silent_source_mask is not None:
            retained_first_silent = self._route_for_retained_logits(
                first_silent_source_mask,
                logits_to_keep,
                preds.shape,
                preds.device,
            )
            if retained_first_silent is None:
                logger.warning_once(
                    '[dual_action_metrics] first-silent mask does not align with retained logits; '
                    'first/repeat silent metrics were skipped.')

        mode = 'train' if self.model.training else 'eval'
        metrics = self.custom_metrics[mode]

        def update_branch(branch, slot, positive_name, positive_id, cross_name, cross_id, first_silent=None):
            prefix = f'special/{branch}'
            positive_target = slot & (shifted_labels == positive_id)
            silent_target = slot & (shifted_labels == silent_id)
            positive_prediction = slot & (preds == positive_id)
            legal_prediction = (preds == positive_id) | (preds == silent_id)

            self._update_mean_metric(metrics, f'{prefix}/{positive_name}_recall',
                                     (preds[positive_target] == positive_id))
            self._update_mean_metric(metrics, f'{prefix}/{positive_name}_precision',
                                     (shifted_labels[positive_prediction] == positive_id))
            self._update_mean_metric(metrics, f'{prefix}/{positive_name}_fpr', (preds[silent_target] == positive_id))
            self._update_mean_metric(metrics, f'{prefix}/silent_acc', (preds[silent_target] == silent_id))
            self._update_mean_metric(metrics, f'{prefix}/illegal_top1_rate', ~legal_prediction[slot])
            self._update_mean_metric(metrics, f'{prefix}/{cross_name}_rate', (preds[slot] == cross_id))

            positive_logits = logits[..., positive_id].float()
            silent_logits = logits[..., silent_id].float()
            margin = positive_logits - silent_logits
            # Ties are silence for both branches instead of depending on the
            # special tokens' vocabulary-id ordering.
            restricted_positive = positive_logits > silent_logits
            restricted_positive_prediction = slot & restricted_positive
            binary_tie = positive_logits == silent_logits
            restricted_correct = torch.where(shifted_labels == positive_id, restricted_positive, ~restricted_positive)
            self._update_mean_metric(metrics, f'{prefix}/restricted_binary_acc', restricted_correct[slot])
            self._update_mean_metric(metrics, f'{prefix}/restricted_binary_recall',
                                     restricted_positive[positive_target])
            self._update_mean_metric(metrics, f'{prefix}/restricted_binary_precision',
                                     shifted_labels[restricted_positive_prediction] == positive_id)
            self._update_mean_metric(metrics, f'{prefix}/restricted_binary_fpr', restricted_positive[silent_target])
            self._update_mean_metric(metrics, f'{prefix}/restricted_binary_tie_rate', binary_tie[slot])
            self._update_mean_metric(metrics, f'{prefix}/{positive_name}_margin_on_{positive_name}',
                                     margin[positive_target])
            self._update_mean_metric(metrics, f'{prefix}/{positive_name}_margin_on_silent', margin[silent_target])
            if first_silent is not None:
                marked_action = slot & first_silent
                if (marked_action & ~silent_target).any():
                    raise RuntimeError(
                        'dual action metric contract failed: '
                        f'branch {branch} first-silent mask marks a non-silent target')
                first_silent_target = silent_target & first_silent
                repeat_silent_target = silent_target & ~first_silent
                self._update_mean_metric(metrics, f'{prefix}/first_silent_fpr',
                                         preds[first_silent_target] == positive_id)
                self._update_mean_metric(metrics, f'{prefix}/repeat_silent_fpr',
                                         preds[repeat_silent_target] == positive_id)
                self._update_mean_metric(metrics, f'{prefix}/first_silent_margin', margin[first_silent_target])
                self._update_mean_metric(metrics, f'{prefix}/repeat_silent_margin', margin[repeat_silent_target])
            self._update_count_metric(metrics, f'{prefix}/{positive_name}_count', positive_target)
            self._update_count_metric(metrics, f'{prefix}/silent_count', silent_target)
            self._update_count_metric(metrics, f'{prefix}/pred_{positive_name}_count', positive_prediction)
            self._update_count_metric(metrics, f'{prefix}/restricted_pred_{positive_name}_count',
                                      restricted_positive_prediction)

        update_branch('B', b_slot, 'inner', inner_id, 'cross_speak', speak_id)
        update_branch(
            'A',
            a_slot,
            'speak',
            speak_id,
            'cross_inner',
            inner_id,
            first_silent=retained_first_silent,
        )

        structural_ids = self._get_dual_metric_structural_token_ids()
        structural = torch.tensor(structural_ids, device=shifted_labels.device, dtype=shifted_labels.dtype)
        structural_target = (shifted_labels[..., None] == structural).any(dim=-1)
        supervised = shifted_labels != -100
        newline_id = self._get_dual_metric_terminal_newline_id()
        if newline_id is None:
            a_terminal_newline = torch.zeros_like(route_A)
        else:
            a_terminal_newline = route_A_terminal & (shifted_labels == newline_id)
        b_content = route_B & supervised & ~structural_target
        a_content = route_A & supervised & ~structural_target & ~a_terminal_newline
        self._update_mean_metric(metrics, 'content/B/token_acc', preds[b_content] == shifted_labels[b_content])
        self._update_mean_metric(metrics, 'content/A/token_acc', preds[a_content] == shifted_labels[a_content])
        self._update_count_metric(metrics, 'content/B/token_count', b_content)
        self._update_count_metric(metrics, 'content/A/token_count', a_content)

    @staticmethod
    def _normalise_text_position_ids(position_ids, input_ids):
        pos = position_ids
        if pos is not None:
            if pos.dim() == 3:
                pos = pos[-1]
            if pos.shape != input_ids.shape:
                pos = None
        return pos

    @staticmethod
    def _sample_slices_from_position_row(pos_row, length: int):
        if pos_row is None:
            return [(0, length)]
        starts = (pos_row == 0).nonzero(as_tuple=True)[0].tolist()
        if not starts or starts[0] != 0:
            starts = [0] + starts
        starts = sorted(set(int(x) for x in starts if int(x) < length))
        starts.append(length)
        return [(starts[i], starts[i + 1]) for i in range(len(starts) - 1) if starts[i] < starts[i + 1]]

    @staticmethod
    def _find_branch_action_pos(row_ids, start: int, end: int, action_ids) -> Optional[int]:
        for pos in range(start, end):
            if int(row_ids[pos]) in action_ids:
                return pos
        return None

    def _override_first_silent_dual_branch(
        self,
        loss_scale,
        input_ids,
        labels,
        position_ids,
        ids,
        names,
        first_w,
        first_silent_mask=None,
    ) -> Optional[int]:
        """Dual-branch FIRST_SILENT_WEIGHT.

        In source-fork dual training, the A and B branches are independent
        action sequences:

            B: <inner> | <silent>
            A: <speak> | <silent>

        So a "first silent" run must be detected separately per branch. A's
        silent state must not be reset by B's <inner>, and B's silent state
        must not be reset by A's <speak>.
        """
        if os.environ.get('MSS_DUAL_BRANCH', '').strip() in ('', '0', 'false', 'False'):
            return None
        if first_silent_mask is not None and first_silent_mask.shape != loss_scale.shape:
            raise ValueError('first_silent_mask must have the same shape as loss_scale')
        required = {'<inner>', '<silent>', '<speak>'}
        if not required.issubset(set(names)):
            return None

        name_to_id = {name: int(tok_id) for name, tok_id in zip(names, ids)}
        inner_id = name_to_id['<inner>']
        silent_id = name_to_id['<silent>']
        speak_id = name_to_id['<speak>']
        b_action_ids = {inner_id, silent_id}
        a_action_ids = {speak_id, silent_id}

        try:
            assistant_role_id = self._get_assistant_role_id()
        except Exception:  # noqa: BLE001
            return None

        if labels is not None and labels.shape == input_ids.shape:
            label_mask = labels != -100
        else:
            label_mask = torch.ones_like(input_ids, dtype=torch.bool)

        pos = Seq2SeqTrainer._normalise_text_position_ids(position_ids, input_ids)
        pos_cpu = pos.cpu() if pos is not None else None
        ids_cpu = input_ids.detach().cpu()
        label_mask_cpu = label_mask.detach().cpu()

        flat_b, flat_p = [], []
        n_first_by_branch = {'A': 0, 'B': 0}
        saw_dual = False

        try:
            from swift.template.step_mask.meta import build_step_mask_meta
        except Exception:  # noqa: BLE001
            return None

        B, L = input_ids.shape
        for b in range(B):
            slices = self._sample_slices_from_position_row(None if pos_cpu is None else pos_cpu[b], L)
            for seg_start, seg_end in slices:
                row_ids = ids_cpu[b, seg_start:seg_end]
                try:
                    meta = build_step_mask_meta(
                        row_ids.numpy(),
                        assistant_role_id=assistant_role_id,
                        include_branch_source=None,
                    )
                except Exception:  # noqa: BLE001
                    continue
                if not meta.is_dual_branch:
                    continue
                saw_dual = True

                for branch_name, spans, action_ids in (
                    ('B', meta.b_spans, b_action_ids),
                    ('A', meta.a_spans, a_action_ids),
                ):
                    prev_silent = False
                    for span_start, span_end in spans:
                        local_pos = self._find_branch_action_pos(row_ids, int(span_start), int(span_end), action_ids)
                        if local_pos is None:
                            prev_silent = False
                            continue
                        global_pos = seg_start + local_pos
                        if not bool(label_mask_cpu[b, global_pos]):
                            continue

                        tok_id = int(ids_cpu[b, global_pos])
                        cur_silent = tok_id == silent_id
                        if cur_silent and not prev_silent:
                            flat_b.append(b)
                            flat_p.append(global_pos)
                            n_first_by_branch[branch_name] += 1
                        prev_silent = cur_silent

        if not saw_dual:
            return None

        if flat_b:
            bi = torch.tensor(flat_b, device=loss_scale.device, dtype=torch.long)
            pi = torch.tensor(flat_p, device=loss_scale.device, dtype=torch.long)
            loss_scale[bi, pi] = first_w
            if first_silent_mask is not None:
                first_silent_mask[bi, pi] = True

        if self is not None:
            self._last_first_silent_override_detail = {
                'mode': 'dual_branch',
                'A': n_first_by_branch['A'],
                'B': n_first_by_branch['B'],
                'total': len(flat_b),
            }
        return len(flat_b)

    def _override_first_silent(
        self,
        loss_scale,
        input_ids,
        labels,
        position_ids,
        ids,
        names,
        first_w,
        first_silent_mask=None,
    ):
        """Opt-in (gated by FIRST_SILENT_WEIGHT env var): override loss_scale
        at the first <silent> of a consecutive-silent action run.

        Dual source-fork data is handled branch-aware before this legacy path:
        A and B each keep their own silent-run state. The fallback below is for
        single-branch data, preserving the earlier behavior.

        Run boundary = the previous label-bearing assistant action is not <silent>
        (so an <inner> or <speak> in the same sample, or simply no prior action in
        this sample, opens a new silent run). Sample boundaries come from
        position_ids == 0 (padding_free packing). Inner/speak weights untouched.
        """
        if self is not None:
            self._last_first_silent_override_detail = {'mode': 'legacy', 'total': 0}
        if first_silent_mask is not None and first_silent_mask.shape != loss_scale.shape:
            raise ValueError('first_silent_mask must have the same shape as loss_scale')

        dual_n = None
        if self is not None:
            dual_n = self._override_first_silent_dual_branch(loss_scale, input_ids, labels, position_ids, ids, names,
                                                             first_w, first_silent_mask=first_silent_mask)
        if dual_n is not None:
            return dual_n

        silent_id = ids[names.index('<silent>')]
        if labels is not None and labels.shape == input_ids.shape:
            label_mask = labels != -100
        else:
            label_mask = torch.ones_like(input_ids, dtype=torch.bool)

        pos = Seq2SeqTrainer._normalise_text_position_ids(position_ids, input_ids)

        # Find label-bearing special-action positions, vectorized.
        is_special = torch.zeros_like(input_ids, dtype=torch.bool)
        for tid in ids:
            is_special |= (input_ids == tid)
        is_special &= label_mask

        # Pull to CPU once; per-sample walk is tiny vs. transfer cost.
        is_special_cpu = is_special.cpu()
        is_silent_cpu = (input_ids == silent_id).cpu() & is_special_cpu
        pos_cpu = pos.cpu() if pos is not None else None

        B, L = input_ids.shape
        flat_b, flat_p = [], []
        for b in range(B):
            if pos_cpu is not None:
                starts = (pos_cpu[b] == 0).nonzero(as_tuple=True)[0].tolist()
                if not starts or starts[0] != 0:
                    starts = [0] + starts
            else:
                starts = [0]
            starts.append(L)
            for si in range(len(starts) - 1):
                s, e = starts[si], starts[si + 1]
                if s >= e:
                    continue
                special_pos = is_special_cpu[b, s:e].nonzero(as_tuple=True)[0].tolist()
                prev_silent = False
                for p in special_pos:
                    cur_silent = bool(is_silent_cpu[b, s + p])
                    if cur_silent and not prev_silent:
                        flat_b.append(b)
                        flat_p.append(s + p)
                    prev_silent = cur_silent

        if flat_b:
            bi = torch.tensor(flat_b, device=loss_scale.device, dtype=torch.long)
            pi = torch.tensor(flat_p, device=loss_scale.device, dtype=torch.long)
            loss_scale[bi, pi] = first_w
            if first_silent_mask is not None:
                first_silent_mask[bi, pi] = True
        if self is not None:
            self._last_first_silent_override_detail = {'mode': 'legacy', 'total': len(flat_b)}
        return len(flat_b)

    def _maybe_build_special_token_loss_scale(self, inputs):
        names = self.args.special_token_names
        weights = self.args.special_token_weights
        if not names:
            return
        if inputs.get('loss_scale') is not None:
            return
        if 'input_ids' not in inputs or 'labels' not in inputs:
            return

        input_ids = inputs['input_ids']
        ids, ws = self._get_special_weight_table(tuple(names), tuple(weights))
        loss_scale = torch.ones_like(input_ids, dtype=torch.float32)
        for tok_id, w in zip(ids, ws):
            loss_scale = torch.where(
                input_ids == tok_id,
                torch.full_like(loss_scale, w),
                loss_scale,
            )

        # Opt-in: FIRST_SILENT_WEIGHT env var overrides the first <silent> of each
        # consecutive-silent run. In dual source-fork data this is branch-aware:
        # A and B runs are tracked independently. Unset/empty → legacy weights.
        fsw_env = os.environ.get('FIRST_SILENT_WEIGHT')
        if self._repeat_silent_focal_config() is not None and not fsw_env:
            raise ValueError('REPEAT_SILENT_FOCAL_LAMBDA>0 requires FIRST_SILENT_WEIGHT so '
                             'first and repeat silent positions are defined explicitly')
        first_w = None
        n_first = 0
        first_silent_target_mask = None
        if fsw_env and '<silent>' in names:
            first_w = float(fsw_env)
            pos = inputs.get('text_position_ids')
            if pos is None:
                pos = inputs.get('position_ids')
            first_silent_target_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            n_first = self._override_first_silent(
                loss_scale,
                input_ids,
                inputs.get('labels'),
                pos,
                ids,
                names,
                first_w,
                first_silent_mask=first_silent_target_mask,
            )

        inputs['loss_scale'] = loss_scale
        if first_silent_target_mask is not None:
            # Metrics consume logits at the source position that predicts each
            # target token. Keep this mask in the original source coordinate
            # space so logits_to_keep can apply the exact same source selector.
            first_silent_source_mask = torch.zeros_like(first_silent_target_mask)
            first_silent_source_mask[..., :-1] = first_silent_target_mask[..., 1:]
            inputs['_dual_first_silent_source_mask'] = first_silent_source_mask

        if first_w is None:
            # Legacy log line, byte-identical to pre-FIRST_SILENT_WEIGHT, so existing
            # dashboards / grep on stream_trans*.sh runs are unaffected.
            logger.info_once(f'[special_token] applied (pre-truncate): '
                             f'nnz_above_1={(loss_scale > 1.0).sum().item()}, '
                             f'nnz_below_1={(loss_scale < 1.0).sum().item()}, '
                             f'mean={loss_scale.mean().item():.4f}')
        else:
            # Bucket dump — only when FIRST_SILENT_WEIGHT is active. Counts come from
            # input_ids and sum to loss_scale.numel() exactly.
            detail = getattr(self, '_last_first_silent_override_detail', {})
            branch_detail = ''
            if detail.get('mode') == 'dual_branch':
                branch_detail = (f" first_silent_A={int(detail.get('A', 0))}"
                                 f" first_silent_B={int(detail.get('B', 0))}")
            total = input_ids.numel()
            lines = [
                f'[loss_scale buckets] total={total} mean={loss_scale.mean().item():.4f} '
                f'first_silent_override={n_first}{branch_detail}'
            ]
            n_special = 0
            for nm, tid, w in zip(names, ids, ws):
                cnt = int((input_ids == tid).sum().item())
                n_special += cnt
                if nm == '<silent>' and n_first > 0:
                    lines.append(f'  silent (other)   w={w:<5g} count={cnt - n_first}')
                    lines.append(f'  silent (first)   w={first_w:<5g} count={n_first}')
                else:
                    lines.append(f'  {nm:<15s} w={w:<5g} count={cnt}')
            lines.append(f'  non-special      w=1     count={total - n_special}')
            logger.info_once('\n'.join(lines))

    def _repeat_silent_focal_loss(
        self,
        outputs,
        labels,
        logits_to_keep,
        first_silent_source_mask,
        gamma: float,
        min_speak_prob: Optional[float],
    ) -> Optional[torch.Tensor]:
        """Mean negative-class focal loss on A-branch repeat silence only.

        For each selected target, the binary probability is formed from only
        ``<silent>`` and ``<speak>`` logits. First silence, genuine speak,
        branch B, and every content/structural token are excluded.
        """
        from swift.template.step_mask.dual_lora import env_enabled, get_current_route_masks

        if not env_enabled():
            raise RuntimeError('repeat-silent focal loss is only valid with MSS_DUAL_BRANCH=1')
        if outputs.logits is None:
            raise RuntimeError('repeat-silent focal loss requires model logits')
        if first_silent_source_mask is None:
            raise RuntimeError('repeat-silent focal loss requires the first-silent source mask')

        token_ids = self._get_dual_action_token_ids()
        if not token_ids:
            raise RuntimeError('repeat-silent focal loss requires <inner>/<silent>/<speak> token ids')

        logits = outputs.logits
        shifted_labels = torch.full_like(labels, -100)
        shifted_labels[..., :-1] = labels[..., 1:]
        shifted_labels = shifted_labels.to(logits.device)
        if tuple(logits.shape[:-1]) != tuple(shifted_labels.shape):
            raise RuntimeError('repeat-silent focal labels do not align with retained logits')

        route_a, _ = get_current_route_masks()
        route_a = self._route_for_retained_logits(
            route_a,
            logits_to_keep,
            shifted_labels.shape,
            logits.device,
        )
        first_silent = self._route_for_retained_logits(
            first_silent_source_mask,
            logits_to_keep,
            shifted_labels.shape,
            logits.device,
        )
        if route_a is None or first_silent is None:
            raise RuntimeError('repeat-silent focal masks do not align with retained logits')

        silent_target = route_a & (shifted_labels == token_ids['silent'])
        first_silent_a = route_a & first_silent
        if (first_silent_a & ~silent_target).any():
            raise RuntimeError('first-silent mask marks a non-silent A-branch target')
        repeat_silent = silent_target & ~first_silent_a
        count = int(repeat_silent.sum().item())
        mode = 'train' if self.model.training else 'eval'
        metrics = self.custom_metrics[mode]
        metrics['special/A/repeat_silent_focal_count'].update(count)
        if count == 0:
            return None

        binary_logits = torch.stack(
            (logits[..., token_ids['silent']], logits[..., token_ids['speak']]),
            dim=-1,
        ).float()
        log_probs = F.log_softmax(binary_logits, dim=-1)
        speak_prob = log_probs[..., 1].exp()
        focal = speak_prob.pow(gamma) * (-log_probs[..., 0])
        selected = focal[repeat_silent]
        if min_speak_prob is None:
            active = torch.ones_like(selected, dtype=torch.bool)
        else:
            # The gate is a hard example selector, not a differentiable part
            # of the loss. Normalizing over every repeat-silent target makes
            # the auxiliary term naturally decay as false positives vanish.
            active = speak_prob.detach()[repeat_silent] > min_speak_prob
            selected = torch.where(active, selected, torch.zeros_like(selected))

        metrics['special/A/repeat_silent_focal_loss'].update(selected.detach())
        metrics['special/A/repeat_silent_focal_active_fraction'].update(active.detach())
        metrics['special/A/repeat_silent_focal_active_count'].update(int(active.sum().item()))
        return selected.mean()

    def _repeat_silent_focal_effective_lambda(
        self,
        focal_lambda: float,
        delay_steps: int,
        ramp_steps: int,
    ) -> float:
        step = int(getattr(getattr(self, 'state', None), 'global_step', 0))
        if step < delay_steps:
            return 0.0
        if ramp_steps == 0:
            return focal_lambda
        progress = min(max((step - delay_steps) / ramp_steps, 0.0), 1.0)
        return focal_lambda * progress

    def _maybe_add_repeat_silent_focal_loss(
        self,
        loss,
        outputs,
        labels,
        logits_to_keep,
        first_silent_source_mask,
    ):
        """Add the opt-in auxiliary term, preserving the exact off path."""
        config = self._repeat_silent_focal_config()
        if config is None:
            return loss
        if labels is None:
            raise RuntimeError('repeat-silent focal loss requires labels')
        focal_lambda, gamma, delay_steps, ramp_steps, min_speak_prob = config
        focal_loss = self._repeat_silent_focal_loss(
            outputs,
            labels,
            logits_to_keep,
            first_silent_source_mask,
            gamma,
            min_speak_prob,
        )
        if focal_loss is None:
            return loss
        effective_lambda = self._repeat_silent_focal_effective_lambda(
            focal_lambda,
            delay_steps,
            ramp_steps,
        )
        mode = 'train' if self.model.training else 'eval'
        self.custom_metrics[mode]['special/A/repeat_silent_focal_effective_lambda'].update(
            torch.tensor(effective_lambda, device=loss.device))
        logger.info_once('[repeat_silent_focal] enabled: '
                         f'lambda={focal_lambda:g} gamma={gamma:g} delay_steps={delay_steps} '
                         f'ramp_steps={ramp_steps} min_speak_prob={min_speak_prob} '
                         'normalization=mean_repeat_silent')
        return loss + effective_lambda * focal_loss.to(loss.device)

    def _a_temporal_rank_loss(
        self,
        outputs,
        labels,
        logits_to_keep,
        position_ids,
        margin: float,
        radius: int,
        hard_negatives: int,
    ) -> Optional[torch.Tensor]:
        """Rank A ``<speak>`` above nearby A ``<silent>`` decisions.

        Distances are measured in A action decisions, not raw tokens.  Each
        positive uses up to ``hard_negatives`` highest-margin silent actions
        within ``radius`` decisions on either side.  Pair losses are averaged
        per positive before the batch mean, preventing positives with more
        available neighbors from receiving extra weight.
        """
        from swift.template.step_mask.dual_lora import env_enabled, get_current_route_masks

        if not env_enabled():
            raise RuntimeError('A temporal ranking loss is only valid with MSS_DUAL_BRANCH=1')
        if outputs.logits is None:
            raise RuntimeError('A temporal ranking loss requires model logits')

        token_ids = self._get_dual_action_token_ids()
        if not token_ids:
            raise RuntimeError('A temporal ranking loss requires <inner>/<silent>/<speak> token ids')

        logits = outputs.logits
        expected_shape = logits.shape[:-1]
        shifted_labels = torch.full_like(labels, -100)
        shifted_labels[..., :-1] = labels[..., 1:]
        shifted_labels = shifted_labels.to(logits.device)
        if tuple(shifted_labels.shape) != tuple(expected_shape):
            raise RuntimeError('A temporal ranking labels do not align with retained logits')

        route_a, _ = get_current_route_masks()
        route_a = self._route_for_retained_logits(route_a, logits_to_keep, expected_shape, logits.device)
        sample_ids = self._sample_ids_for_retained_logits(
            position_ids,
            logits_to_keep,
            expected_shape,
            logits.device,
        )
        if route_a is None or sample_ids is None:
            raise RuntimeError('A temporal ranking route/sample masks do not align with retained logits')

        speak_id = token_ids['speak']
        silent_id = token_ids['silent']
        a_action = route_a & ((shifted_labels == speak_id) | (shifted_labels == silent_id))
        action_margin = (logits[..., speak_id] - logits[..., silent_id]).float()
        per_positive_losses = []
        selected_gaps = []
        selected_violations = []
        eligible_positive_count = 0
        pair_count = 0

        for sample_id in sample_ids[a_action].unique():
            sample_action = a_action & (sample_ids == sample_id)
            flat_action_idx = sample_action.reshape(-1).nonzero(as_tuple=True)[0]
            if flat_action_idx.numel() == 0:
                continue
            flat_labels = shifted_labels.reshape(-1)[flat_action_idx]
            positive_order = (flat_labels == speak_id).nonzero(as_tuple=True)[0]
            negative_order = (flat_labels == silent_id).nonzero(as_tuple=True)[0]
            if positive_order.numel() == 0 or negative_order.numel() == 0:
                continue

            flat_margin = action_margin.reshape(-1)[flat_action_idx]
            positive_margin = flat_margin[positive_order]
            negative_margin = flat_margin[negative_order]
            distance = (positive_order[:, None] - negative_order[None, :]).abs()
            valid = distance <= radius
            candidate_scores = negative_margin.detach().unsqueeze(0).expand_as(valid).masked_fill(~valid, -torch.inf)
            topk = min(hard_negatives, negative_order.numel())
            _, hard_index = candidate_scores.topk(topk, dim=-1)
            selected_valid = valid.gather(1, hard_index)
            if not selected_valid.any():
                continue

            selected_negative = negative_margin[hard_index]
            pair_score = margin + selected_negative - positive_margin.unsqueeze(-1)
            pair_loss = F.softplus(pair_score)
            per_positive_pair_count = selected_valid.sum(dim=-1)
            eligible_positive = per_positive_pair_count > 0
            per_positive_loss = (pair_loss * selected_valid).sum(dim=-1) / per_positive_pair_count.clamp_min(1)
            per_positive_losses.append(per_positive_loss[eligible_positive])
            selected_gaps.append(
                (positive_margin.unsqueeze(-1) - selected_negative)[selected_valid])
            selected_violations.append((pair_score > 0)[selected_valid])
            eligible_positive_count += int(eligible_positive.sum().item())
            pair_count += int(selected_valid.sum().item())

        mode = 'train' if self.model.training else 'eval'
        metrics = self.custom_metrics[mode]
        metrics['special/A/temporal_rank_positive_count'].update(eligible_positive_count)
        metrics['special/A/temporal_rank_pair_count'].update(pair_count)
        if not per_positive_losses:
            return None

        per_positive_loss = torch.cat(per_positive_losses)
        gaps = torch.cat(selected_gaps)
        violations = torch.cat(selected_violations)
        metrics['special/A/temporal_rank_loss'].update(per_positive_loss.detach())
        metrics['special/A/temporal_rank_margin_gap'].update(gaps.detach())
        metrics['special/A/temporal_rank_violation_rate'].update(violations.detach())
        return per_positive_loss.mean()

    def _maybe_add_a_temporal_rank_loss(
        self,
        loss,
        outputs,
        labels,
        logits_to_keep,
        position_ids,
    ):
        """Add the opt-in local ranking term, preserving the exact off path."""
        config = self._a_temporal_rank_config()
        if config is None:
            return loss
        if labels is None:
            raise RuntimeError('A temporal ranking loss requires labels')
        rank_lambda, margin, radius, hard_negatives = config
        rank_loss = self._a_temporal_rank_loss(
            outputs,
            labels,
            logits_to_keep,
            position_ids,
            margin,
            radius,
            hard_negatives,
        )
        if rank_loss is None:
            return loss
        logger.info_once('[a_temporal_rank] enabled: '
                         f'lambda={rank_lambda:g} margin={margin:g} radius={radius} '
                         f'hard_negatives={hard_negatives} normalization=mean_per_positive')
        return loss + rank_lambda * rank_loss.to(loss.device)

    @staticmethod
    def _predict_data_collator(batch):
        return {'_data': batch}

    @contextmanager
    def _patch_predict_with_generate(self):
        origin_data_collator = self.data_collator
        self.data_collator = self._predict_data_collator
        packing = self.template.packing
        padding_free = self.template.padding_free
        self.template.packing = False
        self.template.padding_free = False
        try:
            yield
        finally:
            self.template.packing = packing
            self.template.padding_free = padding_free
            self.data_collator = origin_data_collator

    def evaluate(self, *args, **kwargs):
        context = self._patch_predict_with_generate() if self.args.predict_with_generate else nullcontext()
        with context:
            res = super().evaluate(*args, **kwargs)
            gc_collect()
            return res

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.args.predict_with_generate or prediction_loss_only:
            with self.template.forward_context(self.model, inputs):
                return super().prediction_step(
                    model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys)
        data_list = inputs['_data']
        labels_list = [InferRequest.remove_response(data['messages']) for data in data_list]
        with unwrap_model_for_generation(
                self.model_wrapped, self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation), self.template.generate_context():
            resp_list = self.infer_engine.infer(
                data_list,
                RequestConfig(max_tokens=self.model.generation_config.max_new_tokens),
                use_tqdm=False,
            )

        response_list = []
        jsonl_cache = []
        device = self.args.device
        for data, resp, labels in zip(data_list, resp_list, labels_list):
            response = resp.choices[0].message.content
            jsonl_cache.append({'response': response, 'labels': labels, **data})
            response_list.append(Serializer.to_tensor(resp.choices[0].message.content).to(device=device))
        self.jsonl_writer.append(jsonl_cache, gather_obj=True)
        labels_list = [Serializer.to_tensor(labels).to(device=device) for labels in labels_list]
        response_list = pad_sequence(response_list, batch_first=True, padding_value=0)
        labels_list = pad_sequence(labels_list, batch_first=True, padding_value=0)
        return None, response_list, labels_list

    def _prepare_inputs(self, inputs):
        args = self.args
        inputs = super()._prepare_inputs(inputs)
        if self.template.sequence_parallel_size > 1:
            sequence_parallel.prepare_inputs(inputs)

        # Build special_token_weights loss_scale BEFORE prepare_logits_to_keep so
        # the latter filters loss_scale by the same mask it applies to labels.
        self._maybe_build_special_token_loss_scale(inputs)

        use_logits_to_keep = self.get_use_logits_to_keep(self.template.sequence_parallel_size == 1)
        if use_logits_to_keep:
            self.prepare_logits_to_keep(inputs)
            if args.tuner_backend == 'unsloth' and isinstance(inputs['logits_to_keep'], torch.Tensor):
                inputs['logits_to_keep'] = int(inputs['logits_to_keep'].sum())

        base_model = self.template.get_base_model(self.model)
        if self.model.model_info.is_moe_model and 'output_router_logits' in inspect.signature(
                base_model.forward).parameters:
            HfConfigFactory.set_config_attr(base_model.config, 'router_aux_loss_coef', args.router_aux_loss_coef)
            base_model.router_aux_loss_coef = args.router_aux_loss_coef
            logger.info_once(f'router_aux_loss_coef: {args.router_aux_loss_coef}')
            if args.router_aux_loss_coef > 0:
                inputs['output_router_logits'] = True
        inputs['compute_loss_func'] = self.compute_loss_func
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = None
        compute_loss_func: Callable = inputs.pop('compute_loss_func', None)
        loss_scale = inputs.pop('loss_scale', None)
        first_silent_source_mask = inputs.pop('_dual_first_silent_source_mask', None)
        dual_action_loss_mode = self._dual_action_loss_mode()
        a_temporal_rank_config = self._a_temporal_rank_config()
        if dual_action_loss_mode == 'restricted_binary':
            if compute_loss_func is not None:
                raise RuntimeError('restricted_binary dual action loss is incompatible with compute_loss_func')
            if self.label_smoother is not None:
                raise RuntimeError('restricted_binary dual action loss is incompatible with label smoothing')
        names = self.args.special_token_names
        weights = self.args.special_token_weights
        if (names and loss_scale is None and 'input_ids' in inputs and 'labels' in inputs):
            input_ids = inputs['input_ids']
            ids, ws = self._get_special_weight_table(tuple(names), tuple(weights))
            loss_scale = torch.ones_like(input_ids, dtype=torch.float32)
            for tok_id, w in zip(ids, ws):
                loss_scale = torch.where(
                    input_ids == tok_id,
                    torch.full_like(loss_scale, w),
                    loss_scale,
                )
            logger.info_once(f'[special_token] applied: nnz_above_1={(loss_scale > 1.0).sum().item()}, '
                             f'nnz_below_1={(loss_scale < 1.0).sum().item()}, '
                             f'mean={loss_scale.mean().item():.4f}, '
                             f'max={loss_scale.max().item():.4f}, min={loss_scale.min().item():.4f}')
        text_position_ids = inputs.pop('text_position_ids', None)
        if text_position_ids is None:
            text_position_ids = inputs.get('position_ids')
        channels = inputs.pop('channel', None)

        if (self.label_smoother is not None or compute_loss_func is not None or loss_scale is not None
                or dual_action_loss_mode == 'restricted_binary'
                or a_temporal_rank_config is not None
                or self.args.enable_dft_loss or self.args.enable_channel_loss
                or self.template.sequence_parallel_size > 1) and 'labels' in inputs:
            if self.args.use_liger_kernel:
                logger.warning_once('The cross_entropy loss function defined in Liger Kernel will not '
                                    'take effect, potentially leading to increased GPU memory consumption.')
            labels = inputs.pop('labels')
        outputs = model(**inputs)
        if getattr(outputs, 'aux_loss', None) is not None:
            mode = 'train' if self.model.training else 'eval'
            self.custom_metrics[mode]['aux_loss'].update(outputs.aux_loss)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if hasattr(self.args, 'past_index') and self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is None:
            labels = inputs['labels']
            outputs.loss = outputs.loss.to(labels.device)
            # fix https://github.com/huggingface/transformers/issues/34263
            if num_items_in_batch is not None:
                outputs.loss = outputs.loss * ((labels[:, 1:] != -100).sum() / num_items_in_batch)

            if isinstance(outputs, dict) and 'loss' not in outputs:
                raise ValueError(
                    'The model did not return a loss from the inputs, only the following keys: '
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}.")
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
        else:
            outputs.loss = None
            if (self.args.enable_dft_loss or loss_scale is not None or self.args.enable_channel_loss
                    or dual_action_loss_mode == 'restricted_binary'
                    or a_temporal_rank_config is not None
                    or self.template.sequence_parallel_size > 1):
                if self.template.sequence_parallel_size > 1:
                    outputs.loss = per_token_loss_func_sp(outputs, labels, enable_dft_loss=self.args.enable_dft_loss)
                else:
                    outputs.loss = per_token_loss_func(outputs, labels, enable_dft_loss=self.args.enable_dft_loss)

                outputs.loss = self._maybe_replace_dual_action_loss(
                    outputs.loss,
                    outputs,
                    labels,
                    inputs.get('logits_to_keep'),
                )

                if loss_scale is not None:
                    loss_scale = torch.roll(loss_scale, shifts=-1, dims=-1).view(-1)
                    # DEBUG: verify post-filter loss_scale is actually multiplied
                    _pre_loss_mean = outputs.loss.detach().float().mean().item()
                    outputs.loss = outputs.loss * loss_scale
                    _post_loss_mean = outputs.loss.detach().float().mean().item()
                    _ls = loss_scale.detach()
                    logger.info_once(f'[special_token] post-filter loss_scale: len={_ls.numel()}, '
                                     f'nnz_above_1={(_ls > 1.0).sum().item()}, '
                                     f'nnz_below_1={(_ls < 1.0).sum().item() - (_ls == 0).sum().item()}, '
                                     f'nnz_eq_0={(_ls == 0).sum().item()}, '
                                     f'mean_ls={_ls.mean().item():.4f} | '
                                     f'loss_mean: before={_pre_loss_mean:.4f} after={_post_loss_mean:.4f} '
                                     f'ratio={_post_loss_mean / max(_pre_loss_mean, 1e-9):.4f}')

                if self.args.enable_channel_loss:
                    mode = 'train' if self.model.training else 'eval'
                    metrics = self.custom_metrics[mode]
                    masks = torch.roll(labels, shifts=-1, dims=-1).view(-1) != -100
                    if self.template.padding_free:
                        cu_seqlens = self.get_cu_seqlens(text_position_ids, inputs.get('logits_to_keep'))
                    else:
                        cu_seqlens = torch.arange(0, labels.shape[0] + 1) * labels.shape[1]
                    for i in range(cu_seqlens.shape[0] - 1):
                        channel = None if channels is None else channels[i]
                        slice_ = slice(cu_seqlens[i], cu_seqlens[i + 1])
                        metrics[f'loss_{channel}'].update(outputs.loss[slice_][masks[slice_]])

            unwrapped_model = self.accelerator.unwrap_model(model)
            if is_peft_available() and isinstance(unwrapped_model, PeftModel):
                model_name = unwrapped_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if compute_loss_func is not None:
                loss = compute_loss_func(
                    outputs, labels, num_items_in_batch=num_items_in_batch, loss_scale=loss_scale, trainer=self)
            elif self.label_smoother is None:
                # Handle the outputs.loss generated by loss_scale.
                if num_items_in_batch is None:
                    num_items_in_batch = (labels[:, 1:] != -100).sum()
                loss = outputs.loss.sum() / num_items_in_batch
            else:
                if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                    loss = self.label_smoother(outputs, labels, shift_labels=True)
                else:
                    loss = self.label_smoother(outputs, labels)

            if self.model.model_info.is_moe_model and self.args.router_aux_loss_coef is not None:
                aux_loss = outputs.get('aux_loss')
                if aux_loss is not None:
                    if num_items_in_batch is not None:
                        aux_loss = aux_loss * ((labels[:, 1:] != -100).sum() / num_items_in_batch)
                    loss = loss + self.args.router_aux_loss_coef * aux_loss.to(loss.device)

        if getattr(self.args, 'average_tokens_across_devices',
                   False) and self.model_accepts_loss_kwargs and num_items_in_batch is not None:
            loss *= self.accelerator.num_processes

        loss = self._maybe_add_repeat_silent_focal_loss(
            loss,
            outputs,
            labels,
            inputs.get('logits_to_keep'),
            first_silent_source_mask,
        )
        loss = self._maybe_add_a_temporal_rank_loss(
            loss,
            outputs,
            labels,
            inputs.get('logits_to_keep'),
            text_position_ids,
        )

        if outputs.logits is not None and labels is not None:
            self._update_dual_action_token_metrics(
                outputs,
                labels,
                inputs.get('logits_to_keep'),
                first_silent_source_mask=first_silent_source_mask,
            )

        if (outputs.logits is not None and labels is not None and self.args.tuner_backend != 'unsloth'):
            cu_seqlens = None
            if self.template.padding_free and self.args.acc_strategy == 'seq':
                cu_seqlens = self.get_cu_seqlens(text_position_ids, inputs.get('logits_to_keep'))
            # Liger does not have logits
            # Unsloth has a bug with output logits
            self._compute_acc(outputs, labels, cu_seqlens=cu_seqlens)
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, *args, **kwargs):
        with self.template.forward_context(self.model, inputs):
            return super().training_step(model, inputs, *args, **kwargs)
