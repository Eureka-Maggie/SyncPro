"""Streaming proactive-video inference with the step+window attention mask.

Drop-in alternative to ``infer_stream_vllm_v2.py`` for models trained with
``STEP_MASK_WINDOW`` enabled. The mask cannot run under vLLM (vLLM's
PagedAttention doesn't accept arbitrary masks), so this script drives the
model directly via HF transformers with flex_attention and reuses the KV
cache across frames.

Per-frame flow:

  1. Re-encode the full conversation via the swift template; compare against
     the running KV cache state to compute the delta tokens (the new turn's
     tokens + the assistant prefix).
  2. Forward through the model with ``past_key_values=`` from the prior
     frame. Our patched ``create_causal_mask`` consults the streaming state
     to build a BlockMask of shape (q_len = new tokens, kv_len = full cache).
  3. Sample greedy until ``<|im_end|>``. Each decoded token is appended to
     the state and fed back as the next forward's input.
  4. After the response is closed, optionally **physically evict** the K/V
     entries for visual tokens that the step+window rule now masks out from
     every future query (so KV memory stays bounded across long streams).

Usage mirrors ``infer_stream_vllm_v2.py``:

    python infer_stream_masked.py --jsonl_idx 0 \\
        --model /path/to/merged_checkpoint \\
        --window 30

Set ``--no_evict`` to disable physical KV eviction (mask still applied, just
no memory reclaim — useful for A/B testing).
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import os
import re
import sys
import time
import torch
from pathlib import Path
from PIL import Image
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from swift import get_template
from swift.template.step_mask import SPECIAL_TOKEN_IDS, enable_flex_attn_support, patch_qwen3_5_for_step_mask
from swift.template.step_mask.inference import StreamingStepMaskState, reset_ctx, streaming_ctx_token

DEFAULT_MODEL = ''  # pass --model <checkpoint dir>
DEFAULT_JSONL = ''  # no bundled demo data; pass --from_jsonl
DEFAULT_SYSTEM_PROMPT_PATH = str(REPO_ROOT / 'data/system_prompt.txt')

_IM_END = SPECIAL_TOKEN_IDS['im_end']
_NEWLINE_TOK = 198  # '\n' in the Qwen3 tokenizer


def load_case_from_jsonl(jsonl_path: str, idx: int) -> Tuple[str, List[str], dict]:
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if i == idx:
                d = json.loads(line)
                first_user = d['messages'][0]['content']
                q = first_user.replace('<image>', '').replace('Query:', '', 1).strip()
                meta = {
                    'video_id': d.get('video_id'),
                    'data_source': d.get('data_source'),
                    'n_frames': len(d['images']),
                }
                return q, d['images'], meta
    raise IndexError(f'jsonl_idx {idx} out of range in {jsonl_path}')


def classify_response(response: str) -> str:
    if '<silent>' in response:
        return 'SILENT'
    parts = []
    if '<inner>' in response:
        parts.append('INNER')
    if '<speak>' in response:
        parts.append('SPEAK')
    return '+'.join(parts) if parts else 'UNKNOWN'


class MaskedStreamingRunner:
    """Owns the model + KV cache + streaming mask state for one video."""

    def __init__(self,
                 model,
                 tokenizer,
                 window_frames: int,
                 assistant_role_id: int,
                 device,
                 evict: bool = True,
                 max_response_tokens: int = 512,
                 use_streaming_cache: bool = False,
                 max_kv_len: int = 8192,
                 use_graph_decode: bool = False,
                 use_graph_evict: bool = False,
                 frame_callback=None,
                 sampler=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.evict = evict
        self.max_response_tokens = max_response_tokens
        self.use_streaming_cache = use_streaming_cache
        self.max_kv_len = int(max_kv_len) if use_streaming_cache else None
        self.use_graph_decode = use_graph_decode and use_streaming_cache
        if use_graph_decode and not use_streaming_cache:
            raise ValueError('--graph_decode requires --streaming_cache')
        self.use_graph_evict = use_graph_evict and use_streaming_cache
        if use_graph_evict and not use_streaming_cache:
            raise ValueError('--graph_evict requires --streaming_cache')
        # GraphDecoder is built lazily on first decode (after first prefill);
        # the captured graph then serves every subsequent decode step across
        # all frames.
        self.graph_decoder = None
        # EvictionGraph is built lazily on first eviction call. Same pattern.
        self.eviction_graph = None

        self.state = StreamingStepMaskState(
            window_frames=window_frames,
            assistant_role_id=assistant_role_id,
        )

        # KV cache across frames. Two options:
        #
        # * Legacy ``Qwen3_5DynamicCache``: grows seq_len axis by torch.cat on
        #   every update. Variable shape -> not CUDA-graph capturable.
        # * ``StreamingCache``: pre-allocated (B, H, max_kv_len, D) buffer per
        #   full-attention layer; update() writes at ``cur_len`` slot in-place
        #   and returns the FULL buffer. mask_mod gets a ``valid_kv_len``
        #   predicate (kv_idx < valid_kv_len) so trailing buffer slots are
        #   block-pruned from attention at no FLOPs cost. Required for the
        #   Phase 3 GraphDecoder.
        text_cfg = getattr(self.model.config, 'text_config', self.model.config)
        if use_streaming_cache:
            from swift.template.step_mask.streaming_cache import StreamingCache
            self.past_kv = StreamingCache(
                config=text_cfg, max_kv_len=self.max_kv_len, batch_size=1, device=device, dtype=torch.bfloat16)
        else:
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache
            self.past_kv = Qwen3_5DynamicCache(config=text_cfg)

        # ``kept_positions[slot]`` = original sequence position currently
        # occupying KV-cache slot ``slot``. After eviction, slots are
        # compacted but original positions are preserved (RoPE was applied
        # at the original position and the K vector carries that).
        self.kept_positions: List[int] = []

        # Optional logits dump for SC vs Legacy bit-level comparison.
        # When non-None, ``logits_dump`` accumulates a list of fp32 tensors,
        # one per prefill-end (= the prediction for the first response token).
        self.logits_dump: Optional[list] = None

        # Optional per-frame callback for RL reward / trajectory recording.
        # Signature: ``frame_callback(info_dict)`` where info_dict is
        # ``{'frame_idx': int, 'mode': str, 'content': str,
        #    'sampled_tokens': [int], 'sampled_positions': [int]}``.
        # ``sampled_tokens`` are the EXACT ids the runner appended to the
        # state (= the model's decoded action sequence including any
        # <silent>/<inner>...</inner>/<speak>...</speak> tokens, NOT the
        # trailing <|im_end|>\n which the runner stops before forwarding).
        # ``sampled_positions`` are the original positions these tokens
        # occupy in ``state.input_ids_np`` -- useful for ref_logp pickup.
        # Default None = no callback = bit-exact same path as before.
        self.frame_callback = frame_callback
        # Internal frame counter -- incremented after each
        # prefill_then_decode() call. Used to label callbacks.
        self._frame_idx: int = 0

        # Optional sampler. Default None -> EXACT same greedy argmax path as
        # before, no extra tensor ops, no logp computation. When set (RL
        # rollout), each decode step samples via the sampler and records
        # (token, logp) -- the logps are added to the frame_callback dict
        # under the key 'sampled_logps'.
        self.sampler = sampler

    @torch.no_grad()
    def prefill_then_decode(self, new_input_ids: torch.Tensor, new_pixel_values: Optional[torch.Tensor],
                            new_image_grid_thw: Optional[torch.Tensor]) -> str:
        """Process one frame's delta tokens and decode the assistant response.

        ``new_input_ids`` MUST include the trailing assistant prefix so that
        the first sampled token IS the action token.
        """
        new_input_ids = new_input_ids.to(self.device)
        N = int(new_input_ids.shape[1])

        # Extend state + kept_positions.
        start_pos = self.state.extend(new_input_ids[0])
        for p in range(start_pos, start_pos + N):
            self.kept_positions.append(p)

        # Prefill. Prefill writes Q>1 tokens at once via slice assignment, so
        # the cache must be in NON-graph_mode here (graph_mode's index_copy_
        # uses a 1-element cur_len_tensor index). If GraphDecoder has been
        # built already, switch off; we re-enable before the decode loop.
        if self.use_graph_decode and self.past_kv.graph_mode:
            self.past_kv.disable_graph_mode()

        q_pos = torch.arange(start_pos, start_pos + N, device=self.device, dtype=torch.long)
        kv_pos, valid_kv_len = self._build_kv_pos()

        out = self._forward_with_mask(
            input_ids=new_input_ids,
            pixel_values=new_pixel_values,
            image_grid_thw=new_image_grid_thw,
            q_token_positions=q_pos,
            kv_token_positions=kv_pos,
            valid_kv_len=valid_kv_len,
        )
        # StreamingCache: advance write pointer past the N slots we just wrote.
        # Legacy DynamicCache grows on its own via torch.cat in update().
        if self.use_streaming_cache:
            self.past_kv.advance(N)
        if self.use_graph_decode and self.graph_decoder is not None:
            # Re-enable graph_mode for the upcoming captured-decode replays.
            self.past_kv.enable_graph_mode(device=self.device)
            # The prefill rebound cache.conv_states[i] to a fresh tensor
            # with post-prefill values. The captured graph still references
            # the original READ tensors -- bridge their values via copy_
            # (no re-capture). This is the cross-frame graph-reuse path.
            self.graph_decoder.bridge_post_prefill()
        if self.logits_dump is not None:
            self.logits_dump.append(out['logits'][0, -1].detach().to(torch.float32).cpu())
        if self.sampler is None:
            # Default greedy path -- bit-exact same as before sampler hook.
            next_token = int(out['logits'][0, -1].argmax().item())
            sampled_logps: List[float] = []
        else:
            next_token, _logp = self.sampler(out['logits'][0, -1])
            sampled_logps = [_logp]
        decoded = [next_token]

        # Decode loop.
        #
        # Invariant: state, kept_positions, and past_kv MUST stay byte-for-byte
        # aligned. We extend all three only when we actually forward the token
        # through the model.
        #
        # When the sampled token is <|im_end|>, we STOP without forwarding it.
        # The trailing ``<|im_end|>\n`` is part of every assistant turn's chat-
        # template encoding, so the next frame's delta (computed via
        # common_prefix_len) will start with these two tokens and feed them
        # through prefill -- keeping cache consistent without a special-case
        # forward here.
        # Build the GraphDecoder lazily on first frame; subsequent frames
        # invalidate after their prefill and re-capture here. Re-using the
        # GraphDecoder instance keeps the static tensors allocated; only
        # the CUDAGraph object itself is rebuilt.
        if self.use_graph_decode:
            if self.graph_decoder is None:
                from swift.template.step_mask.graph_decoder import GraphDecoder
                self.past_kv.enable_graph_mode(device=self.device)
                self.graph_decoder = GraphDecoder(
                    model=self.model,
                    cache=self.past_kv,
                    state=self.state,
                    max_kv_len=self.max_kv_len,
                    device=self.device,
                )
            if self.graph_decoder.graph is None:
                self.graph_decoder.capture(
                    initial_position=self.state.total_len,
                    initial_kept_positions=list(self.kept_positions),
                )

        while (next_token != _IM_END and len(decoded) < self.max_response_tokens):
            cur_pos = self.state.total_len
            self.state.extend([next_token])
            self.kept_positions.append(cur_pos)

            if self.use_graph_decode:
                if self.sampler is None:
                    next_token = self.graph_decoder.decode_step(
                        next_token_id=next_token,
                        next_position=cur_pos,
                        kept_positions=self.kept_positions,
                    )
                else:
                    next_token, _logp = self.graph_decoder.decode_step(
                        next_token_id=next_token,
                        next_position=cur_pos,
                        kept_positions=self.kept_positions,
                        sampler=self.sampler,
                    )
                    sampled_logps.append(_logp)
                decoded.append(next_token)
                continue

            tok_tensor = torch.tensor([[next_token]], device=self.device, dtype=torch.long)
            q_pos = torch.tensor([cur_pos], device=self.device, dtype=torch.long)
            kv_pos, valid_kv_len = self._build_kv_pos()
            out = self._forward_with_mask(
                input_ids=tok_tensor,
                pixel_values=None,
                image_grid_thw=None,
                q_token_positions=q_pos,
                kv_token_positions=kv_pos,
                valid_kv_len=valid_kv_len,
            )
            if self.use_streaming_cache:
                self.past_kv.advance(1)
            if self.sampler is None:
                next_token = int(out['logits'][0, -1].argmax().item())
            else:
                next_token, _logp = self.sampler(out['logits'][0, -1])
                sampled_logps.append(_logp)
            decoded.append(next_token)

        # Optional physical KV eviction at step boundaries.
        if self.evict:
            self._evict_unused_kv()

        # Optional reward / trajectory recording callback. Default None
        # = no-op = bit-exact same path as before this hook existed.
        if self.frame_callback is not None:
            # decoded[i] was sampled by forward i (i=0 is the prefill-end
            # logits, i>=1 is from the decode-loop forwards). decoded[0..i-1]
            # were extended into state at positions
            # [start_pos + N, start_pos + N + i - 1].
            # The final decoded[-1] is the FIRST out-of-loop sample: if it's
            # <|im_end|> the runner does NOT extend state with it (the
            # subsequent frame's prefill picks it up via the chat-template
            # encoding). Either way we report all sampled tokens; the
            # consumer can drop the trailing im_end if it doesn't want it
            # in the trajectory.
            prefill_end_pos = start_pos + N
            sampled_positions = list(range(prefill_end_pos, prefill_end_pos + len(decoded)))
            response_text = self.tokenizer.decode(decoded, skip_special_tokens=False)
            clean = response_text.replace('<|im_end|>', '').strip()
            cb_info = {
                'frame_idx': self._frame_idx,
                'mode': classify_response(clean),
                'content': clean,
                'sampled_tokens': list(decoded),
                'sampled_positions': sampled_positions,
            }
            if self.sampler is not None:
                # Only present when stochastic sampling was used. For RL,
                # ``sampled_logps[i]`` is log P(sampled_tokens[i] | context)
                # under the (temperature/top_p-adjusted) policy distribution.
                cb_info['sampled_logps'] = list(sampled_logps)
            self.frame_callback(cb_info)
        self._frame_idx += 1

        return self.tokenizer.decode(decoded, skip_special_tokens=False)

    def _build_kv_pos(self):
        """Build the ``kv_token_positions`` tensor consumed by mask_mod.

        With StreamingCache the K/V buffer is fixed at ``max_kv_len`` so the
        BlockMask must match — we pad ``kept_positions`` with zeros and tell
        the mask builder ``valid_kv_len = len(kept_positions)`` so the tail
        is block-pruned at zero FLOPs cost. With the legacy DynamicCache the
        buffer grows with KV so ``kv_pos`` is just the kept_positions list.
        """
        if self.use_streaming_cache:
            n_valid = len(self.kept_positions)
            pos = list(self.kept_positions) + [0] * (self.max_kv_len - n_valid)
            return (
                torch.tensor(pos, device=self.device, dtype=torch.long),
                n_valid,
            )
        return (
            torch.tensor(self.kept_positions, device=self.device, dtype=torch.long),
            None,
        )

    def _forward_with_mask(self,
                           input_ids,
                           pixel_values,
                           image_grid_thw,
                           q_token_positions,
                           kv_token_positions,
                           valid_kv_len=None):
        token = streaming_ctx_token(self.state, q_token_positions, kv_token_positions, valid_kv_len=valid_kv_len)
        try:
            kwargs = dict(
                input_ids=input_ids,
                past_key_values=self.past_kv,
                use_cache=True,
                logits_to_keep=1,
                cache_position=q_token_positions,
                # Two kernel_options work together:
                #
                # 1) BLOCK_M/BLOCK_N = 32: small triton tiles so the compiled
                #    kernel fits on GPUs with limited shared memory (L20Z:
                #    232KB/SM). Default 64x64 needs 411KB SMEM -> OOM at
                #    compile time.
                #
                # 2) FORCE_USE_FLEX_ATTENTION = True: skip the decode kernel
                #    (flex_decoding.py) and route every call through the
                #    prefill kernel (flex_attention.py). Required because the
                #    decode kernel asserts
                #        seq_len_q * gqa_shared_heads <= BLOCK_M
                #    (flex_decoding.py:286). For Qwen3.5 (gqa_shared_heads=4)
                #    with our BLOCK_M=32, this only holds for seq_len_q <= 8.
                #    Any Q>8 (e.g., prefill of 76 tokens) crashes during
                #    lowering. Forcing the prefill kernel side-steps this
                #    constraint at the cost of ~5x slower decode (Q=1).
                kernel_options={
                    'BLOCK_M': 32,
                    'BLOCK_N': 32,
                    'FORCE_USE_FLEX_ATTENTION': True,
                },
            )
            if pixel_values is not None:
                kwargs['pixel_values'] = pixel_values
            if image_grid_thw is not None:
                kwargs['image_grid_thw'] = image_grid_thw
            out = self.model(**kwargs)
        finally:
            reset_ctx(token)
        # StreamingCache lives in-place on ``self.past_kv`` (update() does not
        # rebind); the legacy DynamicCache path may return a new instance via
        # ``out.past_key_values`` in edge cases, so keep the rebind for that
        # path only.
        if not self.use_streaming_cache:
            self.past_kv = out.past_key_values
        return {'logits': out.logits}

    def _evict_unused_kv(self) -> None:
        """Drop K/V entries that the visibility rule excludes from every
        future query. Only run after the response is closed.

        Two eviction sources mirror the two ways the visibility rule blocks a
        visual key from a future query:

          (1) Closed segment: ``seg < cur_seg``. A step_close at the end of
              that segment means no future query can ever attend back to its
              visual tokens.
          (2) Open current segment but out of window: ``seg == cur_seg`` and
              ``frame_id <= cur_frame - W``. The mask predicate
              ``cur_frame[q] - frame_id[k] < W`` makes these tokens invisible
              to the current and every future frame within this segment, and
              the next step_close will evict them via (1) anyway. Freeing them
              now keeps the KV cache strictly bounded during long silent
              stretches (no step_close for many frames).
        """
        self.state._refresh_meta()
        seg = self.state._seg_id
        is_vis = self.state._is_visual
        frame_id = self.state._frame_id
        cur_seg = int(seg[-1])
        cur_frame_val = int(self.state._cur_frame[-1])
        W = int(self.state.window_frames)
        closed_seg_evict = is_vis & (seg < cur_seg)
        open_window_evict = is_vis & (seg == cur_seg) & (frame_id <= cur_frame_val - W)
        evictable_orig = set(int(p) for p in np.flatnonzero(closed_seg_evict | open_window_evict))
        if not evictable_orig:
            return
        # Which currently-kept positions are in the evictable set?
        keep_mask = [p not in evictable_orig for p in self.kept_positions]
        keep_indices = [i for i, k in enumerate(keep_mask) if k]
        new_kept = [self.kept_positions[i] for i in keep_indices]
        n_dropped = len(self.kept_positions) - len(new_kept)
        if n_dropped == 0:
            return
        idx_t = torch.tensor(keep_indices, device=self.device, dtype=torch.long)
        if self.use_streaming_cache:
            if self.use_graph_evict:
                # Phase 4: replay the captured K/V compaction graph instead
                # of running eager index_select + copy + zero per layer.
                if self.eviction_graph is None:
                    from swift.template.step_mask.eviction_graph import EvictionGraph
                    self.eviction_graph = EvictionGraph(self.past_kv, device=self.device)
                    self.eviction_graph.capture()
                n_keep_new = self.eviction_graph.replay(idx_t)
                self.past_kv.cur_len = n_keep_new
            else:
                # StreamingCache eager: in-place compact via the cache's own
                # evict(). Shape never changes -- the fixed (B, H, max_kv_len,
                # D) buffer gets surviving slots gathered to the front, tail
                # zeroed, cur_len reset to len(keep_indices).
                self.past_kv.evict(idx_t)
        else:
            # Legacy DynamicCache: rebuild each layer's tensor at the new
            # smaller seq_len via index_select. Shape varies per eviction.
            for li in range(len(self.past_kv.key_cache)):
                k = self.past_kv.key_cache[li]
                v = self.past_kv.value_cache[li]
                # Qwen3.5 hybrid: 24 of 32 layers are linear-attention
                # (Qwen3_5GatedDeltaNet). They store conv_state + recurrent_state
                # instead of K/V, so their slot in key_cache/value_cache is None.
                # Those layers carry their full-stream state in O(d) tensors and
                # don't need (or support) per-position eviction.
                if k is None or v is None:
                    continue
                if k.numel() == 0:
                    continue
                self.past_kv.key_cache[li] = k.index_select(2, idx_t).contiguous()
                self.past_kv.value_cache[li] = v.index_select(2, idx_t).contiguous()
        self.kept_positions = new_kept
        self.state.mark_evicted(n_dropped)


def common_prefix_len(a: np.ndarray, b: np.ndarray) -> int:
    """Length of the longest common prefix of two 1-D arrays."""
    n = int(min(a.shape[0], b.shape[0]))
    if n == 0:
        return 0
    eq = (a[:n] == b[:n])
    if eq.all():
        return n
    first_diff = int(np.argmin(eq))
    return first_diff


def slice_pixel_values_for_image(pv: torch.Tensor, grid_thw: torch.Tensor,
                                 image_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the pixel_values slice and 1-row grid_thw for ``image_index``."""
    patches = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()
    start = sum(patches[:image_index])
    end = sum(patches[:image_index + 1])
    return pv[start:end], grid_thw[image_index:image_index + 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', type=str, default=DEFAULT_MODEL)
    ap.add_argument('--window', type=int, default=30)
    ap.add_argument('--no_evict', action='store_true')
    ap.add_argument('--max_tokens', type=int, default=512)
    ap.add_argument('--from_jsonl', type=str, default=DEFAULT_JSONL)
    ap.add_argument('--jsonl_idx', type=int, default=0)
    ap.add_argument('--max_frames', type=int, default=None, help='Limit processing to first N frames')
    ap.add_argument('--system_prompt_path', type=str, default=DEFAULT_SYSTEM_PROMPT_PATH)
    ap.add_argument('--no_system', action='store_true')
    ap.add_argument('--device', type=str, default='cuda:0')
    ap.add_argument(
        '--streaming_cache',
        action='store_true',
        help='Use pre-allocated fixed-shape KV cache (Phase 2). '
        'Required for the Phase 3 CUDA-graph decoder.')
    ap.add_argument(
        '--max_kv_len',
        type=int,
        default=8192,
        help='Pre-allocated KV buffer length when --streaming_cache '
        'is set. ~134MB at 8192 for Qwen3.5-4B bf16.')
    ap.add_argument(
        '--dump_logits',
        type=str,
        default=None,
        help='If set, save per-frame prefill-end logits to this '
        'path (.pt). Used to compare SC vs Legacy numerics.')
    ap.add_argument(
        '--graph_decode',
        action='store_true',
        help='Use CUDA-graph-captured decode (Phase 3). Requires '
        '--streaming_cache. ~9x decode-step speedup on probe.')
    ap.add_argument(
        '--graph_evict',
        action='store_true',
        help='Use CUDA-graph-captured K/V eviction (Phase 4). '
        'Requires --streaming_cache.')
    ap.add_argument(
        '--dump_frames',
        type=str,
        default=None,
        help='If set, write a jsonl with per-frame info '
        '(idx, mode, content, sampled_tokens, '
        'sampled_positions). Used to wire up RL reward / '
        'trajectory recording without touching runner '
        'internals.')
    ap.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='Stochastic sampling temperature for the runner. '
        '0 (default) = greedy argmax = bit-exact same path '
        'as before this flag existed. >0 = enable Sampler '
        '+ track per-token logp.')
    ap.add_argument(
        '--top_p',
        type=float,
        default=1.0,
        help='Nucleus sampling threshold. Only meaningful when '
        '--temperature > 0.')
    args = ap.parse_args()

    os.environ.setdefault('IMAGE_MAX_TOKEN_NUM', '128')
    print(f'[Init] window={args.window}  evict={"OFF" if args.no_evict else "ON"}')
    print(f'[Init] model={args.model}')

    from transformers import AutoConfig, Qwen3_5ForConditionalGeneration

    # Use swift's get_processor (not transformers.AutoProcessor) so the
    # returned processor carries the ``model_info`` attribute that
    # ``get_template`` needs to look up the qwen3_5 chat-template definition.
    from swift import get_processor

    enable_flex_attn_support()
    config = AutoConfig.from_pretrained(args.model)
    text_cfg = getattr(config, 'text_config', config)
    text_cfg._attn_implementation = 'flex_attention'
    config._attn_implementation = 'flex_attention'

    print('[Init] loading model...')
    # device_map='auto' splits layers across all visible GPUs (pipeline-style).
    # HF transformers auto-moves tensors between devices as they flow through
    # the model. For our 4B model on multiple GPUs, this adds per-layer .to()
    # overhead -- expected to be SLOWER than single-GPU for this size, but
    # included for testing.
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, config=config, dtype=torch.bfloat16, device_map=args.device)
    model.eval()
    if args.device == 'auto':
        # Resolve the input/output device (where embeddings live) so the
        # runner can place its tensors on the device where the first layer
        # expects them. HF will handle subsequent inter-layer moves.
        actual_device = str(model.get_input_embeddings().weight.device)
        print(f'[Init] device_map=auto resolved input device to {actual_device}')
        print(f'[Init] model layer device map (first 6 + last 3):')
        from collections import OrderedDict
        dev_map = OrderedDict()
        for name, param in list(model.named_parameters())[:8] + list(model.named_parameters())[-3:]:
            dev_map[name] = str(param.device)
        for n, d in dev_map.items():
            print(f'    {d}  {n}')
        args.device = actual_device

    processor = get_processor(args.model)
    tok = processor.tokenizer if hasattr(processor, 'tokenizer') else processor

    asst_ids = tok.encode('assistant', add_special_tokens=False)
    assert len(asst_ids) == 1, f"'assistant' encoded to: {asst_ids}"
    assistant_role_id = int(asst_ids[0])

    patch_qwen3_5_for_step_mask(model, window_frames=args.window, assistant_role_id=assistant_role_id)

    # Template config MUST match the vLLM inference script (infer_stream_vllm_v2.py):
    #   enable_thinking=False, response_prefix='', add_non_thinking_prefix=True
    # Without enable_thinking=False the encoder leaves the prompt ending at
    # ``<think>\n`` (incomplete prefix), and the model -- trained with the
    # FULL ``<think>\n\n</think>\n\n`` non-thinking prefix -- generates
    # free-form text ("The man is holding...") instead of an action tag.
    template = get_template(
        processor,
        default_system=(Path(args.system_prompt_path).read_text(encoding='utf-8') if not args.no_system else None),
        template_type='qwen3_5',
        use_chat_template=True,
        enable_thinking=False,
        response_prefix='',
        add_non_thinking_prefix=True,
    )
    template.set_mode('train')

    query, image_paths, meta = load_case_from_jsonl(args.from_jsonl, args.jsonl_idx)
    if args.max_frames is not None:
        image_paths = image_paths[:args.max_frames]
    print(f'[Init] Case #{args.jsonl_idx}: video_id={meta["video_id"]}  n_frames={len(image_paths)}')
    print(f'[Init] Query: {query!r}')
    print('=' * 80)

    # Wire up the --dump_frames callback if requested. This is what an RL
    # training loop would do (replace open-file with reward computation
    # callback).
    frame_callback = None
    _frames_fh = None
    if args.dump_frames is not None:
        _frames_fh = open(args.dump_frames, 'w')

        def frame_callback(info):
            _frames_fh.write(json.dumps(info, ensure_ascii=False) + '\n')
            _frames_fh.flush()

    # Build a stochastic Sampler only when --temperature > 0 or --top_p < 1.
    # Default (greedy, top_p=1) leaves sampler=None so the runner uses the
    # original ``int(logits.argmax().item())`` line -- bit-exact path.
    sampler = None
    if args.temperature > 0 or args.top_p < 1.0:
        from swift.template.step_mask.sampler import Sampler
        sampler = Sampler(temperature=args.temperature, top_p=args.top_p)
        print(f'[Init] Sampler: temperature={args.temperature} top_p={args.top_p}')

    runner = MaskedStreamingRunner(
        model=model,
        tokenizer=tok,
        window_frames=args.window,
        assistant_role_id=assistant_role_id,
        device=args.device,
        evict=not args.no_evict,
        max_response_tokens=args.max_tokens,
        use_streaming_cache=args.streaming_cache,
        max_kv_len=args.max_kv_len,
        use_graph_decode=args.graph_decode,
        use_graph_evict=args.graph_evict,
        frame_callback=frame_callback,
        sampler=sampler,
    )
    if args.dump_logits is not None:
        runner.logits_dump = []

    # The swift template only auto-injects the non-thinking prefix
    # ``<think>\n\n</think>\n\n`` when the last message in ``messages`` is an
    # assistant message (i.e. during training-time encoding). At inference our
    # last message is the user's image frame and the encoder stops at
    # ``<|im_start|>assistant\n``. To match the train-time distribution we
    # manually append the prefix tokens after each ``template.encode``.
    NON_THINK_PREFIX_IDS = np.asarray(
        tok.encode('<think>\n\n</think>\n\n', add_special_tokens=False),
        dtype=np.int64,
    )
    print(f'[Init] non-thinking prefix tokens to append: {NON_THINK_PREFIX_IDS.tolist()}')

    messages: list = []
    if not args.no_system:
        sys_prompt = Path(args.system_prompt_path).read_text(encoding='utf-8')
        messages.append({'role': 'system', 'content': sys_prompt})
    all_images: list = []

    for t, img_path in enumerate(image_paths):
        t_start = time.time()
        frame = Image.open(img_path).convert('RGB')
        all_images.append(frame)
        content = f'Query: {query}<image>' if t == 0 else '<image>'
        messages.append({'role': 'user', 'content': content})

        # Re-encode full conversation; delta = what state hasn't seen yet.
        encoded = template.encode({'messages': messages, 'images': all_images})
        full_ids = np.asarray(encoded['input_ids'], dtype=np.int64)
        # Append the non-thinking prefix so the model gets the same prefix it
        # saw at training. Without this, the prompt ends at
        # ``<|im_start|>assistant\n`` and the model is forced to invent the
        # ``<think>...</think>`` block from scratch -> free-form garbage.
        full_ids = np.concatenate([full_ids, NON_THINK_PREFIX_IDS])
        state_ids = runner.state.input_ids_np

        common = common_prefix_len(state_ids, full_ids)
        if common != int(state_ids.shape[0]):
            print(f'  [warn] state diverges from encoded at pos {common} '
                  f'(state_len={state_ids.shape[0]}, encoded_len={len(full_ids)}). '
                  f'Re-encoding may have shifted prior content.')
        delta_ids = full_ids[common:]
        if delta_ids.size == 0:
            print(f'  [warn] empty delta at frame {t}, skipping')
            continue

        delta_input_ids = torch.tensor(delta_ids, dtype=torch.long).unsqueeze(0)

        # Slice pixel_values for the new image (only this frame's patches).
        pv = encoded.get('pixel_values')
        gt = encoded.get('image_grid_thw')
        if pv is None or gt is None:
            new_pv = new_gt = None
        else:
            new_pv_cpu, new_gt_cpu = slice_pixel_values_for_image(pv, gt, image_index=t)
            new_pv = new_pv_cpu.to(device=args.device, dtype=torch.bfloat16)
            new_gt = new_gt_cpu.to(device=args.device, dtype=torch.long)

        response = runner.prefill_then_decode(delta_input_ids, new_pv, new_gt)
        elapsed = time.time() - t_start

        # Strip trailing <|im_end|>\n that we appended via tokens.
        clean_resp = response.replace('<|im_end|>', '').strip()
        mode = classify_response(clean_resp)
        print(f'[t={t:03d}s] [{mode}] {clean_resp!r}'
              f'   ({elapsed:.2f}s  cache={runner.state.cache_len}/{runner.state.total_len})')

        messages.append({'role': 'assistant', 'content': clean_resp})

    # Summary.
    assistant_msgs = [m['content'] for m in messages if m['role'] == 'assistant']
    print('=' * 80)
    print(f'Done. {len(assistant_msgs)} frames processed.')
    silent = sum('<silent>' in c for c in assistant_msgs)
    inner = sum('<inner>' in c for c in assistant_msgs)
    speak = sum('<speak>' in c for c in assistant_msgs)
    n = max(len(assistant_msgs), 1)
    print(f'  Silent: {silent} ({100*silent/n:.1f}%)')
    print(f'  Inner:  {inner} ({100*inner/n:.1f}%)')
    print(f'  Speak:  {speak} ({100*speak/n:.1f}%)')
    print(f'  Final cache_len / total_len: {runner.state.cache_len} / {runner.state.total_len} '
          f'(saved {runner.state.n_evicted} positions via eviction)')

    if args.dump_logits is not None and runner.logits_dump:
        torch.save(
            {
                'logits_per_frame': runner.logits_dump,
                'streaming_cache': args.streaming_cache,
                'max_kv_len': args.max_kv_len if args.streaming_cache else None,
                'window': args.window,
                'jsonl_idx': args.jsonl_idx
            }, args.dump_logits)
        print(f'  dumped {len(runner.logits_dump)} frames of prefill-end logits to {args.dump_logits}')

    if _frames_fh is not None:
        _frames_fh.close()
        print(f'  dumped per-frame info to {args.dump_frames}')


if __name__ == '__main__':
    main()
