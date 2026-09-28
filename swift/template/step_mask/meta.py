"""Step+window mask metadata extraction for Qwen3.5 stream training.

Given a tokenized sample's ``input_ids`` and ``labels`` (where ``labels==-100``
marks non-assistant positions), compute derived arrays that describe how
the step+window visibility rule applies. Also detect dual-branch (single
assistant turn containing both B (inner) and A (speak) sub-outputs, separated
by a mid ``<|im_end|>\\n``) and populate per-branch metadata.

Derived arrays (all shape [L]):

* ``is_visual``  : token belongs to a "visual" block that can be evicted
                   once its enclosing step closes. Covers
                   ``<|vision_start|>...<|vision_end|>``; for pure-image
                   user turns it extends to the full ``<|im_start|>user\\n
                   ...<|im_end|>\\n`` wrapper; also covers ``<silent>`` spans
                   (whole assistant turn for a fully-silent turn; only the
                   silent branch's span for a mixed dual-branch turn).
* ``frame_id``   : 0-indexed frame number for visual tokens, -1 otherwise.
* ``step_close`` : True at the position where the "step" closes.
                   Single-branch mode:  fires at ``</inner>`` / ``</speak>``.
                   Dual-branch mode:    fires at the assistant turn's CLOSING
                   ``<|im_end|>`` (the second one in the turn, after A's content).
                   The dual-branch choice means image tokens expire at frame
                   boundaries -- A can still see image_t within its own frame
                   even if B fired ``<inner>...</inner>`` earlier in that frame.

Dual-branch-only fields:

* ``route_A``    : bool [L], True for tokens in an A (speak) branch span.
                   Under dual-source training this includes the duplicated
                   source prefix token immediately before the first A action
                   token, so that ``<speak>``/the A ``<silent>`` is predicted
                   through LoRA_A.
* ``route_B``    : bool [L], True for tokens in a B (inner) branch span.
                   Under dual-source training this includes the duplicated
                   source prefix token immediately before the first B action
                   token, so that ``<inner>``/the B ``<silent>`` is predicted
                   through LoRA_B.
* ``b_spans``    : list of (start, end) tuples (end-exclusive), physical
                   indices of each dual-branch turn's B branch span.
* ``a_spans``    : list of (start, end) tuples (end-exclusive), physical
                   indices of each dual-branch turn's A branch span.
* ``frame_start_idx`` : list of int, physical index where each frame's
                        ``<|vision_start|>`` begins. Used by the linear-attn
                        fork wrapper to identify frame boundaries.
* ``is_dual_branch``  : bool, True iff at least one assistant turn was detected
                        as dual-branch (i.e., has two ``<|im_end|>`` tokens).

These arrays are the only thing the attention-time mask construction needs.
The actual visibility predicate is implemented in :mod:`mask`.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple

# Token IDs are fixed by the Qwen3.5 tokenizer plus the five
# ``--new_special_tokens`` added in stream_trans.sh. They are intentionally
# hard-coded: the values are stable across the training runs we care about,
# and probing the tokenizer at every encode step would couple metadata
# extraction to a live tokenizer instance.
SPECIAL_TOKEN_IDS = {
    'inner_open': 248077,  # <inner>
    'inner_close': 248078,  # </inner>
    'silent': 248079,  # <silent>
    'speak_open': 248080,  # <speak>
    'speak_close': 248081,  # </speak>
    'im_start': 248045,  # <|im_start|>
    'im_end': 248046,  # <|im_end|>
    'vision_start': 248053,  # <|vision_start|>
    'vision_end': 248054,  # <|vision_end|>
    'image_pad': 248056,  # <|image_pad|>
    'think_open': 248068,  # <think>
    'think_close': 248069,  # </think>
    'newline_double': 271,  # '\n\n' (single token in Qwen tokenizer)
    'newline': 198,  # '\n'
}

_STEP_CLOSE_IDS = frozenset({SPECIAL_TOKEN_IDS['inner_close'], SPECIAL_TOKEN_IDS['speak_close']})
_SILENT_ID = SPECIAL_TOKEN_IDS['silent']
_INNER_OPEN_ID = SPECIAL_TOKEN_IDS['inner_open']
_SPEAK_OPEN_ID = SPECIAL_TOKEN_IDS['speak_open']
_VIS_START = SPECIAL_TOKEN_IDS['vision_start']
_VIS_END = SPECIAL_TOKEN_IDS['vision_end']
_IM_START = SPECIAL_TOKEN_IDS['im_start']
_IM_END = SPECIAL_TOKEN_IDS['im_end']
_THINK_CLOSE = SPECIAL_TOKEN_IDS['think_close']
_B_ACTION_START_IDS = frozenset({_INNER_OPEN_ID, _SILENT_ID})
_A_ACTION_START_IDS = frozenset({_SPEAK_OPEN_ID, _SILENT_ID})

# Debug flag -- when os.environ['STEP_MASK_META_DEBUG'] is truthy, print
# per-sample summary of dual-branch detection for the first few calls. Kept
# separate from STEP_MASK_DEBUG (which lives in patch.py for mask-time
# diagnostics) so we can toggle metadata prints without producing per-step
# mask density spam.
import os as _os

_META_DEBUG_MAX = 5
_meta_debug_calls = 0


def _meta_debug_enabled() -> bool:
    return _os.environ.get('STEP_MASK_META_DEBUG', '').strip() not in ('', '0', 'false', 'False')


def _dual_source_enabled_by_env() -> bool:
    dual_enabled = _os.environ.get('MSS_DUAL_BRANCH', '').strip() not in ('', '0', 'false', 'False')
    fork_source = _os.environ.get('MSS_DUAL_BRANCH_FORK_SOURCE', '1').strip() not in ('0', 'false', 'False')
    return dual_enabled and fork_source


def _branch_action_start(ids, start: int, end: int, action_ids) -> int:
    """Return the first action-token index inside a branch span.

    In source-inclusive dual training the span starts with the duplicated
    terminal-prefix token, followed by the action token. Legacy spans start
    directly at the action token.
    """
    if start < end and int(ids[start]) in action_ids:
        return start
    if start + 1 < end and int(ids[start + 1]) in action_ids:
        return start + 1
    return start


@dataclass
class StepMaskMeta:
    is_visual: np.ndarray  # bool [L]
    frame_id: np.ndarray  # int32 [L], -1 for non-visual
    step_close: np.ndarray  # bool [L]
    # Dual-branch fields. For single-branch data these stay empty / all-False.
    route_A: np.ndarray  # bool [L]
    route_B: np.ndarray  # bool [L]
    b_spans: List[Tuple[int, int]] = field(default_factory=list)
    a_spans: List[Tuple[int, int]] = field(default_factory=list)
    frame_start_idx: List[int] = field(default_factory=list)
    is_dual_branch: bool = False


def _find_assistant_turns(ids, assistant_role_id):
    """Return list of (start, end) physical indices for every assistant turn.

    ``start`` = position of the leading ``<|im_start|>``.
    ``end``   = position of the terminating ``<|im_end|>`` (LAST one in the
                turn -- for dual-branch this is the closing one, i.e. the
                second ``<|im_end|>``; for single-branch this is the only one).

    Detection: for each ``<|im_start|>`` immediately followed by
    ``assistant_role_id``, walk forward collecting ``<|im_end|>`` until we
    hit the next ``<|im_start|>`` or end-of-sequence.

    Also returns the list of ``mid_end`` positions (None for single-branch
    turns, or the position of the first ``<|im_end|>`` for dual-branch turns).
    """
    L = ids.shape[0]
    im_starts = np.flatnonzero(ids == _IM_START)
    im_ends = np.flatnonzero(ids == _IM_END)

    turns = []  # list of (start_idx, close_end_idx, mid_end_or_None)
    for i, s in enumerate(im_starts):
        if s + 1 >= L or ids[s + 1] != assistant_role_id:
            continue
        # Bound: search up to the next <|im_start|> (or end of sequence)
        next_start = int(im_starts[i + 1]) if i + 1 < len(im_starts) else L
        turn_ends = im_ends[(im_ends > s) & (im_ends < next_start)]
        if len(turn_ends) == 0:
            # Malformed -- assistant turn without any <|im_end|>. Skip.
            continue
        if len(turn_ends) == 1:
            # Single-branch turn: one <|im_end|>, which is the close.
            turns.append((int(s), int(turn_ends[0]), None))
        else:
            # Dual-branch: first <|im_end|> is mid delimiter, LAST is close.
            # (We take the LAST rather than the second, in case future data
            # ever grows to 3+ branches; today second == last.)
            mid = int(turn_ends[0])
            close = int(turn_ends[-1])
            turns.append((int(s), close, mid))
    return turns


def _find_think_close_in_turn(ids, turn_start, turn_end):
    """Return position of ``</think>`` inside an assistant turn, or None if
    the turn has no non-thinking prefix (which happens when swift's
    ``--add_non_thinking_prefix false`` is used).
    """
    for i in range(turn_start + 1, turn_end + 1):
        if ids[i] == _THINK_CLOSE:
            return i
    return None


def build_step_mask_meta(
    input_ids,
    labels=None,
    assistant_role_id=None,
    include_branch_source=None,
) -> StepMaskMeta:
    """Compute per-token metadata for the step+window mask.

    Two ways to discriminate "real assistant output" from system-prompt literal
    text (the system prompt describes ``<silent>``/``<inner>...</inner>``/``<speak>``
    as part of the task spec, so the bare token IDs alone are ambiguous):

    * ``labels != -100``  (when ``labels`` is provided)
    * ``<|im_start|>`` immediately followed by the ``assistant`` role token
      (when ``assistant_role_id`` is provided)

    The second path is used at training time inside the model forward, where
    ``labels`` has already been stripped by swift's trainer.

    Exactly one of the two must be provided.
    """
    global _meta_debug_calls
    ids = np.asarray(input_ids, dtype=np.int64)
    L = ids.shape[0]
    assert (labels is not None) ^ (assistant_role_id is not None), ('pass exactly one of labels= or assistant_role_id=')
    if include_branch_source is None:
        include_branch_source = _dual_source_enabled_by_env()

    if labels is not None:
        lbl = np.asarray(labels, dtype=np.int64)
        assert lbl.shape[0] == L, f'input_ids and labels length mismatch: {L} vs {lbl.shape[0]}'
        is_assistant_output = lbl != -100
        # We still need to walk turns for dual-branch detection even when
        # labels are provided. Use a "virtual" assistant_role_id derived by
        # locating the token that appears immediately after <|im_start|> for
        # every assistant-output span. We prefer the assistant_role_id path
        # in practice (training path), so keep this branch simple: fall back
        # to scanning by is_assistant_output only for single-branch legacy.
        # For dual-branch data, callers should pass assistant_role_id.
        assistant_role_id_effective = None
    else:
        assistant_role_id_effective = assistant_role_id
        # Detect assistant turns and, for each, mark whether we're in the
        # single-branch or dual-branch case.
        is_assistant_output = np.zeros(L, dtype=bool)

    is_visual = np.zeros(L, dtype=bool)
    frame_id = np.full(L, -1, dtype=np.int32)
    step_close = np.zeros(L, dtype=bool)
    route_A = np.zeros(L, dtype=bool)
    route_B = np.zeros(L, dtype=bool)
    b_spans: List[Tuple[int, int]] = []
    a_spans: List[Tuple[int, int]] = []
    frame_start_idx: List[int] = []
    is_dual_branch = False

    # --- 1. Detect assistant turns (start, close_end, mid_end_or_None).
    turns: List[Tuple[int, int, 'int | None']] = []
    if assistant_role_id_effective is not None:
        turns = _find_assistant_turns(ids, assistant_role_id_effective)
        # Populate is_assistant_output from detected turn spans.
        for s, close_end, _ in turns:
            is_assistant_output[s:close_end + 1] = True
    else:
        # Labels path: derive turn spans from consecutive is_assistant_output
        # runs. Each contiguous True run is one turn's span [s, e].
        starts = np.flatnonzero(is_assistant_output & ~np.roll(is_assistant_output, 1))
        ends = np.flatnonzero(is_assistant_output & ~np.roll(is_assistant_output, -1))
        # For each such run, look for zero, one, or two <|im_end|>s inside.
        for s, e in zip(starts, ends):
            im_end_positions_in_turn = np.flatnonzero(ids[s:e + 1] == _IM_END) + s
            if len(im_end_positions_in_turn) == 0:
                continue
            if len(im_end_positions_in_turn) == 1:
                turns.append((int(s), int(im_end_positions_in_turn[0]), None))
            else:
                turns.append((
                    int(s),
                    int(im_end_positions_in_turn[-1]),
                    int(im_end_positions_in_turn[0]),
                ))

    # --- 2. Mark vision blocks: <|vision_start|> ... <|vision_end|>.
    vs_pos = np.flatnonzero(ids == _VIS_START)
    ve_pos = np.flatnonzero(ids == _VIS_END)
    assert vs_pos.shape == ve_pos.shape, (f'unbalanced vision_start/vision_end: {vs_pos.shape} vs {ve_pos.shape}')
    im_start_pos = np.flatnonzero(ids == _IM_START)
    im_end_pos = np.flatnonzero(ids == _IM_END)
    for fi, (s, e) in enumerate(zip(vs_pos, ve_pos)):
        assert s < e, f'vision_start at {s} >= vision_end at {e}'
        lo, hi = int(s), int(e)
        prev_starts = im_start_pos[im_start_pos <= s]
        next_ends = im_end_pos[im_end_pos >= e]
        if prev_starts.size and next_ends.size:
            t_start = int(prev_starts[-1])
            t_end = int(next_ends[0])
            pure_image = (s - t_start == 3) and (t_end - e == 1)
            if pure_image:
                lo = t_start
                hi = t_end
                if hi + 1 < L and ids[hi + 1] == 198:
                    hi += 1
        is_visual[lo:hi + 1] = True
        frame_id[lo:hi + 1] = fi
        frame_start_idx.append(lo)

    # --- 3. Process each assistant turn: silent visibility, route_A/B, spans.
    #     Also decide step_close placement (dual-branch -> closing im_end;
    #     single-branch -> </inner>/</speak>).
    n_dual = 0
    n_single = 0
    for turn_start, close_end, mid_end in turns:
        # Which frame does this turn belong to? Nearest vision_end <= turn_start.
        prev_ve = ve_pos[ve_pos < turn_start]
        f_idx = int(prev_ve.size - 1) if prev_ve.size > 0 else 0

        # Include trailing newline (id 198) in the turn span, matches
        # single-branch behavior for clean eviction.
        end_incl = close_end
        if end_incl + 1 < L and ids[end_incl + 1] == 198:
            end_incl += 1

        if mid_end is None:
            # --- Single-branch turn (backward-compat path) ---
            n_single += 1
            # Check for <silent>: if the turn contains a <silent> token that
            # is inside is_assistant_output, mark the whole turn visual with
            # the frame_id (matches original single-branch eviction behavior).
            silent_in_turn = np.any((ids[turn_start:close_end + 1] == _SILENT_ID)
                                    & is_assistant_output[turn_start:close_end + 1])
            if silent_in_turn:
                is_visual[turn_start:end_incl + 1] = True
                frame_id[turn_start:end_incl + 1] = f_idx
            # step_close placement: at </inner> or </speak> tokens (original).
            # (Set below in step 4.)
        else:
            # --- Dual-branch turn ---
            n_dual += 1
            is_dual_branch = True

            # Locate the shared prefix's </think>. Legacy dual data has one
            # terminal-prefix token after </think>, then the B action token:
            #
            #     </think> P <inner/silent> ...
            #
            # Dual-source training rewrites that terminal P into the B branch
            # source token and inserts a second P before A:
            #
            #     </think> P_B <inner/silent> ... mid_end \n P_A <speak/silent> ...
            #
            # So the source-inclusive B span starts at think_close + 1, while
            # the legacy action-only span starts at think_close + 2.
            think_close_pos = _find_think_close_in_turn(ids, turn_start, close_end)
            if think_close_pos is not None:
                b_content_start = think_close_pos + (1 if include_branch_source else 2)
            else:
                # No prefix -- content starts right after '<|im_start|> assistant \n'
                # i.e., turn_start + 3.
                b_content_start = turn_start + 3

            # B branch span: [b_content_start, mid_end + 2)
            #   Includes the mid <|im_end|> (B's stop marker) AND the following
            #   \n. The \n is included so the linear-attn fork wrapper can
            #   treat the entire "B tail" as one contiguous segment -- otherwise
            #   the \n would be an orphan sitting between B and A that neither
            #   branch owns.
            # A branch span: [mid_end + 2, close_end + 1)
            #   Legacy layout starts directly at A's action token. Dual-source
            #   layout starts with the duplicated P_A source token inserted
            #   before A's first action token. It runs through the closing
            #   <|im_end|>. Trailing \n (close_end + 1) is left in the
            #   post-merge shared segment.
            b_start = int(b_content_start)
            b_end = int(mid_end + 2)  # end-exclusive
            a_start = int(mid_end + 2)
            a_end = int(close_end + 1)  # end-exclusive
            assert b_start < b_end <= a_start < a_end <= L, (
                f'dual-branch span logic broken: turn_start={turn_start}, '
                f'think_close={think_close_pos}, mid_end={mid_end}, '
                f'close_end={close_end}, b=[{b_start},{b_end}), a=[{a_start},{a_end})')
            b_spans.append((b_start, b_end))
            a_spans.append((a_start, a_end))
            route_B[b_start:b_end] = True
            route_A[a_start:a_end] = True

            # Silent handling: mark silent tokens (and their branch spans if
            # they're purely silent) as visual so they can be evicted with
            # the frame.
            # Action-only silent spans are:
            #   B: <silent> + mid <|im_end|> + \n
            #   A: <silent> + close <|im_end|>
            # Source-inclusive spans have one extra P_B/P_A token before that.
            b_action_start = _branch_action_start(ids, b_start, b_end, _B_ACTION_START_IDS)
            a_action_start = _branch_action_start(ids, a_start, a_end, _A_ACTION_START_IDS)
            b_is_silent = (b_end - b_action_start == 3) and ids[b_action_start] == _SILENT_ID
            a_is_silent = (a_end - a_action_start == 2) and ids[a_action_start] == _SILENT_ID
            # Non-silent branches contain inner/speak content that should NOT
            # be marked visual (it's persistent text).
            if b_is_silent and a_is_silent:
                # Fully-silent turn: mark whole turn visual to evict cleanly.
                is_visual[turn_start:end_incl + 1] = True
                frame_id[turn_start:end_incl + 1] = f_idx
            else:
                if b_is_silent:
                    is_visual[b_start:b_end] = True
                    frame_id[b_start:b_end] = f_idx
                if a_is_silent:
                    is_visual[a_start:a_end] = True
                    frame_id[a_start:a_end] = f_idx

    # --- 4. step_close placement.
    #     Dual-branch: fires at the assistant turn's closing <|im_end|>.
    #     Single-branch: fires at </inner>/</speak> (original behavior).
    if is_dual_branch:
        # Every dual-branch turn -> close_end. Single-branch turns interspersed
        # in the same sequence also fire at their sole <|im_end|>.
        for turn_start, close_end, mid_end in turns:
            step_close[close_end] = True
    else:
        step_close = np.isin(ids, list(_STEP_CLOSE_IDS)) & is_assistant_output

    # --- DEBUG print (first _META_DEBUG_MAX calls) ---
    if _meta_debug_enabled() and _meta_debug_calls < _META_DEBUG_MAX:
        print(
            f'[step_mask.meta][call#{_meta_debug_calls}] L={L} '
            f'n_turns={len(turns)} (dual={n_dual}, single={n_single}) '
            f'is_dual_branch={is_dual_branch} '
            f'n_frames={len(frame_start_idx)} '
            f'n_step_close={int(step_close.sum())} '
            f'n_route_A={int(route_A.sum())} n_route_B={int(route_B.sum())} '
            f'n_visual={int(is_visual.sum())}',
            flush=True,
        )
        if is_dual_branch and b_spans:
            b_lens = [e - s for s, e in b_spans]
            a_lens = [e - s for s, e in a_spans]
            print(
                f'[step_mask.meta][call#{_meta_debug_calls}] '
                f'b_span_lens: min={min(b_lens)} max={max(b_lens)} '
                f'a_span_lens: min={min(a_lens)} max={max(a_lens)} '
                f'first_b={b_spans[0]} first_a={a_spans[0]}',
                flush=True,
            )
        _meta_debug_calls += 1

    return StepMaskMeta(
        is_visual=is_visual,
        frame_id=frame_id,
        step_close=step_close,
        route_A=route_A,
        route_B=route_B,
        b_spans=b_spans,
        a_spans=a_spans,
        frame_start_idx=frame_start_idx,
        is_dual_branch=is_dual_branch,
    )
