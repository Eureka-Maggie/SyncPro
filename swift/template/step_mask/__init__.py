# Dual-branch training extensions (activated by MSS_DUAL_BRANCH=1)
from .dual_lora import DualLoraLinear, enable_dual_lora
from .dual_lora import env_enabled as dual_lora_enabled
from .dual_lora import inject_dual_lora_ffn
from .eviction_graph import EvictionGraph
from .graph_decoder import GraphDecoder, copy_block_mask_
from .inference import DEFAULT_WARMUP_Q_LENS, StreamingStepMaskState, warmup_flex_attention
from .linear_fork import (GatedAdditiveMerge, enable_padding_free_linear_attn, fork_and_merge_scan, get_merge_mode,
                          inject_dual_branch_linear_attn)
from .mask import build_visibility_matrix_numpy
from .meta import SPECIAL_TOKEN_IDS, StepMaskMeta, build_step_mask_meta
from .patch import enable_flex_attn_support, env_window_frames, patch_qwen3_5_for_step_mask
from .ref_logp import compute_logps_for_trajectory
from .sampler import Sampler
from .streaming_cache import StreamingCache
