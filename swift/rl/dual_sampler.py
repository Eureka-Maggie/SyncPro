"""Grammar-constrained stochastic policy for dual-branch RL rollout."""

from __future__ import annotations

import torch

from swift.template.step_mask.sampler import Sampler

DUAL_POLICY_REPLAY_CONTRACT_VERSION = 12


class DualBranchSampler(Sampler):
    """Sample valid B/A turns while preserving a differentiable policy.

    The branch action and structural close tokens are sampled from explicitly
    masked distributions.  This guarantees that every rollout can be
    re-encoded by the fail-closed source-fork training template.  Content is
    otherwise sampled normally. ``policy_log_probs`` is shared by rollout and
    actor replay, so log-probabilities use exactly the same stable reduction.
    """

    CONTROL_TOKENS = (
        '<inner>',
        '</inner>',
        '<silent>',
        '<speak>',
        '</speak>',
        '<|im_end|>',
    )

    # Do not rely on ``tokenizer.all_special_ids`` here.  The Qwen3.5
    # checkpoint tokenizer does not report every template delimiter through
    # that property (notably ``<|im_start|>``).  Sampling one of these tokens
    # as branch free text changes how step-mask metadata parses the physical
    # history and can make a merged dual turn close at B rather than final A.
    # Resolve every delimiter explicitly and fail closed if the tokenizer no
    # longer represents one as a single token.
    TEMPLATE_STRUCTURAL_TOKENS = (
        '<|im_start|>',
        '<|im_end|>',
        '<|vision_start|>',
        '<|vision_end|>',
        '<|image_pad|>',
        '<|video_pad|>',
        '<think>',
        '</think>',
    )

    def __init__(self,
                 tokenizer,
                 *,
                 max_response_tokens: int,
                 temperature: float = 1.0,
                 top_p: float = 1.0,
                 seed: int | None = None):
        if temperature <= 0:
            raise ValueError('dual RL sampling requires temperature > 0')
        # Positive branches need four tokens at minimum:
        # action + non-empty content + close + im_end.
        if max_response_tokens < 4:
            raise ValueError('max_response_tokens must be >= 4')
        super().__init__(temperature=temperature, top_p=top_p, seed=seed)
        self.max_response_tokens = int(max_response_tokens)
        self.ids = {token: self._single_token_id(tokenizer, token) for token in self.CONTROL_TOKENS}
        self._routes = {
            'B': {
                'positive': self.ids['<inner>'],
                'close': self.ids['</inner>'],
            },
            'A': {
                'positive': self.ids['<speak>'],
                'close': self.ids['</speak>'],
            },
        }
        self.silent_id = self.ids['<silent>']
        self.im_end_id = self.ids['<|im_end|>']
        self._control_ids = set(self.ids.values())
        self._template_structural_ids = {
            self._single_token_id(tokenizer, token)
            for token in self.TEMPLATE_STRUCTURAL_TOKENS
        }
        self._forbidden_content_ids = (
            self._control_ids | self._template_structural_ids
            | {int(token_id)
               for token_id in getattr(tokenizer, 'all_special_ids', ())})

    @staticmethod
    def _single_token_id(tokenizer, token: str) -> int:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f'dual control token {token!r} must encode to one id; '
                             f'got {encoded}')
        token_id = int(encoded[0])
        converted = tokenizer.convert_tokens_to_ids(token)
        if converted is not None and int(converted) != token_id:
            raise ValueError(f'tokenizer disagrees on id for {token!r}: '
                             f'encode={token_id} convert={converted}')
        return token_id

    def _allowed_mask(self, logits: torch.Tensor, *, stream: str, history: list[int]) -> torch.Tensor:
        if stream not in self._routes:
            raise ValueError(f'dual sampler stream must be A|B, got {stream!r}')
        route = self._routes[stream]
        positive = route['positive']
        close = route['close']
        history = [int(token) for token in history]
        allowed = torch.zeros_like(logits, dtype=torch.bool)

        if not history:
            allowed[positive] = True
            allowed[self.silent_id] = True
            return allowed

        action = history[0]
        if action == self.silent_id:
            if len(history) != 1:
                raise ValueError(f'{stream} silent branch has tokens after <silent>: '
                                 f'{history}')
            allowed[self.im_end_id] = True
            return allowed
        if action != positive:
            raise ValueError(f'{stream} branch starts with illegal action id {action}')

        if self.im_end_id in history:
            raise ValueError(f'{stream} branch continued after im_end')
        content = history[1:]
        if close in content:
            if content[-1] != close or content.count(close) != 1:
                raise ValueError(f'{stream} branch continued after close')
            illegal_content = self._forbidden_content_ids - {close}
            if any(token in illegal_content for token in content[:-1]):
                raise ValueError(f'{stream} branch content contains a forbidden '
                                 'template/control token')
            allowed[self.im_end_id] = True
            return allowed
        illegal_content = self._forbidden_content_ids - {positive, close}
        if any(token in illegal_content for token in content):
            raise ValueError(f'{stream} branch content contains a forbidden '
                             'template/control token')

        # Reserve two slots for the required close + im_end.  At the boundary
        # the close distribution is a point mass, so the production runner's
        # max-length safety im_end is never needed for a constrained rollout.
        if len(history) >= self.max_response_tokens - 2:
            allowed[close] = True
            return allowed

        allowed.fill_(True)
        for token_id in self._forbidden_content_ids:
            if 0 <= token_id < allowed.numel():
                allowed[token_id] = False
        # A positive branch must contain at least one content token.
        if len(history) >= 2:
            allowed[close] = True
        return allowed

    def policy_logits(self, logits_1d: torch.Tensor, *, stream: str, history: list[int]) -> torch.Tensor:
        """Return fp32 logits after grammar, temperature, and top-p masks."""
        sorted_policy = self._sorted_policy_logits(
            logits_1d,
            stream=stream,
            history=history,
        )
        if sorted_policy is None:
            return self._grammar_logits(
                logits_1d,
                stream=stream,
                history=history,
            )
        sorted_logits, sorted_idx = sorted_policy
        return torch.empty_like(sorted_logits).scatter(0, sorted_idx, sorted_logits)

    def _grammar_logits(self, logits_1d: torch.Tensor, *, stream: str, history: list[int]) -> torch.Tensor:
        if logits_1d.ndim != 1:
            raise ValueError(f'expected one-dimensional logits, got {logits_1d.shape}')
        logits = logits_1d.to(torch.float32) / self.temperature
        allowed = self._allowed_mask(logits, stream=stream, history=history)
        return logits.masked_fill(~allowed, float('-inf'))

    def _sorted_policy_logits(
        self,
        logits_1d: torch.Tensor,
        *,
        stream: str,
        history: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.top_p >= 1.0:
            return None
        logits = self._grammar_logits(
            logits_1d,
            stream=stream,
            history=history,
        )
        # BF16 model logits contain many exact ties. Stable sorting gives
        # those ties a token-id order, so rollout and actor retain the same
        # boundary token when nucleus probability cuts through a tie.
        sorted_logits, sorted_idx = logits.sort(
            descending=True,
            stable=True,
        )
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        mask = sorted_probs.cumsum(dim=-1) > self.top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        return sorted_logits.masked_fill(mask, float('-inf')), sorted_idx

    def policy_log_probs(self, logits_1d: torch.Tensor, *, stream: str, history: list[int]) -> torch.Tensor:
        """Return token-id log-probs using one canonical reduction order."""
        sorted_policy = self._sorted_policy_logits(
            logits_1d,
            stream=stream,
            history=history,
        )
        if sorted_policy is None:
            logits = self._grammar_logits(
                logits_1d,
                stream=stream,
                history=history,
            )
            return torch.log_softmax(logits, dim=-1)
        sorted_logits, sorted_idx = sorted_policy
        sorted_log_probs = torch.log_softmax(sorted_logits, dim=-1)
        return torch.empty_like(sorted_log_probs).scatter(0, sorted_idx, sorted_log_probs)

    def is_structurally_deterministic(self, *, stream: str, history: list[int]) -> bool:
        """Whether grammar leaves exactly one token before seeing logits.

        These tokens are part of the physical trajectory and are still
        checked during training-layout replay, but they are not policy
        decisions and must not dilute the policy-loss denominator.
        """
        if stream not in self._routes:
            raise ValueError(f'dual sampler stream must be A|B, got {stream!r}')
        history = [int(token) for token in history]
        if not history:
            return False
        route = self._routes[stream]
        action = history[0]
        if action == self.silent_id:
            return True
        if action != route['positive']:
            raise ValueError(f'{stream} branch starts with illegal action id {action}')
        if history[-1] == route['close']:
            return True
        return len(history) >= self.max_response_tokens - 2

    @torch.no_grad()
    def sample(self, logits_1d: torch.Tensor, *, stream: str = 'default', history=None) -> tuple[int, float]:
        history = list(history or ())
        log_probs = self.policy_log_probs(logits_1d.detach(), stream=stream, history=history)
        token = int(
            torch.multinomial(log_probs.exp(), num_samples=1, generator=self._generator(log_probs, stream)).item())
        return token, float(log_probs[token].item())

    def selected_log_prob(self, logits_1d: torch.Tensor, *, token_id: int, stream: str,
                          history: list[int]) -> torch.Tensor:
        """Differentiable actor log-prob under the rollout policy."""
        log_probs = self.policy_log_probs(logits_1d, stream=stream, history=history)
        return log_probs[int(token_id)]
