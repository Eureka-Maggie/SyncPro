"""Stochastic sampler for RL rollout.

Default greedy decoding stays on a separate code path; this module is only
invoked when ``--temperature > 0`` or ``--top_p < 1`` is set. The sampler
returns both the sampled token id AND its log-probability under the
temperature/top_p-adjusted distribution, which the RL trainer needs for
importance ratio + KL terms.

Interface (callable):
    sampler(logits_1d: torch.Tensor) -> (token_id: int, log_prob: float)
"""

from __future__ import annotations

import hashlib
import torch


class Sampler:
    """Temperature + nucleus (top_p) sampler with logp tracking.

    Args:
        temperature: scaling factor on logits before softmax. ``> 0`` enables
            stochastic sampling. Setting ``temperature == 0`` falls back to
            argmax greedy (but logp is still returned).
        top_p: nucleus sampling threshold. Slots with cumulative probability
            above ``top_p`` get masked out (set to ``-inf``). Default ``1.0``
            disables top_p (use full distribution).
    """

    def __init__(self, temperature: float = 1.0, top_p: float = 1.0, seed: int | None = None):
        if temperature < 0:
            raise ValueError(f'temperature must be >= 0, got {temperature}')
        if not (0 < top_p <= 1.0):
            raise ValueError(f'top_p must be in (0, 1], got {top_p}')
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.seed = None if seed is None else int(seed)
        self._generators: dict[tuple[str, str], torch.Generator] = {}

    def _generator(self, logits: torch.Tensor, stream: str) -> torch.Generator | None:
        """Return a deterministic generator isolated by logical stream.

        A dual rollout owns one :class:`Sampler`, but A and B can be
        scheduled serially, as a two-row batch, or inside a ragged
        multi-query batch.  Sharing one RNG would make those equivalent
        schedulers consume random numbers in different orders.  Keying the
        RNG by branch keeps sampling invariant to that implementation detail.
        """
        if self.seed is None:
            return None
        device = str(logits.device)
        key = (device, str(stream))
        generator = self._generators.get(key)
        if generator is None:
            digest = hashlib.blake2b(str(stream).encode('utf-8'), digest_size=8).digest()
            stream_offset = int.from_bytes(digest, 'little', signed=False)
            # torch.Generator.manual_seed accepts a signed 64-bit seed.
            stream_seed = (self.seed + stream_offset) % (2**63 - 1)
            generator = torch.Generator(device=logits.device)
            generator.manual_seed(stream_seed)
            self._generators[key] = generator
        return generator

    @torch.no_grad()
    def sample(self, logits_1d: torch.Tensor, *, stream: str = 'default', history=None) -> tuple[int, float]:
        """Sample from one named logical stream and return token plus logp."""
        return self._sample_impl(logits_1d, generator=self._generator(logits_1d, stream))

    @torch.no_grad()
    def __call__(self, logits_1d: torch.Tensor) -> tuple[int, float]:
        """Sample one token. Returns (token_id, log_prob).

        ``log_prob`` is computed under the SAME post-temperature /
        post-top_p distribution that the token was sampled from. For RL,
        this matches the policy distribution used at rollout time.
        """
        return self.sample(logits_1d)

    def _sample_impl(self, logits_1d: torch.Tensor, *, generator: torch.Generator | None) -> tuple[int, float]:
        # Work in fp32 to keep log_softmax numerically stable.
        logits = logits_1d.detach().to(torch.float32)

        if self.temperature == 0.0:
            # Greedy. logp under the un-tempered distribution.
            token = int(logits.argmax().item())
            log_prob = float(torch.log_softmax(logits, dim=-1)[token].item())
            return token, log_prob

        logits = logits / self.temperature

        if self.top_p < 1.0:
            sorted_logits, sorted_idx = logits.sort(descending=True)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumprobs = sorted_probs.cumsum(dim=-1)
            # Keep tokens up to (and including) the one that crosses top_p.
            # mask[i] = True means "drop the i-th sorted slot".
            mask = cumprobs > self.top_p
            # shift right so the first slot crossing the threshold stays
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            sorted_logits[mask] = float('-inf')
            # Scatter back to original token-id space so multinomial /
            # log_softmax see the masked-out positions at the right ids.
            logits = torch.empty_like(logits).scatter_(0, sorted_idx, sorted_logits)

        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        token = int(torch.multinomial(probs, num_samples=1, generator=generator).item())
        log_prob = float(log_probs[token].item())
        return token, log_prob
