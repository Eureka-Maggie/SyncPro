from __future__ import annotations

import json
import importlib.util
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BALANCED = ROOT / 'data' / 'rl' / 'balanced'

# Load the stdlib-only reward module without importing the full training stack.
# This keeps release-contract tests runnable in lightweight CI environments.
_REWARD_SPEC = importlib.util.spec_from_file_location(
    'syncpro_dual_rewards', ROOT / 'swift' / 'rl' / 'dual_rewards.py'
)
assert _REWARD_SPEC and _REWARD_SPEC.loader
_REWARDS = importlib.util.module_from_spec(_REWARD_SPEC)
sys.modules[_REWARD_SPEC.name] = _REWARDS
_REWARD_SPEC.loader.exec_module(_REWARDS)
DualRewardWeights = _REWARDS.DualRewardWeights
aggregate_dual_reward = _REWARDS.aggregate_dual_reward
parse_dual_target = _REWARDS.parse_dual_target


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


class RewardContractTest(unittest.TestCase):

    def test_hard_window_f1_is_chronological_one_to_one(self):
        frames = []
        for time_s, speak in ((11.0, True), (12.0, True), (40.0, True)):
            frames.append({
                't': time_s,
                'internal_response': '<silent><|im_end|>',
                'external_response': ('<speak>event</speak>' if speak else '<silent>') + '<|im_end|>',
            })
        reward = aggregate_dual_reward(
            frames,
            {'gt_windows': [{'start_s': 10.0, 'end_s': 14.0}, {'start_s': 32.0, 'end_s': 36.0}]},
            DualRewardWeights(),
        )
        self.assertEqual((reward['tp'], reward['fp'], reward['fn']), (1, 2, 1))
        self.assertAlmostEqual(reward['f1'], 0.4)
        self.assertEqual(reward['duplicate_speak_count'], 1)
        self.assertEqual(reward['out_of_window_speak_count'], 1)
        self.assertNotIn('content', reward)

    def test_sft_example_obeys_dual_target_grammar(self):
        row = _jsonl(ROOT / 'data' / 'examples' / 'sft.example.jsonl')[0]
        targets = [message['content'] for message in row['messages'] if message['role'] == 'assistant']
        self.assertEqual(len(targets), len(row['images']))
        for target in targets:
            branch_b, branch_a = parse_dual_target(target)
            self.assertTrue(branch_b.startswith(('<silent>', '<inner>')))
            self.assertTrue(branch_a.startswith(('<silent>', '<speak>')))


class PortableDataContractTest(unittest.TestCase):

    def test_sft_launcher_freezes_gamma_at_half(self):
        launcher = (ROOT / 'scripts' / 'train_sft.sh').read_text(encoding='utf-8')
        contract_writer = (
            ROOT / 'examples' / 'models' / 'qwen3_5' / 'write_dual_infer_contract.py'
        ).read_text(encoding='utf-8')
        regex_line = next(
            line for line in launcher.splitlines()
            if line.startswith('TRAINABLE_PARAMETERS_REGEX=')
        )
        self.assertNotIn('gamma_logit', regex_line)
        self.assertIn('MSS_MERGE_GAMMA=0.5', launcher)
        self.assertIn("'merge_policy': 'fixed0p5'", contract_writer)

    def test_balanced_schedule_and_relative_paths(self):
        train = _jsonl(BALANCED / 'train.jsonl')
        validation = _jsonl(BALANCED / 'val.jsonl')
        manifest = json.loads((BALANCED / 'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(len(train), 800)
        self.assertEqual(len(validation), 58)
        self.assertEqual(manifest['optimizer_steps'], 50)
        self.assertEqual(manifest['query_batch_size'], 16)
        self.assertEqual(manifest['schedule_arm'], 'balanced')
        self.assertEqual(manifest['video_roots'], {})
        for step in range(50):
            rows = train[step * 16:(step + 1) * 16]
            counts = Counter(row['screening']['behavior_category'] for row in rows)
            self.assertEqual(sorted(counts.values()), [4, 4, 4, 4])
            self.assertTrue(all(row['screening']['schedule']['step'] == step for row in rows))
        for row in train + validation:
            self.assertFalse(Path(row['video']['path']).is_absolute())

    def test_release_has_no_internal_markers_or_extra_shell_launchers(self):
        forbidden = (
            'ali' + 'baba',
            'ali' + 'yun',
            'dash' + 'scope',
            'llm' + 'chat',
            'pri' + 'mus',
            'tian' + 'xueyun',
        )
        for path in ROOT.rglob('*'):
            if not path.is_file() or '.git' in path.parts or path.suffix == '.pyc':
                continue
            try:
                text = path.read_text(encoding='utf-8').lower()
            except UnicodeDecodeError:
                continue
            for marker in forbidden:
                self.assertNotIn(marker, text, f'{marker!r} remains in {path.relative_to(ROOT)}')
        shell_files = sorted(path.relative_to(ROOT).as_posix() for path in ROOT.rglob('*.sh'))
        self.assertEqual(shell_files, ['scripts/train_rl.sh', 'scripts/train_sft.sh'])


if __name__ == '__main__':
    unittest.main()
