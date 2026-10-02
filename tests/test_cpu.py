# SPDX-License-Identifier: MIT
"""Synthetic CPU tests and optional exact-tokenizer protocol checks."""
import ast
import copy
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from canopy_jev.prompt import clean_questions, prepare, zero_shot_prompt
from canopy_jev.model import validate_geometry
from canopy_jev.artifacts import load_manifest, state_sha, verify_base_model, verify_runtime_sources


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return ''.join(m['role']+'\n'+m['content']+'\n' for m in messages)+'assistant\n'
    def __call__(self, text, **kwargs):
        return {'input_ids': [ord(c) for c in text], 'offset_mapping': [(i, i+1) for i in range(len(text))]}


class ProtocolTests(unittest.TestCase):
    def test_no_model_import_side_effect(self):
        import canopy_jev
        self.assertNotIn('canopy_jev.tail', sys.modules)
        self.assertNotIn('canopy_jev.stem', sys.modules)
        self.assertEqual(canopy_jev.__version__, '0.1.0')

    def test_gold_fields_never_enter_prompt(self):
        q = {'question': 'Which action?', 'options': ['Wait', 'Retry']}
        expected = prepare(FakeTokenizer(), 'Synthetic state', [q])
        mutated = dict(q, answer='FORBIDDEN_ANSWER', target='FORBIDDEN_TARGET', reference='FORBIDDEN_REF', answer_index=1, rationale='FORBIDDEN_TRACE')
        self.assertEqual(expected, prepare(FakeTokenizer(), 'Synthetic state', [mutated]))
        text = ''.join(chr(x) for x in expected[1]+expected[2][0])
        self.assertNotIn('FORBIDDEN', text)

    def test_candidate_adapters(self):
        rows = clean_questions([
            {'type': 'noul', 'question': 'Ready?'},
            {'type': 'score', 'question': 'Level?', 'levels': ['Low', 'High']},
            {'question': 'Choice?', 'criteria': {'z': 'Last', 'a': 'First'}}])
        self.assertEqual(rows[0]['keys'], ['no', 'yes'])
        self.assertEqual(rows[1]['options'], ['0: Low', '1: High'])
        self.assertEqual(rows[2]['keys'], ['z', 'a'])
        self.assertEqual(rows[2]['options'], ['z: Last', 'a: First'])

    def test_candidate_failures(self):
        cases = [[], [{'question': 'Q', 'options': []}], [{'question': 'Q', 'options': ['x']*11}],
                 [{'question': '', 'options': ['x']}], [{'question': 'Q', 'type': 'unknown'}],
                 [{'question': 'Q', 'options': ['x', 'y'], 'keys': ['a', 'a']}],
                 [{'id': 'a', 'question': 'Q', 'options': ['x']}, {'id': 'a', 'question': 'R', 'options': ['y']}]]
        for q in cases:
            with self.subTest(q=q), self.assertRaises(ValueError):
                clean_questions(q)

    def test_prefix_and_geometry(self):
        rows, root, branches = prepare(FakeTokenizer(), 'S', [
            {'question': 'Q1', 'options': ['A', 'B']}, {'question': 'Q2', 'options': ['C']}])
        self.assertGreater(len(root), 0)
        self.assertTrue(all(branches))
        self.assertEqual(validate_geometry(root, branches, [2, 1], None, 16384), [[0, 1], [0]])
        for args in [([], [[1]], [1], None, 8), ([1], [[]], [1], None, 8),
                     ([1], [[2]], [11], None, 8), ([1], [[2]], [True], None, 8),
                     ([1], [[2]], [2], [[0, 0]], 8), ([1], [[2]], [1], None, 1)]:
            with self.assertRaises(ValueError):
                validate_geometry(*args)

    def test_legacy_requires_explicit_demos(self):
        with self.assertRaises(ValueError):
            prepare(FakeTokenizer(), 'S', [{'question': 'Q', 'options': ['A']}], 'legacy_five_demo')
        with self.assertRaises(ValueError):
            prepare(FakeTokenizer(), 'S', [{'question': 'Q', 'options': ['A']}], 'zero_shot_choice', [{}]*5)

    def test_zero_shot_exact_reference_template(self):
        tokenizer = FakeTokenizer()
        row = {'instructions': 'Choose?', 'options': ['a', 'b']}
        ids, boundary = zero_shot_prompt(tokenizer, {'x': 1}, row)
        text = ''.join(chr(i) for i in ids)
        self.assertIn('Application state:\n{"x": 1}\n\nQuestion:\nChoose?\nOptions:\nA. a\nB. b\nAnswer: ', text)
        self.assertTrue(text.endswith('The answer is ('))
        self.assertEqual(text[boundary:], 'A. a\nB. b\nAnswer: \nassistant\nThe answer is (')


class TensorTests(unittest.TestCase):
    def test_probability_policy_and_rejection(self):
        import torch
        from canopy_jev.policy import probabilities
        logits = torch.tensor([[1., 2., -1e4], [1., -1e4, -1e4]], dtype=torch.float32)
        mask = torch.tensor([[True, True, False], [True, False, False]])
        p = probabilities(logits, mask)
        center = logits.softmax(-1)
        center = .99*center+.01*mask/mask.sum(-1, keepdim=True)
        mean = (256*center+.001*mask)/(256+.001*mask.sum(-1, keepdim=True))
        expected = (mean.clamp_min(1e-30).log().masked_fill(~mask, -1e4)/1.1422139406204224).softmax(-1)
        self.assertTrue(torch.equal(p, expected))
        self.assertTrue((p[~mask] == 0).all()); self.assertEqual(float(p[1, 0]), 1.)
        with self.assertRaises(ValueError): probabilities(logits, mask, exploration=0.)
        with self.assertRaises(ValueError): probabilities(logits, mask, temperature=1.)
        self.assertFalse(torch.cuda.is_initialized())

    def test_prior_decode_exact_fp32(self):
        import torch
        from canopy_jev.policy import decode
        torch.manual_seed(41)
        hidden = torch.randn(3, 5120).to(torch.bfloat16)
        state = {'weight': torch.randn(10, 5120), 'bias': torch.randn(10), 'log_temperature': torch.tensor(.2)}
        prior = {'state_dict': state, 'temperature': 1.0659387111663818}
        letters = torch.tensor([[0, 1], [4, 2], [9, 8]])
        mask = torch.tensor([[1, 1], [1, 0], [1, 1]], dtype=torch.bool)
        actual = decode(hidden, prior, letters, mask)
        expected = torch.nn.functional.linear(hidden.float(), state['weight'], state['bias'])/(state['log_temperature'].float().exp().clamp(.2, 10)*prior['temperature'])
        self.assertTrue(torch.equal(actual, expected.gather(1, letters).masked_fill(~mask, -1e4)))
        self.assertEqual(actual.dtype, torch.float32)

    def test_real_artifacts_roundtrip(self):
        path = ROOT/'artifacts'
        if not (path/'manifest.json').is_file(): self.skipTest('Artifact files not present')
        import torch
        from safetensors.torch import load_file
        manifest = load_manifest(path)
        for file, key in [('adapter.safetensors', 'adapter_state_sha256'), ('prior.safetensors', 'prior_state_sha256')]:
            state = load_file(str(path/file), device='cpu')
            self.assertEqual(state_sha(state), manifest[key])
            self.assertTrue(all(v.dtype == torch.float32 for v in state.values()))
        self.assertEqual(manifest['adapter_parameters'], 6258688)
        self.assertFalse(torch.cuda.is_initialized())

    @unittest.skipUnless(os.environ.get('CANOPY_BASE_MODEL'), 'Set CANOPY_BASE_MODEL for real tokenizer/library checks')
    def test_real_tokenizer_and_installed_runtime_sources(self):
        from transformers import AutoTokenizer
        path = Path(os.environ['CANOPY_BASE_MODEL']); manifest = load_manifest(ROOT/'artifacts')
        verify_base_model(path, manifest); verify_runtime_sources(manifest)
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        questions = json.loads((ROOT/'examples/request.json').read_text())['questions']
        rows, prefix, branches = prepare(tokenizer, 'Public synthetic test state.', questions)
        self.assertEqual(len(branches), 3); self.assertTrue(all(branches)); self.assertGreater(len(prefix), 0)
        # A separate explicit reference reconstruction matches every token/boundary.
        for row in rows:
            body = 'Application state:\nPublic synthetic test state.\n\nQuestion:\n'+row['instructions']+'\nOptions:\n'
            offset = len(body)
            body += ''.join(chr(65+i)+'. '+option+'\n' for i, option in enumerate(row['options']))+'Answer: '
            text = tokenizer.apply_chat_template([{'role':'system','content':'Choose the correct option according to the question and its criteria. Return only its letter; do not write a reasoning trace.'}, {'role':'user','content':body}], tokenize=False, add_generation_prompt=True, enable_thinking=False)+'The answer is ('
            encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            boundary = next(i for i, (_, b) in enumerate(encoded['offset_mapping']) if b > text.index(body.rstrip())+offset)
            self.assertEqual(zero_shot_prompt(tokenizer, 'Public synthetic test state.', row), (encoded['input_ids'], boundary))
        import torch
        self.assertFalse(torch.cuda.is_initialized())

if __name__ == '__main__':
    unittest.main(verbosity=2)
