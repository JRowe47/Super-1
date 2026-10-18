"""Measured model/pool pairing and runtime replay for experimental libraries."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
import torch
from . import ROOT
from .hierarchy_profiles import load_hierarchy_profile


class HierarchyProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.index = ROOT / 'auto-research/results/function-kv-text-descent-20261018-v1/LOADABLE_LIBRARY.json'
        cls.data = json.loads(cls.index.read_text())
        cls.libraries = sorted((ROOT / 'auto-research/results').glob('function-kv-text-descent*/LOADABLE_LIBRARY.json'))

    def test_measured_profiles_load_and_replay_chunks(self):
        cases = [(index, entry) for index in self.libraries for entry in json.loads(index.read_text())['entries']]
        for index, entry in cases:
            with self.subTest(profile=entry['id']):
                runtime, evidence = load_hierarchy_profile(entry['id'], index)
                self.assertFalse(evidence['active'])
                ids = torch.tensor([list(b'The library preserves learned curve sampling and trained KV pooling.')])
                with torch.no_grad():
                    whole, _ = runtime(ids)
                    first, cache = runtime(ids[:, :31])
                    rest, _ = runtime(ids[:, 31:], cache)
                torch.testing.assert_close(whole, torch.cat((first, rest), 1), atol=2e-5, rtol=2e-4)

    def test_wrong_backbone_pair_is_rejected(self):
        data = copy.deepcopy(self.data)
        entry = data['entries'][0]
        other = next(e for e in data['entries'] if e['base_checkpoint'] != entry['base_checkpoint'])
        entry['base_checkpoint'] = other['base_checkpoint']
        with tempfile.TemporaryDirectory(dir=self.index.parent) as directory:
            path = Path(directory) / 'bad.json'
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'measured pair'):
                load_hierarchy_profile(entry['id'], path)

    def test_unknown_profile_is_not_automatically_selected(self):
        with self.assertRaises(LookupError):
            load_hierarchy_profile('unknown/use-case', self.index)


if __name__ == '__main__':
    unittest.main()
