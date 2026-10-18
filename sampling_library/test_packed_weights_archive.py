"""Compact-container integrity and lossless byte-level round-trip checks."""
import tempfile
import unittest
from pathlib import Path
import torch
from sampling_library.packed_weights import PackedModel,export
from sampling_library.packed_weights_archive import read_archive,write_archive

ROOT=Path(__file__).resolve().parents[1]

class ArchiveTests(unittest.TestCase):
    def test_roundtrip_and_damage_rejection(self):
        model=torch.nn.Linear(7,3,bias=False)
        with torch.no_grad(): model.weight.copy_(torch.linspace(-1,1,21).reshape(3,7))
        bundle=export(model,{'layers':4},'nano_q4')
        with tempfile.TemporaryDirectory(prefix='packed-test-',dir=ROOT.parent) as tmp:
            p=Path(tmp)/'weights.s1pw'; write_archive(p,bundle)
            restored=read_archive(p)
            for key in ('data','scale'):
                self.assertTrue(torch.equal(bundle['entries']['weight'][key],restored['entries']['weight'][key]))
            m=PackedModel(restored)
            self.assertEqual(m.parameter_bytes,34)
            with self.assertRaises(FileExistsError): write_archive(p,bundle)
            raw=p.read_bytes()
            for label,broken in [('magic',b'BADMAGIC'+raw[8:]),('header',raw[:10]),('truncated',raw[:-1]),('corrupt',raw[:-1]+bytes([raw[-1]^1]))]:
                bad=Path(tmp)/(label+'.s1pw'); bad.write_bytes(broken)
                with self.subTest(label=label),self.assertRaises(ValueError): read_archive(bad)

if __name__=='__main__': unittest.main()
