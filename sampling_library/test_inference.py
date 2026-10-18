"""Trained-profile equivalence and stale-cache rejection, without changing evidence."""
import json,unittest
import torch
from .profiles import INDEX,load_profile
from .inference import CachedSampledCurve,cache_bytes


class CachedInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.entries=json.loads(INDEX.read_text())['entries']
    def test_all_trained_profiles_preserve_logits_positions_and_weights(self):
        for entry in self.entries:
            with self.subTest(profile=entry['id']):
                model,_=load_profile(entry['id']);before={k:v.detach().clone() for k,v in model.state_dict().items()}
                ids=torch.tensor([list(b'The learned curve still generates every projection.')])
                runtime=CachedSampledCurve(model)
                self.assertIs(runtime.model,model)
                with torch.no_grad():
                    expected,expected_t=model(ids,positions=True);full,full_cache,full_t=runtime(ids,positions=True)
                    torch.testing.assert_close(full,expected,atol=2e-5,rtol=2e-4);torch.testing.assert_close(full_t,expected_t,atol=2e-5,rtol=2e-4)
                    state=None;parts=[];ts=[]
                    for start,end in [(0,13),(13,14),(14,31),(31,ids.shape[1])]:
                        logits,state,t=runtime(ids[:,start:end],state,positions=True);parts.append(logits);ts.append(t)
                    torch.testing.assert_close(torch.cat(parts,1),expected,atol=2e-5,rtol=2e-4)
                    torch.testing.assert_close(torch.cat(ts,1),expected_t,atol=2e-5,rtol=2e-4)
                    self.assertEqual(cache_bytes(state),entry['execution_depth']*2*ids.shape[1]*model.base.tok.embedding_dim*4)
                    self.assertEqual(cache_bytes(state),cache_bytes(full_cache))
                self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))
    def test_cache_rejects_changed_depth_weights_policy_batch_and_owner(self):
        model,_=load_profile(self.entries[-1]['id']);runtime=CachedSampledCurve(model)
        with torch.no_grad():_,state=runtime(torch.tensor([[65,66,67]]))
        model.depth-=1
        if hasattr(model,'execution_depth'):model.execution_depth-=1
        with self.assertRaisesRegex(ValueError,'prefill'):runtime(torch.tensor([[68]]),state)
        model.depth+=1
        if hasattr(model,'execution_depth'):model.execution_depth+=1
        with self.assertRaisesRegex(ValueError,'prefill'):runtime(torch.tensor([[68]]),state,uniform=True)
        with self.assertRaisesRegex(ValueError,'batch'):runtime(torch.tensor([[68],[69]]),state)
        other=CachedSampledCurve(model)
        with self.assertRaisesRegex(ValueError,'different runtime'):other(torch.tensor([[68]]),state)
        with torch.no_grad():next(model.router.parameters()).add_(.01)
        with self.assertRaisesRegex(ValueError,'prefill'):runtime(torch.tensor([[68]]),state)
        runtime(torch.tensor([[65,66,67,68]]),cache=None)
    def test_sampling_and_curve_gradients_survive_chunking(self):
        entry=next(e for e in self.entries if e['execution_depth']==4);a,_=load_profile(entry['id']);b,_=load_profile(entry['id'])
        ids=torch.tensor([list(b'A causal learned function curve.')]);expected=a(ids);expected.square().mean().backward()
        runtime=CachedSampledCurve(b);p,s=runtime(ids[:,:11]);q,s=runtime(ids[:,11:],s);torch.cat((p,q),1).square().mean().backward()
        for (name,p),(other,q) in zip(a.named_parameters(),b.named_parameters()):
            self.assertEqual(name,other)
            if p.grad is None:self.assertIsNone(q.grad)
            else:torch.testing.assert_close(p.grad,q.grad,atol=2e-5,rtol=2e-4)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in b.router.parameters()))


if __name__=='__main__':unittest.main()
