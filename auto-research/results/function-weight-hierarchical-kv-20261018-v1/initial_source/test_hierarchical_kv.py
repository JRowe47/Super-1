"""Verify hierarchy is a causal bounded KV backend on the same learned curve."""
import json,unittest
import torch
from torch.nn import functional as F
from .profiles import INDEX,load_profile
from .inference import CachedSampledCurve,ExactKV,cache_bytes
from .hierarchical_kv import HierarchicalKV


class HierarchicalKVTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.entries=json.loads(INDEX.read_text())['entries']
    def load(self,depth=4):return load_profile(next(e['id'] for e in self.entries if e['execution_depth']==depth))[0]
    def runtime(self,model,recent=4,levels=(2,2)):
        return CachedSampledCurve(model,lambda i,h,d:HierarchicalKV(h,d,recent_capacity=recent,level_capacities=levels)).eval()
    def test_no_compression_matches_all_trained_depth_profiles(self):
        for entry in self.entries:
            with self.subTest(profile=entry['id']):
                model,_=load_profile(entry['id']);ids=torch.tensor([list(b'Function weights remain in every projection.')]);runtime=self.runtime(model,recent=64)
                with torch.no_grad():
                    expected,t=model(ids,positions=True);actual,state,at=runtime(ids,positions=True)
                    torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-4);torch.testing.assert_close(at,t,atol=2e-5,rtol=2e-4)
    def test_compressed_prefill_decode_is_chunk_invariant_and_causal(self):
        for depth in (2,3,4):
            model=self.load(depth);before={k:v.clone() for k,v in model.state_dict().items()};runtime=self.runtime(model);torch.manual_seed(depth);ids=torch.randint(256,(2,81))
            with torch.no_grad():
                full,fullstate,t=runtime(ids,positions=True);cache=None;outputs=[];ts=[]
                for start,end in [(0,17),(17,18),(18,39),(39,81)]:
                    result,cache,at=runtime(ids[:,start:end],cache,positions=True);outputs.append(result);ts.append(at)
                torch.testing.assert_close(full,torch.cat(outputs,1),atol=2e-5,rtol=2e-4);torch.testing.assert_close(t,torch.cat(ts,1),atol=2e-5,rtol=2e-4)
                changed=ids.clone();changed[:,40:]=(changed[:,40:]+17)%256;other,_=runtime(changed);torch.testing.assert_close(full[:,:40],other[:,:40],atol=0,rtol=0)
                for memory,state in zip(runtime.memory,cache.layers):
                    nodes=list(state.recent)+[n for level in state.levels for n in level];self.assertLessEqual(len(nodes),memory.capacity);self.assertEqual(sum(n.count for n in nodes),ids.shape[1])
                self.assertLess(cache_bytes(cache),depth*2*ids.numel()*model.base.tok.embedding_dim*4)
            self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))
    def test_uniform_key_mass_accounting_matches_exact_attention(self):
        torch.manual_seed(11);q=torch.randn(1,2,43,8);k=torch.ones_like(q);v=torch.randn_like(q)
        memory=HierarchicalKV(2,8,recent_capacity=4,level_capacities=(2,2))
        expected,_=ExactKV()(q,k,v);actual,state=memory(q,k,v)
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
    def test_compressor_learns_without_modifying_function_weights(self):
        model=self.load();model.requires_grad_(False);runtime=self.runtime(model);before={k:v.clone() for k,v in model.state_dict().items()};torch.manual_seed(91);ids=torch.randint(256,(2,32))
        with torch.no_grad():teacher=model(ids).softmax(-1)
        opt=torch.optim.Adam(runtime.memory.parameters(),lr=.03)
        def objective():
            logits,_=runtime(ids);return F.kl_div(logits.log_softmax(-1),teacher,reduction='batchmean')/ids.shape[1]
        initial=float(objective().detach());initial_pool=[m.pool_query.detach().clone() for m in runtime.memory]
        for _ in range(20):
            opt.zero_grad();loss=objective();loss.backward();opt.step()
        final=float(objective().detach());self.assertLess(final,initial)
        self.assertTrue(any(not torch.equal(p,m.pool_query) for p,m in zip(initial_pool,runtime.memory)))
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))
    def test_configuration_and_weight_changes_invalidate_cache(self):
        runtime=self.runtime(self.load())
        with torch.no_grad():_,cache=runtime(torch.tensor([list(b'Prefill before changing memory capacity.')]))
        runtime.memory[0].recent_capacity+=1
        with self.assertRaisesRegex(ValueError,'prefill'):runtime(torch.tensor([[65]]),cache)
        runtime.memory[0].recent_capacity-=1
        with torch.no_grad():runtime.memory[0].pool_query.add_(.1)
        with self.assertRaisesRegex(ValueError,'prefill'):runtime(torch.tensor([[65]]),cache)


if __name__=='__main__':unittest.main()
