import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from real_experiment import binary_metrics, clean_rows, connected_split, paired_bootstrap, train
from tpour import write_jsonl


class RealExperimentTests(unittest.TestCase):
    def test_repeated_document_stays_in_same_split(self):
        rows=[{'query':f'q{i}','preferred':f'p{i}','rejected':f'r{i}'} for i in range(20)]
        rows[3]['preferred']=rows[2]['rejected']
        a,b,stats=connected_split(rows,seed=7,test_fraction=.2)
        where=lambda row:'train' if row in a else 'heldout'
        self.assertEqual(where(rows[2]),where(rows[3]))
        self.assertEqual(stats['cross_split_exact_text_overlap'],0)

    def test_cleaning_counts_and_no_silent_empty_strings(self):
        good={'question':'q','positive_ctxs':['a'],'weak_positive_ctxs':['b']}
        rows=[good,good,{'question':'bad','positive_ctxs':[]},
              {'query':'','preferred':'a','rejected':'b'},
              {'query':'q','preferred':' A ','rejected':'a'}]
        out,s=clean_rows(rows)
        self.assertEqual(len(out),1)
        self.assertEqual([s[k] for k in ['missing_context','invalid_text','identical_pair','duplicate_triplet']],[1,1,1,1])

    def test_multi_positive_metrics_and_missing_hits(self):
        m=binary_metrics(['wrong','one','other','two'],{'one':1,'two':1},3)
        self.assertEqual(m['Recall@3'],.5)
        self.assertEqual(m['MRR@3'],.5)
        self.assertEqual(m['MAP@3'],.25)
        zero=binary_metrics(['wrong'],{'one':1},10)
        self.assertTrue(all(v==0 for v in zero.values()))

    def test_paired_ci_and_query_id_alignment(self):
        a=[{'query_id':str(i),'NDCG@10':.4} for i in range(10)]
        b=[{'query_id':str(i),'NDCG@10':.6} for i in reversed(range(10))]
        out=paired_bootstrap(a,b,iterations=100)
        self.assertAlmostEqual(out['mean_delta'],.2)
        self.assertTrue(all(abs(x-.2)<1e-10 for x in out['ci95']))
        with self.assertRaises(ValueError): paired_bootstrap(a,b[:-1])

    def test_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)
            rows=[{'query':f'question {i}','preferred':f'answer {i}','rejected':f'previous {i}'} for i in range(10)]
            data=path/'data.jsonl'; write_jsonl(data,rows)
            def args(output):
                return SimpleNamespace(command='train',data=str(data),output=str(output),model='toy',objective='tpour',
                    device='cpu',steps=4,batch_size=2,queue_size=8,max_length=16,seed=42,warmup=1,save_every=2,
                    log_every=100,lr=.01,momentum=.9,temperature=.05,beta=.3,ce_weight=.925,delete_prob=.1,
                    normalize=True,amp=False,gradient_checkpointing=False,resume=None)
            # Capture the resumable step-2 state then finish uninterrupted.
            import real_experiment
            original=real_experiment.atomic_save
            captured=path/'step2.pt'
            def saving(obj,filename):
                original(obj,filename)
                if isinstance(obj,dict) and obj.get('step')==2:
                    original(obj,captured)
            real_experiment.atomic_save=saving
            cfg=args(path/'run')
            try: train(cfg)
            finally: real_experiment.atomic_save=original
            expected=torch.load(path/'run/encoder.pt',weights_only=True)['state']
            cfg.resume=str(captured); train(cfg)
            actual=torch.load(path/'run/encoder.pt',weights_only=True)['state']
            self.assertTrue(all(torch.equal(expected[k],actual[k]) for k in expected))


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
