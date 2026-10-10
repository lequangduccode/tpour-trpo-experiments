import math
import unittest
import torch
from tpour import Encoder, TPOUR, metrics, normalize_triplets, trpo_loss


class CorrectnessTests(unittest.TestCase):
    def test_loss_value_and_gradient_direction(self):
        q = torch.tensor([[1., 0.]])
        p = torch.tensor([[0., 0.]], requires_grad=True)
        r = torch.tensor([[0., 0.]], requires_grad=True)
        loss = trpo_loss(q, p, r, q, p.detach(), r.detach(), beta=1, temperature=1)
        self.assertAlmostEqual(loss.item(), math.log(2), places=6)
        loss.backward()
        self.assertLess(p.grad[0, 0].item(), 0)
        self.assertGreater(r.grad[0, 0].item(), 0)
        better = trpo_loss(q, torch.tensor([[1., 0.]]), r.detach(), q,
                           p.detach(), r.detach(), beta=1, temperature=1)
        self.assertLess(better.item(), loss.item())

    def test_reference_is_detached(self):
        values = [torch.randn(2, 4, requires_grad=True) for _ in range(6)]
        trpo_loss(*values).backward()
        self.assertTrue(all(v.grad is None for v in values[3:]))

    def test_queue_wrap_and_momentum(self):
        model = TPOUR(Encoder(), queue_size=3, momentum=0.5)
        keys = torch.arange(5.).unsqueeze(1).expand(5, 64)
        model.enqueue(keys)
        self.assertEqual(int(model.count), 3)
        self.assertEqual(int(model.pointer), 2)
        self.assertEqual(model.queue[:, 0].tolist(), [3., 4., 2.])
        before = next(model.reference.parameters()).clone()
        with torch.no_grad():
            next(model.encoder.parameters()).add_(2)
        model.update_reference()
        self.assertTrue(torch.allclose(next(model.reference.parameters()), before + 1, atol=1e-6))

    def test_metrics_against_hand_calculation(self):
        result = metrics(["wrong", "right", "other"], {"right": 1}, 5)
        self.assertAlmostEqual(result["nDCG@5"], 1 / math.log2(3))
        self.assertEqual(result["Recall@5"], 1)
        self.assertEqual(result["MRR@5"], 0.5)

    def test_upstream_input_format(self):
        rows = [{"question": "query", "positive_ctxs": ["a"], "weak_positive_ctxs": ["b"]},
                {"question": "other", "positive_ctxs": [{"text": "c"}],
                 "weak_positive_ctxs": [{"text": "d"}]}]
        converted = normalize_triplets(rows)
        self.assertEqual(converted[1]["rejected"], "d")


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
