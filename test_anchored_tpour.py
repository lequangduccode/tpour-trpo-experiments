import unittest
import torch
from tpour import Encoder,TPOUR
from anchored_tpour import retention_loss,AnchoredTPOUR


class RetentionTests(unittest.TestCase):
    def test_equal_embeddings_have_zero_loss(self):
        torch.manual_seed(9); x=tuple(torch.randn(3,8) for _ in range(3))
        loss,_=retention_loss(x,x)
        self.assertLess(abs(float(loss)),1e-6)

    def test_gradient_reduces_loss_and_teacher_detached(self):
        torch.manual_seed(9)
        teacher=tuple(torch.randn(3,8,requires_grad=True) for _ in range(3))
        student=tuple((t.detach()+.2*torch.randn_like(t)).requires_grad_() for t in teacher)
        before,_=retention_loss(student,teacher); before.backward()
        after,_=retention_loss(tuple(s-.05*s.grad for s in student),teacher)
        self.assertLess(float(after.detach()),float(before.detach()))
        self.assertTrue(all(t.grad is None for t in teacher))

    def test_teacher_immutable_under_optimizer_and_momentum(self):
        model=AnchoredTPOUR(Encoder('toy'),queue_size=8)
        initial={k:v.clone() for k,v in model.teacher.state_dict().items()}
        model.train(); optimizer=torch.optim.SGD(model.encoder.parameters(),lr=.01)
        loss,stats,keys=model(['question one','question two'],['answer one','answer two'],['old one','old two'])
        loss.backward(); optimizer.step(); model.update_reference(); model.enqueue(keys)
        self.assertFalse(model.teacher.training)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.teacher.parameters()))
        self.assertTrue(all(torch.equal(initial[k],v) for k,v in model.teacher.state_dict().items()))

    def test_zero_weight_exactly_matches_original_gradient(self):
        torch.manual_seed(9); original=TPOUR(Encoder('toy'),queue_size=8)
        anchored=AnchoredTPOUR(Encoder('toy'),queue_size=8,anchor_weight=0)
        anchored.encoder.load_state_dict(original.encoder.state_dict())
        anchored.reference.load_state_dict(original.reference.state_dict())
        texts=(['query a','query b'],['document a','document b'],['old a','old b'])
        a,_,_=original(*texts); b,_,_=anchored(*texts)
        a.backward(); b.backward()
        self.assertEqual(float(a.detach()),float(b.detach()))
        self.assertTrue(all(torch.equal(p.grad,q.grad) for p,q in zip(original.encoder.parameters(),anchored.encoder.parameters())))


if __name__=='__main__':
    torch.set_num_threads(2); unittest.main()
