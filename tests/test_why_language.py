import numpy as np
import torch
from reader.why_language import pack_why_inputs, WhyAdapter


def test_evidence_changes_prefix_without_actor_gradient():
    hidden = np.random.default_rng(1).normal(size=(2,67,448)).astype(np.float32)
    evidence = np.zeros((2,10,8), np.float32)
    evidence[1,:,1] = 1
    hidden[1] = hidden[0]
    x = torch.tensor(pack_why_inputs(hidden,evidence))
    adapter = WhyAdapter(32)
    output = adapter(x.detach())
    assert output.shape == (2,13,32)
    torch.testing.assert_close(output[0,:3], output[1,:3])
    assert not torch.allclose(output[0,3:],output[1,3:])
    output.square().mean().backward()
    assert x.grad is None
    assert adapter.probe[0].weight.grad.abs().sum() > 0


def test_structured_reader_learns_comparison_without_reference_inputs():
    from reader.why_language import StructuredWhyAdapter
    adapter=StructuredWhyAdapter(32,width=32)
    x=torch.randn(2,77,448)
    words,heads=adapter.forward_with_aux(x)
    assert words.shape==(2,16,32)
    assert heads["probe"].shape==(2,11)
    (words.square().mean()+heads["probe"].square().mean()).backward()
    assert adapter.probe_head.weight.grad.abs().sum()>0
    assert x.grad is None


def test_pointer_shares_scorer_and_only_compares_measured_eligible_probes():
    from reader.why_language import PointerWhyAdapter
    adapter=PointerWhyAdapter(32,width=32)
    x=torch.randn(2,77,448);x[:,67:,5]=0;x[0,70,5]=1
    _,heads=adapter.forward_with_aux(x)
    assert heads["probe"].argmax(-1).tolist()==[4,0]
    assert torch.isfinite(heads["effect"]).all()
