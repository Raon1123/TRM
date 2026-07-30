"""Correctness gates for the looped-transformer baseline (Model_ACTV3).

These are the properties the looped-vs-multi-layer comparison rests on:

1. T=1 degeneracy  -- looped(H_layers=D, H_cycles=1) is forward-identical to
   transformers_baseline(H_layers=D). Without this the two arms of the grid
   are not a matched pair, they are two unrelated models.
2. Weight tying    -- parameter count is independent of H_cycles, and equals
   the untied stack at the *block* depth, not the effective depth.
3. Gradient path   -- loop_grad_cycles=0 (full BPTT) and =1 (TRM-style 1-step
   gradient) give the same forward but different gradients.
4. transformers_baseline is untouched -- its H_cycles remains dead code, so
   the module-ablation cohort already running on 10.0.12.93 keeps its meaning.
"""

import torch
import pytest

from models.recursive_reasoning.looped_transformer import Model_ACTV3
from models.recursive_reasoning.transformers_baseline import Model_ACTV2


SEQ_LEN = 11  # data/sigma_k_10/<k> dataset.json
VOCAB = 11


def _cfg(**over):
    cfg = dict(
        batch_size=4,
        seq_len=SEQ_LEN,
        puzzle_emb_ndim=0,
        num_puzzle_identifiers=1,
        vocab_size=VOCAB,
        H_cycles=1,
        H_layers=2,
        hidden_size=64,
        expansion=4,
        num_heads=4,
        pos_encodings="rope",
        halt_max_steps=1,
        halt_exploration_prob=0.1,
        forward_dtype="float32",
    )
    cfg.update(over)
    return cfg


def _batch(bs=4):
    g = torch.Generator().manual_seed(0)
    return {
        "inputs": torch.randint(0, VOCAB, (bs, SEQ_LEN), generator=g),
        "labels": torch.randint(0, VOCAB, (bs, SEQ_LEN), generator=g),
        "puzzle_identifiers": torch.zeros((bs,), dtype=torch.int32),
    }


def _run(model, batch):
    carry = model.initial_carry(batch)
    _, out = model(carry, batch)
    return out


def test_single_cycle_matches_transformers_baseline():
    """H_cycles=1 must reproduce the untied depth-D stack exactly."""
    torch.manual_seed(0)
    looped = Model_ACTV3(_cfg(H_layers=3, H_cycles=1))
    torch.manual_seed(0)
    deep = Model_ACTV2(_cfg(H_layers=3, H_cycles=1))

    # State dict keys are structurally identical (same submodule names).
    assert set(looped.state_dict()) == set(deep.state_dict())
    deep.load_state_dict(looped.state_dict())

    looped.eval()
    deep.eval()
    batch = _batch()
    with torch.no_grad():
        a = _run(looped, batch)["logits"]
        b = _run(deep, batch)["logits"]
    torch.testing.assert_close(a, b)


def test_weight_tying_param_count_independent_of_cycles():
    """Looping must add depth, not parameters."""
    n = {}
    for T in (1, 3, 6, 12):
        torch.manual_seed(0)
        m = Model_ACTV3(_cfg(H_layers=2, H_cycles=T))
        n[T] = sum(p.numel() for p in m.parameters())
    assert len(set(n.values())) == 1, n

    # ...and a tied 2-layer block looped 6x has strictly fewer params than the
    # untied depth-12 stack it is depth-matched against.
    torch.manual_seed(0)
    untied12 = sum(p.numel() for p in Model_ACTV2(_cfg(H_layers=12)).parameters())
    assert n[6] < untied12


def test_effective_depth_changes_output():
    """T>1 must actually loop (guards against the ACTV2 dead-H_cycles bug)."""
    torch.manual_seed(0)
    m1 = Model_ACTV3(_cfg(H_layers=2, H_cycles=1))
    torch.manual_seed(0)
    m6 = Model_ACTV3(_cfg(H_layers=2, H_cycles=6))
    m6.load_state_dict(m1.state_dict())
    m1.eval()
    m6.eval()
    batch = _batch()
    with torch.no_grad():
        a = _run(m1, batch)["logits"]
        b = _run(m6, batch)["logits"]
    assert not torch.allclose(a, b)


def test_transformers_baseline_ignores_cycles():
    """Pin the known ACTV2 behaviour: H_cycles is dead code there.

    This is why the 18 `abl_tfb_*_cyc6_*` cells were duplicates and were
    skipped on 10.0.12.93 -- if this test ever fails, that cohort's
    interpretation changed and ACTIVE_COHORT.md must be revisited.
    """
    torch.manual_seed(0)
    m1 = Model_ACTV2(_cfg(H_layers=2, H_cycles=1))
    torch.manual_seed(0)
    m6 = Model_ACTV2(_cfg(H_layers=2, H_cycles=6))
    m6.load_state_dict(m1.state_dict())
    m1.eval()
    m6.eval()
    batch = _batch()
    with torch.no_grad():
        torch.testing.assert_close(_run(m1, batch)["logits"], _run(m6, batch)["logits"])


def test_loop_grad_cycles_same_forward_different_grad():
    """1-step gradient must not change the forward, only the gradient."""
    torch.manual_seed(0)
    full = Model_ACTV3(_cfg(H_layers=2, H_cycles=4, loop_grad_cycles=0))
    torch.manual_seed(0)
    onestep = Model_ACTV3(_cfg(H_layers=2, H_cycles=4, loop_grad_cycles=1))
    onestep.load_state_dict(full.state_dict())
    batch = _batch()

    outs = []
    grads = []
    for m in (full, onestep):
        m.train()
        out = _run(m, batch)
        outs.append(out["logits"].detach().clone())
        out["logits"].square().mean().backward()
        grads.append(
            torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None])
        )

    torch.testing.assert_close(outs[0], outs[1])
    assert not torch.allclose(grads[0], grads[1])


def test_injection_flag_changes_output():
    torch.manual_seed(0)
    every = Model_ACTV3(_cfg(H_layers=2, H_cycles=4, input_injection_every_cycle=True))
    torch.manual_seed(0)
    first = Model_ACTV3(_cfg(H_layers=2, H_cycles=4, input_injection_every_cycle=False))
    first.load_state_dict(every.state_dict())
    every.eval()
    first.eval()
    batch = _batch()
    with torch.no_grad():
        assert not torch.allclose(_run(every, batch)["logits"], _run(first, batch)["logits"])
