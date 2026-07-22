"""
ACT V3: Looped (weight-tied) Transformer baseline.

Comparison baseline for TRM, in the sense the literature uses the term
"looped transformer" (Giannou et al. 2023; Yang et al. ICLR 2024;
Fan et al. ICLR 2025): a block of `H_layers` transformer layers applied
`H_cycles` times with **weight sharing**, so effective depth is
`H_layers * H_cycles` while the parameter count is that of `H_layers`.

Relation to the two architectures already in this repo:

  transformers_baseline (Model_ACTV2)   untied depth-D stack; its `H_cycles`
                                        field is dead code (single pass).
                                        => the MULTI-LAYER arm.
  this file (Model_ACTV3)               weight-tied block, looped T times,
                                        FULL backprop through the loop.
                                        => the LOOPED arm.
  trm / trm_singlez                     weight-tied + z-carry + 1-step
                                        gradient (H_cycles-1 cycles run
                                        under torch.no_grad()).

The two axes that separate Model_ACTV3 from trm_singlez are therefore
(a) no z-carry stream at all and (b) full BPTT rather than the 1-step
gradient approximation. `loop_grad_cycles` exposes (b) as a knob so the
1-step-gradient trick can be ablated at fixed architecture.

Degenerate case: with H_cycles=1 this module is forward-identical to
Model_ACTV2 at the same H_layers (verified by tests/test_looped_transformer.py),
which is what makes the looped-vs-multi-layer comparison a clean matched pair.
"""

from typing import Tuple, List, Dict
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding


@dataclass
class Model_ACTV3InnerCarry:
    z_H: torch.Tensor


@dataclass
class Model_ACTV3Carry:
    inner_carry: Model_ACTV3InnerCarry

    steps: torch.Tensor
    halted: torch.Tensor

    current_data: Dict[str, torch.Tensor]


class Model_ACTV3Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int  # loop count T -- ACTIVE here (unlike transformers_baseline)
    H_layers: int  # layers in the tied block

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float
    act_enabled: bool = True
    act_inference: bool = False

    forward_dtype: str = "bfloat16"

    # -- looped-transformer specific --------------------------------------
    # Number of trailing loop iterations that carry gradient. <=0 means full
    # BPTT through all H_cycles (the canonical looped transformer, and the
    # default). 1 reproduces TRM's 1-step-gradient approximation without the
    # z-carry, which is the intended ablation of that approximation.
    loop_grad_cycles: int = 0
    # If True, the input embedding is re-injected at every loop iteration
    # (standard practice in the looped-transformer literature; also what TRM
    # does with z_H + input_embeddings). If False it is injected only on the
    # first iteration, so later iterations see the input solely through the
    # hidden state.
    input_injection_every_cycle: bool = True


class Model_ACTV3Block(nn.Module):
    def __init__(self, config: Model_ACTV3Config) -> None:
        super().__init__()

        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False,
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # Post Norm
        # Self Attention
        hidden_states = rms_norm(
            hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
            variance_epsilon=self.norm_eps,
        )
        # Fully Connected
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states


class Model_ACTV3ReasoningModule(nn.Module):
    def __init__(self, layers: List[Model_ACTV3Block]):
        super().__init__()

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        # Input injection (add)
        hidden_states = hidden_states + input_injection
        # Layers
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)

        return hidden_states


class Model_ACTV3_Inner(nn.Module):
    def __init__(self, config: Model_ACTV3Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype,
        )
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers,
                self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype,
            )

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                base=self.config.rope_theta,
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                init_std=embed_init_std,
                cast_to=self.forward_dtype,
            )
        else:
            raise NotImplementedError()

        # Reasoning Layers -- ONE tied block, applied H_cycles times in forward()
        self.H_level = Model_ACTV3ReasoningModule(
            layers=[Model_ACTV3Block(self.config) for _i in range(self.config.H_layers)]
        )

        # Initial states
        self.H_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)

            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat(
                (puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2
            )

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        return Model_ACTV3InnerCarry(
            z_H=torch.empty(
                batch_size,
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
            ),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: Model_ACTV3InnerCarry):
        return Model_ACTV3InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
        )

    def forward(
        self, carry: Model_ACTV3InnerCarry, batch: Dict[str, torch.Tensor]
    ) -> Tuple[Model_ACTV3InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
        zero_injection = torch.zeros_like(input_embeddings)

        T = self.config.H_cycles
        # Trailing cycles that carry gradient; <=0 => full BPTT (canonical).
        n_grad = T if self.config.loop_grad_cycles <= 0 else min(self.config.loop_grad_cycles, T)

        def injection(cycle_idx: int) -> torch.Tensor:
            if self.config.input_injection_every_cycle or cycle_idx == 0:
                return input_embeddings
            return zero_injection

        z_H = carry.z_H
        # Early cycles without grad (only when loop_grad_cycles is set; the
        # no_grad outputs enter the grad section as constants, exactly as in
        # trm.py's 1-step-gradient scheme).
        with torch.no_grad():
            for t in range(T - n_grad):
                z_H = self.H_level(z_H, injection(t), **seq_info)
        # Trailing cycles with grad
        for t in range(T - n_grad, T):
            z_H = self.H_level(z_H, injection(t), **seq_info)

        # LM Outputs
        new_carry = Model_ACTV3InnerCarry(
            z_H=z_H.detach(),
        )  # New carry no grad
        output = self.lm_head(z_H)[:, self.puzzle_emb_len :]

        # Q head
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class Model_ACTV3(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = Model_ACTV3Config(**config_dict)
        self.inner = Model_ACTV3_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return Model_ACTV3Carry(
            inner_carry=self.inner.empty_carry(
                batch_size
            ),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),  # Default to halted
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(
        self,
        carry: Model_ACTV3Carry,
        batch: Dict[str, torch.Tensor],
        compute_target_q: bool = False,
    ) -> Tuple[Model_ACTV3Carry, Dict[str, torch.Tensor]]:
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)

        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {
            k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v)
            for k, v in carry.current_data.items()
        }

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(
            new_inner_carry, new_current_data
        )

        outputs = {"logits": logits, "q_halt_logits": q_halt_logits, "q_continue_logits": q_continue_logits}

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps

            halted = is_last_step

            # Check if adaptive computation should be used
            use_adaptive = (self.config.halt_max_steps > 1) and (
                (self.training and self.config.act_enabled)
                or (not self.training and self.config.act_inference)
            )

            if use_adaptive:
                # Halt signal based on Q-values (but always halt at max steps)
                q_halt_signal = q_halt_logits > q_continue_logits
                halted = halted | q_halt_signal

                # Store actual steps used for logging (only during inference)
                if not self.training:
                    outputs["actual_steps"] = new_steps.float()

                # Exploration (only during training)
                if self.training:
                    min_halt_steps = (
                        torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob
                    ) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                    halted = halted & (new_steps >= min_halt_steps)

                # Compute target Q (only during training)
                if self.training and compute_target_q:
                    next_q_halt_logits, next_q_continue_logits = self.inner(
                        new_inner_carry, new_current_data
                    )[-1]

                    outputs["target_q_continue"] = torch.sigmoid(
                        torch.where(
                            is_last_step,
                            next_q_halt_logits,
                            torch.maximum(next_q_halt_logits, next_q_continue_logits),
                        )
                    )

        return Model_ACTV3Carry(
            new_inner_carry, new_steps, halted, new_current_data
        ), outputs
