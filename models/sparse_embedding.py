from typing import Union

import torch
from torch import nn
import torch.distributed as dist
from torch.optim.optimizer import Optimizer, ParamsT

from models.common import trunc_normal_init_


class CastedSparseEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, batch_size: int, init_std: float, cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to

        # Real Weights
        # Truncated LeCun normal init
        self.weights = nn.Buffer(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std), persistent=True
        )

        # Local weights and IDs
        # Local embeddings, with gradient, not persistent
        self.local_weights = nn.Buffer(torch.zeros(batch_size, embedding_dim, requires_grad=True), persistent=False)
        # Local embedding IDs, not persistent
        self.local_ids = nn.Buffer(torch.zeros(batch_size, dtype=torch.int32), persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.training:
            # Test mode, no gradient
            return self.weights[inputs].to(self.cast_to)
            
        # Training mode, fill puzzle embedding from weights
        with torch.no_grad():
            self.local_weights.copy_(self.weights[inputs])
            self.local_ids.copy_(inputs)

        return self.local_weights.to(self.cast_to)


class CastedSparseEmbeddingSignSGD_Distributed(Optimizer):
    def __init__(
        self,
        params: ParamsT,

        world_size: int,
        lr: Union[float, torch.Tensor] = 1e-3,
        weight_decay: float = 1e-2,
        dense_update: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            world_size=world_size,
            dense_update=dense_update,
        )
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure=None):  # type: ignore
        for group in self.param_groups:
            # Find the sparse embedding weights
            local_weights_grad = None
            local_ids = None
            weights = None
            
            assert len(group["params"]) == 3
            for p in group["params"]:
                if p.requires_grad:
                    local_weights_grad = p.grad
                elif p.ndim == 1:
                    local_ids = p
                elif p.ndim == 2:
                    weights = p
                else:
                    assert False
                
            assert local_ids is not None
            assert weights is not None
        
            # Apply SignSGD
            # Adam ≈ SignSGD if gradient is very sparse
            if local_weights_grad is not None:
                _sparse_emb_signsgd_dist(
                    local_weights_grad,
                    local_ids,
                    weights,
                    
                    lr=group["lr"],
                    weight_decay=group["weight_decay"],
                    world_size=group["world_size"],
                    dense_update=group["dense_update"],
                )


def _sparse_emb_signsgd_dist(
    local_weights_grad: torch.Tensor,
    local_ids: torch.Tensor,
    weights: torch.Tensor,
    
    lr: float,
    weight_decay: float,
    world_size: int,
    dense_update: bool = False,
) -> None:
    N, D = local_weights_grad.shape
    
    # All-gather
    all_weights_grad = local_weights_grad
    all_ids = local_ids

    if world_size > 1:
        all_weights_grad = torch.empty((world_size * N, D), dtype=local_weights_grad.dtype, device=local_weights_grad.device)
        all_ids = torch.empty(world_size * N,               dtype=local_ids.dtype,          device=local_ids.device)
    
        dist.all_gather_into_tensor(all_weights_grad, local_weights_grad)
        dist.all_gather_into_tensor(all_ids,          local_ids)

    if dense_update:
        # PERF-001 candidate P1-A. Equivalent to the branch below, without the
        # unique() call -- which forces a device-to-host sync because it must
        # know its own output size, and which py-spy measured at 37.6% of wall
        # time on the sigma^k config (num_puzzle_identifiers = 1, so every id in
        # the batch is 0 and unique re-derives a constant on every step).
        #
        # Scatter straight into a full (P, D) buffer instead. Rows that do not
        # appear in the batch accumulate nothing, so sign() of their gradient is
        # 0 and the add is a no-op for them -- but the DECAY is not, which is the
        # trap: applying `1 - lr*wd` to the whole table would decay embeddings
        # this batch never touched. `scale` gates the decay to present rows only.
        # Everything here is asynchronous; nothing reads a size back to the host.
        # Every operation below takes its scalars as kernel arguments. Nothing
        # copies a Python value to the device and nothing reads a size back, so
        # the host never blocks. That is the whole point, and it is easy to lose:
        # the first version of this branch used `weights.new_tensor(1.0 - lr*wd)`
        # inside a torch.where, which is a synchronising host-to-device copy --
        # py-spy showed the stall simply move off unique() and onto that line
        # (42.9% of wall), with no net gain.
        P = weights.shape[0]
        index = all_ids.long()

        # 1.0 where the row appears in this batch, 0.0 otherwise. scatter_ takes
        # a scalar `value`, so no ones-tensor is materialised.
        present = torch.zeros((P, 1), dtype=weights.dtype, device=weights.device)
        present.scatter_(0, index.unsqueeze(-1), 1.0)

        grad = torch.zeros((P, D), dtype=all_weights_grad.dtype, device=all_weights_grad.device)
        grad.scatter_add_(0, index.unsqueeze(-1).expand(-1, D), all_weights_grad)

        # Decay only the rows that appeared: absent rows get scale exactly 1.0.
        # Their gradient is 0, so sign() is 0 and the add is a no-op for them.
        scale = 1.0 - present * (lr * weight_decay)
        weights.mul_(scale).add_(torch.sign(grad), alpha=-lr)
        return

    # Unique
    grad_ids, inv = all_ids.unique(return_inverse=True)

    grad = torch.zeros((grad_ids.shape[0], D), dtype=all_weights_grad.dtype, device=all_weights_grad.device)
    grad.scatter_add_(0, inv.unsqueeze(-1).expand(-1, D), all_weights_grad)

    # SignSGD with decoupled weight decay
    p = weights[grad_ids]

    p.mul_(1.0 - lr * weight_decay).add_(torch.sign(grad), alpha=-lr)

    # Write updated slices back
    weights[grad_ids] = p
