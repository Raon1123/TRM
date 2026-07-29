from typing import Optional, Any, Sequence, List
from dataclasses import dataclass
import os
import math
import yaml
import shutil
import copy

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig
from adam_atan2_pytorch import AdamAtan2 as AdamATan2

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from models.ema import EMAHelper
from utils.z_logging import ZDynamicsLogger
from utils.perf_profiler import PerfProfilerConfig, TrainingProfiler
from utils.perf_benchmark import PerfBenchmarkConfig, TrainingBenchmark


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig


class EvaluatorConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class PretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    # Data
    data_paths: List[str]
    data_paths_test: List[str] = []
    # Evaluators
    evaluators: List[EvaluatorConfig] = []

    # Hyperparams
    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    # Puzzle embedding
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    load_checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Extras
    seed: int = 0
    # σ^k composition depth — metadata only (training never reads it); pass +k=<k>
    # so wandb config carries k/seed filterable columns instead of parsing run names.
    k: Optional[int] = None
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    min_eval_interval: Optional[int] = 0 # when to start eval
    eval_save_outputs: List[str] = []

    ema: bool = False # use Exponential-Moving-Average
    ema_rate: float = 0.999 # EMA-rate
    freeze_weights: bool = False # If True, freeze weights and only learn the embeddings

    # z-dynamics logging (off by default — zero overhead when False)
    log_z_dynamics: bool = False
    z_probe_size: int = 512
    z_snapshot: bool = True
    phase_threshold: float = 0.999
    phase_patience: int = 2

    # The normal path constructs an inert profiler; see perf_profiler.py.
    # PERF-001 candidate P1-A rollback switch. False keeps the registered
    # unique()-based sparse-embedding update; True uses the dense scatter that
    # avoids the per-step device-to-host sync. Off by default until G1/G2 pass.
    puzzle_emb_dense_update: bool = False

    perf_profiler: PerfProfilerConfig = pydantic.Field(default_factory=PerfProfilerConfig)
    # The normal path constructs an inert benchmark; see perf_benchmark.py.
    perf_benchmark: PerfBenchmarkConfig = pydantic.Field(default_factory=PerfBenchmarkConfig)

@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps: int


def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, **kwargs):
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=config.data_paths_test if len(config.data_paths_test)>0 and split=="test" else config.data_paths,
        rank=rank,
        num_replicas=world_size,
        **kwargs
    ), split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True
    )
    return dataloader, dataset.metadata


def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=config.global_batch_size // world_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False  # Non-autoregressive
    )

    # Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        print(model)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model)  # type: ignore

        # Load checkpoint
        if rank == 0:
            load_checkpoint(model, config)

        # Broadcast parameters from rank 0
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    # Optimizers and lr
    if config.arch.puzzle_emb_ndim == 0:
        optimizers = [
            AdamATan2(
                model.parameters(),
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2)
            )
        ]
        optimizer_lrs = [
            config.lr
        ]
    elif config.freeze_weights:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size,
                dense_update=config.puzzle_emb_dense_update,
            )
        ]
        optimizer_lrs = [
            config.puzzle_emb_lr
        ]
    else:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=1.0e-6,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size,
                dense_update=config.puzzle_emb_dense_update,
            ),
            AdamATan2(
                model.parameters(),
                lr=1.0e-10,  # Needs to be set by scheduler
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2)
            )
        ]
        optimizer_lrs = [
            config.puzzle_emb_lr,
            config.lr
        ]

    return model, optimizers, optimizer_lrs

def mix_weights_direct(device, alpha, net, nets):
    sd = []
    for i in range(len(nets)):
        sd += [nets[i].state_dict()]
    sd_alpha = {}
    for k in sd[0].keys():
        comb_net = alpha[0]*sd[0][k].to(device)
        for i in range(1,len(nets)):
            comb_net += alpha[i]*sd[i][k].to(device)
        sd_alpha[k] =  comb_net
    net.load_state_dict(sd_alpha)
    return net

def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))


def init_train_state(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    # Estimated total training steps
    total_steps = int(config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)

    # Model
    model, optimizers, optimizer_lrs = create_model(config, train_metadata, rank=rank, world_size=world_size)

    return TrainState(
        step=0,
        total_steps=total_steps,

        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None
    )


def save_train_state(config: PretrainConfig, train_state: TrainState):
    # FIXME: Only saved model.
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    torch.save(train_state.model.state_dict(), os.path.join(config.checkpoint_path, f"step_{train_state.step}"))


def load_checkpoint(model: nn.Module, config: PretrainConfig):
    if config.load_checkpoint is not None:
        print(f"Loading checkpoint {config.load_checkpoint}")

        # Load state dict
        state_dict = torch.load(config.load_checkpoint, map_location="cuda")

        # Resize and reset puzzle emb if needed
        puzzle_emb_name = "_orig_mod.model.inner.puzzle_emb.weights"
        expected_shape: torch.Size = model.model.puzzle_emb.weights.shape  # type: ignore
        if puzzle_emb_name in state_dict:
            puzzle_emb = state_dict[puzzle_emb_name]
            if puzzle_emb.shape != expected_shape:
                print(f"Resetting puzzle embedding as shape is different. Found {puzzle_emb.shape}, Expected {expected_shape}")
                # Re-initialize using mean
                state_dict[puzzle_emb_name] = (
                    torch.mean(puzzle_emb, dim=0, keepdim=True).expand(expected_shape).contiguous()
                )
        model.load_state_dict(state_dict, assign=True)


def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio
    )



def create_evaluators(config: PretrainConfig, eval_metadata: PuzzleDatasetMetadata) -> List[Any]:
    data_paths =config.data_paths_test if len(config.data_paths_test)>0 else config.data_paths
    # Initialize evaluators
    evaluators = []
    for cfg in config.evaluators:
        for data_path in data_paths:
            cls = load_model_class(cfg.name, "evaluators.")(
                data_path=data_path, eval_metadata=eval_metadata, **cfg.__pydantic_extra__
            )  # type: ignore
            evaluators.append(cls)

    return evaluators

def train_batch(config: PretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int, rank: int, world_size: int, bench: TrainingBenchmark):
    train_state.step += 1
    if train_state.step > train_state.total_steps:  # At most train_total_steps
        return

    # After the total_steps guard, so a skipped update never opens a row; before
    # the H2D copy, so token counts are read off the CPU batch without touching CUDA.
    bench.begin_update(batch, global_batch_size)

    # To device
    with bench.cuda_span("h2d"):
        batch = {k: v.cuda() for k, v in batch.items()}

    # Init carry if it is None
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)  # type: ignore

    # Forward
    with bench.cuda_span("forward_backward"):
        train_state.carry, loss, metrics, _, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])

        ((1 / global_batch_size) * loss).backward()

    # Allreduce
    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)
            
    # Apply optimizer
    lr_this_step = None
    with bench.cuda_span("optimizer"):
        for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
            lr_this_step = compute_lr(base_lr, config, train_state)

            for param_group in optim.param_groups:
                param_group['lr'] = lr_this_step

            optim.step()
            optim.zero_grad()

    # Reduce metrics
    if len(metrics):
        assert not any(v.requires_grad for v in metrics.values())

        metric_keys = list(sorted(metrics.keys()))  # Sort keys to guarantee all processes use the same order.
        # Reduce and reconstruct
        with bench.cuda_span("metrics_device"):
            metric_values = torch.stack([metrics[k] for k in metric_keys])
            if world_size > 1:
                dist.reduce(metric_values, dst=0)

        if rank == 0:
            with bench.wall_span("metrics_wandb"):
                metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}
            
            # Postprocess
            count = max(reduced_metrics["count"], 1)  # Avoid NaNs
            reduced_metrics = {f"train/{k}": v / (global_batch_size if k.endswith("loss") else count) for k, v in reduced_metrics.items()}

            reduced_metrics["train/lr"] = lr_this_step
            return reduced_metrics


def train_batch_profiled(
    config: PretrainConfig,
    train_state: TrainState,
    batch: Any,
    global_batch_size: int,
    rank: int,
    world_size: int,
    profiler: TrainingProfiler,
):
    """Profiled counterpart to ``train_batch``.

    Kept separate so that attaching the CUPTI profiler -- whose `record_function`
    ranges perturb the very timings M2 registers -- stays confined to this
    function and never touches the default path.

    The default ``train_batch`` is NO LONGER free of context managers around
    model execution: it carries `bench.cuda_span(...)` / `bench.wall_span(...)`
    brackets at h2d, forward+backward, optimizer and metrics.  Those are inert
    when the benchmark is disabled -- each is one frozenset membership test, one
    `None` check on the open row, and enter/exit of the shared module-level
    `nullcontext` -- so the disabled hot path is unchanged in behaviour, but it
    is not literally call-free.  This function's separation buys profiler
    isolation, not context-manager absence.
    """
    train_state.step += 1
    if train_state.step > train_state.total_steps:
        return

    with profiler.record("train/h2d"):
        batch = {k: v.cuda() for k, v in batch.items()}

    if train_state.carry is None:
        with profiler.record("train/initial_carry"):
            with torch.device("cuda"):
                train_state.carry = train_state.model.initial_carry(batch)  # type: ignore

    with profiler.record("train/forward_loss"):
        train_state.carry, loss, metrics, _, _ = train_state.model(
            carry=train_state.carry, batch=batch, return_keys=[]
        )

    with profiler.record("train/backward"):
        ((1 / global_batch_size) * loss).backward()

    if world_size > 1:
        with profiler.record("train/allreduce"):
            for param in train_state.model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad)

    lr_this_step = None
    with profiler.record("train/optimizer"):
        for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
            lr_this_step = compute_lr(base_lr, config, train_state)
            for param_group in optim.param_groups:
                param_group['lr'] = lr_this_step
            optim.step()
            optim.zero_grad()

    if len(metrics):
        assert not any(v.requires_grad for v in metrics.values())
        metric_keys = list(sorted(metrics.keys()))
        with profiler.record("train/metrics_reduce"):
            metric_values = torch.stack([metrics[k] for k in metric_keys])
            if world_size > 1:
                dist.reduce(metric_values, dst=0)

        if rank == 0:
            with profiler.record("train/metrics_d2h"):
                metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}
            count = max(reduced_metrics["count"], 1)
            reduced_metrics = {
                f"train/{k}": v / (global_batch_size if k.endswith("loss") else count)
                for k, v in reduced_metrics.items()
            }
            reduced_metrics["train/lr"] = lr_this_step
            return reduced_metrics


def profiled_train_batches(train_loader: DataLoader, profiler: TrainingProfiler):
    """Expose DataLoader blocking time as a record_function range."""
    loader_iter = iter(train_loader)
    while True:
        try:
            with profiler.record("train/data_wait"):
                item = next(loader_iter)
        except StopIteration:
            return
        yield item

def evaluate(
    config: PretrainConfig,
    train_state: TrainState,
    eval_loader: torch.utils.data.DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    evaluators: List[Any],
    rank: int,
    world_size: int,
    cpu_group: Optional[dist.ProcessGroup],
):
    reduced_metrics = None

    with torch.inference_mode():
        return_keys = set(config.eval_save_outputs)
        for evaluator in evaluators:
            evaluator.begin_eval()
            return_keys.update(evaluator.required_outputs)

        # Run evaluation
        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}

        save_preds = {}

        metric_keys = []
        metric_values = None

        carry = None
        processed_batches = 0
        
        for set_name, batch, global_batch_size in eval_loader:
            processed_batches += 1
            if rank == 0:
                print(f"Processing batch {processed_batches}: {set_name}")
            
            # To device
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = train_state.model.initial_carry(batch)  # type: ignore

            # Forward
            inference_steps = 0
            while True:
                carry, loss, metrics, preds, all_finish = train_state.model(
                    carry=carry, batch=batch, return_keys=return_keys
                )
                inference_steps += 1

                if all_finish:
                    break

            if rank == 0:
                print(f"  Completed inference in {inference_steps} steps")

            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        save_preds.setdefault(k, [])
                        save_preds[k].append(v.cpu())  # Move to CPU for saving GPU memory

            for evaluator in evaluators:
                evaluator.update_batch(batch, preds)

            del carry, loss, preds, batch, all_finish

            # Aggregate metrics
            set_id = set_ids[set_name]

            if metric_values is None:
                metric_keys = list(
                    sorted(metrics.keys())
                )  # Sort keys to guarantee all processes use the same order.
                metric_values = torch.zeros(
                    (len(set_ids), len(metrics.values())), dtype=torch.float32, device="cuda"
                )

            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])

            del metrics

        # concatenate save preds
        save_preds = {k: torch.cat(v, dim=0) for k, v in save_preds.items()}

        # Save preds
        if config.checkpoint_path is not None and len(save_preds):
            # Each rank save predictions independently
            os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
            torch.save(
                save_preds, os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}")
            )

        del save_preds

        # Reduce to rank 0
        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)

            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {
                    set_name: {
                        metric_name: reduced_metrics[set_id, metric_id]
                        for metric_id, metric_name in enumerate(metric_keys)
                    }
                    for set_id, set_name in enumerate(set_ids)
                }

                # Postprocess
                for set_name, m in reduced_metrics.items():
                    count = m.pop("count")
                    reduced_metrics[set_name] = {k: v / count for k, v in m.items()}

        # Run evaluators
        if rank == 0:
            print(f"\nRunning {len(evaluators)} evaluator(s)...")
            
        for i, evaluator in enumerate(evaluators):
            if rank == 0:
                print(f"Running evaluator {i+1}/{len(evaluators)}: {evaluator.__class__.__name__}")
                
            # Path for saving
            evaluator_save_path = None
            if config.checkpoint_path is not None:
                evaluator_save_path = os.path.join(
                    config.checkpoint_path,
                    f"evaluator_{evaluator.__class__.__name__}_step_{train_state.step}",
                )
                os.makedirs(evaluator_save_path, exist_ok=True)

            # Run and log
            metrics = evaluator.result(evaluator_save_path, rank=rank, world_size=world_size, group=cpu_group)
            if rank == 0 and metrics is not None:
                if reduced_metrics is None:
                    reduced_metrics = {}

                reduced_metrics.update(metrics)
                print(f"  Completed {evaluator.__class__.__name__}")
                
        if rank == 0:
            print("All evaluators completed!")

    return reduced_metrics

def save_code_and_config(config: PretrainConfig):
    if config.checkpoint_path is None or wandb.run is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Copy code
    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name)
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)

            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    # Dump config as yaml
    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)

    # Log code
    wandb.run.log_code(config.checkpoint_path)


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)  # type: ignore

        # Naming
        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.data_paths[0]).capitalize()}-ACT-torch"
        if config.run_name is None:
            config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)

        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]  # type: ignore


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1
    CPU_PROCESS_GROUP = None

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        
        # CPU GLOO process group
        CPU_PROCESS_GROUP = dist.new_group(backend="gloo")
        assert (
            dist.get_rank(CPU_PROCESS_GROUP) == RANK and dist.get_world_size(CPU_PROCESS_GROUP) == WORLD_SIZE
        )

    # Load sync'ed config
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)

    # Seed RNGs to ensure consistency
    torch.random.manual_seed(config.seed + RANK)

    # Dataset
    train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
    total_iters = config.epochs // train_epochs_per_iter

    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."

    train_loader, train_metadata = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=train_epochs_per_iter, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    try:
        eval_loader,  eval_metadata  = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    except:
        print("NO EVAL DATA FOUND")
        eval_loader = eval_metadata = None

    try:
        evaluators = create_evaluators(config, eval_metadata)
    except:
        print("No evaluator found")
        evaluators = []

    # Train state
    train_state = init_train_state(config, train_metadata, rank=RANK, world_size=WORLD_SIZE)

    # Progress bar and logger
    progress_bar = None
    ema_helper = None
    z_logger = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)
        # PERF-DEV-09 (recorded, NOT ratified): `config.model_dump()` now also
        # carries the `perf_profiler` / `perf_benchmark` blocks, so every future
        # science run's wandb config gains those two keys.  This touches an
        # artifact downstream figure code reads; see the 2026-07-28 ledger in
        # lab/reports/2026-07-26_experiment-speed-action-plan.md.
        wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump(), settings=wandb.Settings(_disable_stats=True))  # type: ignore
        wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)
        save_code_and_config(config)
        if config.log_z_dynamics:
            # Load dataset.json to get ignore_label_id for label remap
            import json as _json
            _ds_meta_path = os.path.join(config.data_paths[0], "train", "dataset.json")
            with open(_ds_meta_path) as _f:
                _ds_meta = _json.load(_f)
            _ignore_label_id_in_file = _ds_meta.get("ignore_label_id", None)
            _snap_path = config.checkpoint_path if config.z_snapshot else None
            z_logger = ZDynamicsLogger(
                data_path=config.data_paths[0],
                probe_size=config.z_probe_size,
                ignore_label_id_in_file=_ignore_label_id_in_file,
                phase_threshold=config.phase_threshold,
                phase_patience=config.phase_patience,
                checkpoint_path=_snap_path,
            )
            if config.ema:
                print("[z_logging] Probe will use EMA model copy for evaluation.")
    if config.ema:
        print('Setup EMA')
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)

    profiler = TrainingProfiler(
        config.perf_profiler, checkpoint_path=config.checkpoint_path, rank=RANK
    )
    # Inert unless perf_benchmark.enabled and RANK == 0.  Constructed BEFORE
    # `profiler.start()` (a documented deviation from the registered insertion
    # order) so that the profiler/benchmark mutual-exclusion ValueError is raised
    # before any profiler resource is acquired -- the raise happens outside the
    # `try:` below, so a `finally: profiler.stop()` would not run for it.
    #
    # `config.perf_profiler.enabled` is passed rather than `profiler.enabled`
    # because the latter is already False on non-zero ranks, which would make the
    # check unreachable there for a different reason.  The check itself is
    # rank-0-only by design (the frozen collector contract raises "when active
    # and profiler_enabled", and TrainingBenchmark returns early on rank != 0):
    # a non-zero rank with both features enabled stays silent, and the rank-0
    # ValueError is what aborts the job under torchrun.
    bench = TrainingBenchmark(
        config.perf_benchmark,
        checkpoint_path=config.checkpoint_path,
        rank=RANK,
        run_name=config.run_name,
        seed=config.seed,
        eval_interval=config.eval_interval,
        resolved_config=config.model_dump(),
        data_paths=config.data_paths,
        profiler_enabled=config.perf_profiler.enabled,
    )

    if profiler.enabled:
        profiler.start()

    # Training Loop
    try:
        for _iter_id in range(total_iters):
            print (f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {_iter_id * train_epochs_per_iter}")

            ############ Train Iter
            if RANK == 0:
                print("TRAIN")
            train_state.model.train()
            if profiler.enabled:
                # This branch alone adds record_function ranges.  (The default
                # branch below is not otherwise untouched -- see its own
                # comment for the inert benchmark calls it carries.)
                for set_name, batch, global_batch_size in profiled_train_batches(train_loader, profiler):
                    metrics = train_batch_profiled(
                        config, train_state, batch, global_batch_size,
                        rank=RANK, world_size=WORLD_SIZE, profiler=profiler,
                    )

                    if RANK == 0 and metrics is not None:
                        with profiler.record("train/wandb"):
                            wandb.log(metrics, step=train_state.step)
                        progress_bar.update(train_state.step - progress_bar.n)  # type: ignore
                    if config.ema:
                        with profiler.record("train/ema"):
                            ema_helper.update(train_state.model)
                    profiler.step()
            else:
                # When the benchmark is off `batches is train_loader`, so this
                # loop is byte-identical to the original hot path apart from the
                # inert benchmark calls: the spans (each one attribute load +
                # bool test + the shared _NULL_SPAN nullcontext) PLUS the
                # non-span `bench.begin_update(...)` inside train_batch and
                # `bench.end_update(...)` below, which return immediately when
                # disabled.
                batches = bench.iter_batches(train_loader) if bench.enabled else train_loader
                for set_name, batch, global_batch_size in batches:
                    metrics = train_batch(config, train_state, batch, global_batch_size, rank=RANK, world_size=WORLD_SIZE, bench=bench)

                    if RANK == 0 and metrics is not None:
                        with bench.wall_span("metrics_wandb"):
                            wandb.log(metrics, step=train_state.step)
                        progress_bar.update(train_state.step - progress_bar.n)  # type: ignore
                    if config.ema:
                        with bench.cuda_span("ema"):
                            ema_helper.update(train_state.model)
                    bench.end_update(train_state.step)

            if _iter_id >= config.min_eval_interval:
                ############ Evaluation
                if RANK == 0:
                    print("EVALUATE")
                if config.ema:
                    print("SWITCH TO EMA")
                    train_state_eval = copy.deepcopy(train_state)
                    train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)
                else:
                    train_state_eval = train_state
                train_state_eval.model.eval()
                with bench.event_span("eval", train_state.step):
                    metrics = evaluate(config,
                        train_state_eval,
                        eval_loader,
                        eval_metadata,
                        evaluators,
                        rank=RANK,
                        world_size=WORLD_SIZE,
                        cpu_group=CPU_PROCESS_GROUP)

                if RANK == 0 and metrics is not None:
                    wandb.log(metrics, step=train_state.step)

                ############ z dynamics logging (rank 0 only, no-op when log_z_dynamics=False)
                if z_logger is not None:
                    import functools as _functools
                    with bench.event_span("zprobe", train_state.step):
                        z_logger.log(
                            model=train_state_eval.model,
                            step=train_state.step,
                            save_train_state_fn=_functools.partial(save_train_state, config),
                            train_state=train_state_eval,
                        )

                ############ Checkpointing
                if RANK == 0:
                    print("SAVE CHECKPOINT")
                if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
                    with bench.event_span("checkpoint", train_state.step):
                        save_train_state(config, train_state_eval)

                if config.ema:
                    del train_state_eval
    finally:
        profiler.stop()
        # Rows live in memory until here; this is the only write site for
        # steady_state.csv / manifest.json.  No-op when the benchmark is off.
        bench.finalize()

    # finalize
    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    launch()
