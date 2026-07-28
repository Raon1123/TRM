---
id: PERF-001
exp_id: PERF-001
slug: experiment-speed-profiling
hypotheses: []
parent_exp: [EXP-001, EXP-005, EXP-007]
registration_mode: pre-registered
wandb_runs: []
status: planned
date_designed: 2026-07-26
date_closed: ~
scope: "operational performance preregistration; no new sigma^k scientific claim"
---

# PERF-001 — σ^k TRM baseline 성능 프로파일링과 의미론 보존 가속 사전등록 — 2026-07-26

> 이 문서는 신규 모델/데이터 가설 실험이 아니라, 이미 활성인 σ^k baseline을 더 빨리 실행할 수
> 있는지를 검사하는 **운영 사전등록**이다. 따라서 ``faster``와 ``scientifically equivalent``를
> 분리한다. GPU update/eval 성능 수치, GPU 병목 비율, 속도 향상률은 아직 **전부 미측정**이며,
> 어떠한 값도 사전 가정하지 않는다. 환경 snapshot과 CPU-only loader diagnostic은 §4.3에 제한적으로
> 기록한다. 구현·enqueue·GPU 사용은 이 문서의 gate와 독립 verifier 검토 후에만 한다.

| 작성일 | 상태 | 담당 파이프라인 | 연계 H-번호 | 사전등록 |
|---|---|---|---|---|
| 2026-07-26 | planned; profiler/가속 코드 모두 미실행 | 설계: deep-reasoner · 구현: 별도 worker · equivalence audit: independent verifier · 커밋: 사용자 | 없음 — 기존 EXP-001/005/007 baseline의 운영 보조 | pre-registered (아래 profiler와 후보 gate가 GPU 사용 전에 고정) |

## 0. 기호 정의와 범위

| 기호 / 용어 | 정의 |
|---|---|
| \(n\) | 순열의 크기. 본 문서의 canonical dataset에서 **\(n=10\) 고정**이다. |
| \(\sigma\) | \(n=10\) 원소의 입력 순열. |
| \(k\) | 합성 깊이: 목표는 \(\sigma^k\). `data/sigma_k_10/<k>`의 dataset condition이며 모델 입력 토큰은 아니다. |
| baseline | `tf_z_iter`: `arch=trm`, `arch.mlp_t=False`, `H_cycles=3`, `L_cycles=6`, `L_layers=2`, `halt_max_steps=1`인 활성 attention TRM 경로. |
| `trm` / `trm_singlez` | `trm`은 \(z_H\)와 \(z_L\) carry를, `trm_singlez`는 \(z_L\)만 보존하는 weight-tied recurrent 모델이다. 둘은 별도 실행 그래프다. |
| `iter` / `noiter` | 각각 \((H_{\mathrm{cycles}},L_{\mathrm{cycles}})=(3,6)\) / \((1,1)\). iter/noiter는 계산량뿐 아니라 recurrence·credit-assignment를 함께 바꾸므로 속도 tuning의 대체 knob가 아니다. |
| \(B\) | global batch size. 활성 baseline은 \(B=2048\)이다. |
| EMA | 매 optimizer update 뒤 \(\mu=0.999\)로 갱신되는 exponential moving average. 이 baseline에서 eval, z probe, 저장 checkpoint의 가중치 의미론을 결정한다. |
| `probe/test_exact` | EMA-copied 모델로 512-example test probe에서 계산되는 sequence exact accuracy. baseline의 primary outcome이다. |
| `all/exact_accuracy` | online evaluation의 raw-weight full-eval exact metric. `probe/test_exact`와 교환할 수 없는 별도 protocol이다. |
| recompute | 저장된 EMA checkpoint로 다시 계산하는 exact accuracy protocol. probe/online과 표본·가중치가 달라, threshold 근방에서 교차검증용이다. |
| training-equivalent | dataset, sample order, update 수·순서, optimizer/LR, architecture, recurrence, EMA가 보존된다. training-equivalent만으로 evidence-equivalent는 아니다. |
| evidence-equivalent | training-equivalent에 더해 probe/eval/z-dynamics/checkpoint의 정의와 필요한 시점이 보존된다. 이 문서에서 canonical baseline이라고 부르려면 이 수준이 필요하다. |

**범위 밖:** \(H_{\mathrm{cycles}}\), \(L_{\mathrm{cycles}}\), `halt_max_steps`, \(B\), dataset,
seed set, `eval_interval`, probe size, loss precision, optimizer 종류를 바꾸어 wall-clock을 줄이는 일은
속도 최적화가 아니라 **새 실험 조건**이다. 그 결과를 기존 EXP-001/005/007 결과와 동일 baseline으로
합치지 않는다.

## 1. 가설

### 1.1 운영 가설 (조건부)

**P-1 (미검증):** 현재 activation baseline의 update wall time 중 하나 이상의 보존 가능한 구현
경로—per-step metric D2H/W&B, per-parameter EMA, z probe의 중복 forward, compiler graph
fragmentation, 혹은 host-side batch construction—가 측정 가능한 비중을 차지할 수 있다. profiler가
그 경로를 상위 병목으로 보이고, §3의 invariant ledger 및 §2의 scenario gate를 통과할 경우에만
그 경로의 구현을 바꾼다.

이는 효율 가설이지 TRM mechanism 가설이 아니다. profiler가 recurrent BF16 compute 또는 FP64 loss를
주 병목으로 보일 경우에도, recurrence 또는 loss precision을 바꿔서는 기존 baseline을 "가속"했다고
부르지 않는다.

### 1.2 현재 실행 경로의 근거 (정적 code audit; 성능 측정 아님)

| 사실 | current path 근거 | 이 문서에서의 의미 |
|---|---|---|
| `torch.compile`는 기본 on | `pretrain.py:145-150` | "compile 켜기"는 새 최적화 후보가 아니다. graph break/recompile 존재 여부는 **미측정**이다. |
| `B=2048`, `eval_interval=2000` | `config/cfg_pretrain.yaml:18-23`; active job common args `scripts/sigma_enqueue.sh:50-56` | throughput은 effective global examples와 supervised tokens 둘 다 보고해야 한다. |
| active baseline은 H3/L6/L2/halt1 | `scripts/sigma_enqueue.sh:38-42, 64-68, 131-133` | H3/L6 대신 H1/L1을 쓰는 것은 42→4 block forward 변화이며 비교 불가다. |
| iter forward는 `L_level` 21회, block은 2개 | `models/recursive_reasoning/trm.py:149-150, 207-216` | update당 block forward는 \(3(6+1)\times2=42\). 이는 정적 호출수이지 FLOP/시간 비율이 아니다. |
| main recurrent activations는 BF16 | `config/arch/trm.yaml:20`; `models/layers.py:59-60, 77-78` | AMP를 단순 drop-in으로 가정할 수 없다. |
| stablemax loss는 FP64 cast | `models/losses.py:19-31` | FP64→FP32/BF16은 numerical/loss protocol 변경이다. 병목 여부도 **미측정**이다. |
| train metric은 매 update D2H 후 W&B log | `pretrain.py:339-350, 738-754` | host synchronization/log overhead의 후보이나, 규모는 **미측정**이다. |
| EMA는 매 update parameter-name loop | `models/ema.py:16-21`; `pretrain.py:742-754` | launch/allocation 비용 후보. EMA tensor rounding 변화는 primary outcome을 바꿀 수 있다. |
| z logger는 각 split에 두 번 forward | `utils/z_logging.py:163-221, 366-403` | final-state capture와 prediction을 별 forward로 한다. eval-side overhead 후보이나, 절감 가능성은 아직 미검증이다. |
| DataLoader는 worker=1, multiworker를 assert로 거부 | `pretrain.py:112-127`; `puzzle_dataset.py:239-245` | `num_workers>1`은 flag-only drop-in이 아니다. |
| sample batching은 Python loop 및 NumPy sampling | `puzzle_dataset.py:12-39, 201-237` | data stall 후보. 특히 `_sample_batch`의 global `np.random.choice`는 local `Generator`와 별도여서 현재 NumPy RNG 재현성은 약하다; sample-order 변경을 허용하지 않는다. |
| EMA eval 및 checkpoint 저장 | `pretrain.py:760-796`; `models/ema.py:30-33` | 저장 checkpoint는 raw training model이 아니라 EMA-copied evaluation model이다. |

**현재 live-path 확인 (2026-07-26, 성능 측정 아님):** queue job `0207`은 `k=6`, \(B=2048\),
H3/L6/L2/halt1, `ema=True`, `log_z_dynamics=True`이지만 **`arch.mlp_t=True`인 MLP z-iter
diagnostic**이다 (`scripts/queue/processing/0207_fig1_mlp_z_iter_k6_s1.job.gpu4`). 본 문서의 canonical
attention tf baseline(`arch.mlp_t=False`)이 아니며, profiler control이나 representative throughput
sample로 사용할 수 없다.

### 1b. 관련 근거 자료 (web · proof · theory canon)

외부 논문/성능 수치가 이 계획의 근거가 되지는 않는다. 아래는 기존 baseline의 의미론을 고정하기 위한
내부 정본이다.

| 종류 | 식별자·링크 | 이 문서에서 쓰는 주장 | 상태 |
|---|---|---|---|
| canon | `lab/experiments/EXP-001_fig12-z-iter-ablation.md` §0, §3 | z×iter baseline, H-collapse 정의, canonical hyperparameter lineage | 기존 experiment record |
| canon | `lab/experiments/EXP-005_fig1-mlp-block-ablation.md:105, 168-184` | EMA/probe/online/recompute 세 protocol의 비등가성과 fractional-cell 교차검증 규율 | 기존 analysis record |
| canon | `lab/experiments/EXP-007_k9-k11-cycletype-discriminator.md:41, 251-279` | `probe/test_exact` decision metric과 final-step/프로토콜 caveat | 기존 analysis record |
| code | `pretrain.py`, `models/recursive_reasoning/trm.py`, `utils/z_logging.py` | 이 문서 §1.2의 실행 경로 | 정적 audit; runtime 성능은 미검증 |
| memory / project rule | `AGENTS.md` §2, §4–5 | n=10, queue, pre-registration, independent verifier, completion gate | 운영 규율 |

## 2. 예상 결과 시나리오와 대응

모든 "병목" 시나리오는 profiler 이전에는 **UNCOVERED/미측정**이다. profiler 결과가 아래 어느
pattern에도 맞지 않으면 **S-0 UNCOVERED**로 기록하고, 사후 rationalization 없이 새 사전등록
amendment를 작성한다.

| 시나리오 | 사전등록된 관측 pattern | 허용 해석 | 대응 |
|---|---|---|---|
| S-1 GPU recurrent-compute dominated | steady-state에서 forward+backward가 update CUDA time의 지배항, host gap 작음 | H3/L6/H1/L1 compute 차이가 runtime을 설명할 수 있음. 이는 iter의 과학적 효과와 별개다. | recurrence/hidden/sequence/loss precision을 바꾸지 않는다. kernel 후보는 profiler 최상위 op가 있을 때만 별도 gate로 검토한다. |
| S-2 logging/D2H dominated | `metrics.cpu().numpy()`·`wandb.log` 구간 또는 GPU idle gap이 의미 있게 관찰됨 | train-side observability 구현이 throughput을 제한할 수 있음 | train metric batching 후보만 A/B한다. eval/probe/z logs는 그대로 보존한다. |
| S-3 EMA dominated | optimizer 뒤 EMA update가 반복적으로 상위 CUDA/CPU 항목 | EMA implementation overhead 가능 | foreach/vectorized EMA는 EMA values·checkpoint·probe를 strict gate로 비교한다. |
| S-4 eval/z-probe dominated | update는 빠르지만 2000-epoch event에서 EMA deepcopy/probe/PCA/W&B/ckpt가 campaign time을 지배 | training compute가 아니라 measurement overhead | 중복 probe forward 제거만 우선 검토; probe size/cadence/primary metric은 바꾸지 않는다. |
| S-5 input/host dominated | next-batch wait 또는 CPU sampling이 GPU idle과 동반 | mmap I/O가 아니라 Python batching/host transfer일 수 있음 | 먼저 pin/H2D overlap을 계측; multiworker/vectorization은 sample-index trace가 동일할 때만 후보가 된다. |
| S-6 compile pathology | trace에서 graph breaks/recompilation 혹은 compiled path의 반복 CPU overhead | existing `torch.compile` path가 최적이 아닐 수 있음 | graph cause를 기록; checkpoint key compatibility와 4-cell numerical gate 후에만 변경한다. |
| S-7 no actionable preserved bottleneck | top cost가 FP64 stablemax, recurrence, larger batch requirement, or effect size가 noise 이하 | throughput 개선이 baseline semantics를 요구할 수 있음 | **no change**. 새 loss/precision/batch experiment를 원하면 별도 EXP preregistration으로 분리한다. |
| S-0 UNCOVERED | 위 pattern 어느 것도 충족하지 않음, 혹은 candidate가 numerical/evidence gate 실패 | 현재 plan의 가정 불충분 | candidate 폐기, raw profiler 증거 보존, amendment와 independent review 전까지 GPU rollout 중지 |

### Confound ledger (해석 전 필수)

- **iter ↔ recurrent compute:** H3/L6와 H1/L1은 호출수도 다르고 learning dynamics도 다르다. H1/L1의
  빠른 update time을 H3/L6 baseline의 speedup으로 보고하지 않는다.
- **three `test_exact` protocols:** primary=`probe/test_exact` (EMA, 512); corroboration=`all/exact_accuracy`
  (raw online); threshold/discordance audit=recompute (EMA checkpoint). 후보가 이 중 어느 하나의 정의·시점을
  바꾸면 evidence-equivalent 실패다.
- **cycle-type / \(\gcd(k,\operatorname{ord}\sigma)\) axis:** profiling canary의 한 \(k\)가 모든
  \(k\)에서 scientific invariance를 확립하지 않는다. promotion에는 intended \(k\) grid 전체에서
  protocol 분류를 재확인한다.
- **NumPy RNG:** 현재 data path의 NumPy global RNG 사용은 약점이지 최적화 승인권이 아니다. vectorization,
  prefetch, worker 변경이 sample indices를 바꾸면 새 data-order protocol이다.

## 3. 코드/설정 수정 내역과 invariant/equivalence ledger

### 3.1 현 상태

이 사전등록을 작성한 시점에 profiler, training code, queue job, config, dataset은 **수정하지 않았다**.
아래 표의 후보는 모두 `planned`, 실제 파일/line은 구현 후에 append한다. 이 보고서 외 파일을 이
planning 단계에서 변경하지 않는다.

| 파일:위치 | planned modification | 이유 | 대안과 pros/cons |
|---|---|---|---|
| TBD `pretrain.py` timing regions | CUDA event/CPU wall timing과 `record_function` 추가 | 병목을 분해하되 science path와 분리 | 외부 profiler만 사용하면 코드 영향은 작지만 data/EMA/log 경계가 불명확하다. instrumentation은 별도 profiling mode로 한정한다. |
| TBD `pretrain.py` train logging | 매 update W&B 대신 사전등록된 cadence/aggregation | S-2일 때 host sync 감소 | 가장 낮은 semantic risk이지만 per-step train trajectory는 사라진다. eval/probe log는 절대 변경하지 않는다. |
| TBD `utils/z_logging.py` | 동일 forward에서 z capture와 prediction을 얻도록 중복 pass 제거 | S-4일 때 eval overhead 감소 | probe size/cadence 축소는 더 빠르지만 evidence protocol 변경이므로 채택하지 않는다. |
| TBD `models/ema.py` | profiler가 S-3일 때만 semantically matched EMA implementation 검토 | per-parameter loop/temporary allocation 감소 가능성 | raw model만 저장/eval하면 빠르지만 EMA baseline을 바꾸므로 금지. |
| TBD compiler config | graph-break 원인에 한해 compiler setting 수정 | existing compile path의 fragmentation 가능성 | `DISABLE_COMPILE`는 진단 control이지 promotion target이 아니다. |

### 3.2 불변성 ledger

| 층 | 반드시 보존할 항목 | 검증 증거 / gate | 위반하면 |
|---|---|---|---|
| Dataset | `data/sigma_k_10/<k>` bytes/metadata, \(n=10\), train/test split, label remap, selected sample-index sequence | data-file hash + per-update sample-index trace hash + dataset metadata | new data/data-order experiment |
| Model | arch (`trm`/`trm_singlez`), `mlp_t`, hidden size, L2, H/L cycles, halting, BF16 path | resolved Hydra config diff + fixed-batch logits/preds | new model/compute experiment |
| Update | \(B=2048\), effective batch, optimizer class/order, LR each update, total updates, gradient scaling, optimizer-step count | first/last 32 update ledger; full run step count | new optimization experiment |
| EMA | \(\mu=0.999\), update once per optimizer step, EMA eval/copy, saved EMA state semantics | EMA tensor/output comparison; final checkpoint reload | evidence-equivalent failure |
| Evaluation | eval interval 2000, full eval, 512 train/test z probes, phase threshold/patience, probe hashes | W&B key/time grid and z-probe artifact comparison | measurement-protocol change |
| Checkpoint | final checkpoint plus current forced phase transition behavior; `checkpoint_every_eval=False`, `z_snapshot=False`; compiled key compatibility | `all_config.yaml`, state-dict key/load smoke, checkpoint timestamps | checkpoint semantic change |
| Reporting | existing `probe/*`, `phase/index`, `z/*`, `all/exact_accuracy` semantics and candidate runtime tag | W&B schema validation | evidence provenance break |

### 3.3 Non-equivalence ledger (explicitly not "free speed")

| Proposed action | Why it changes the experiment | Required treatment |
|---|---|---|
| H/L cycles, L layers, halt max, hidden size, sequence length 변경 | recurrence/compute and potentially learned function change | new EXP, not PERF-001 promotion |
| global batch size change or grad accumulation | update count, LR schedule, data grouping, optimizer/EMA timing change | new EXP |
| stablemax FP64 → FP32/BF16; TF32/autocast toggle | loss arithmetic/numerics change; model already uses explicit BF16 casts | numerical variant EXP |
| AdamAtan2 → fused Adam/AdamW | optimizer algorithm/state changes | optimizer variant EXP |
| fewer evals, smaller z probe, disabled z logging, changed z snapshots/checkpoints | `test_exact`/phase/evidence resolution changes | measurement protocol amendment or new EXP |
| DataLoader worker count/vectorized sampler/preloaded shuffled batches without index equality | data ordering/RNG evolution changes | new data-order protocol |
| DDP/multi-GPU as a flag-only launch | gradients, reductions, device topology, loader sharding and checkpoint/compile behavior change | separate distributed equivalence study |

## 4. Run 매니페스트와 profiler protocol

### 4.1 Canonical command identity

The active baseline is represented by the current `sigma_enqueue.sh` common arguments and tf/z/iter cell,
not by an invented shorter benchmark. The implementation owner must capture the exact emitted job body via
`--dry-run` before any run, then attach its resolved config to the performance artifact.

| field | registered value |
|---|---|
| primary cohort | `tf_z_iter`: `arch=trm`, `arch.mlp_t=False`, H3/L6/L2/halt1 |
| diagnostic coverage cells | `trm`/`trm_singlez` × H3/L6/H1/L1, all with `mlp_t=False`; not pooled as a scientific comparison |
| common training | `epochs=100000`, `eval_interval=2000`, `lr=1e-4`, `puzzle_emb_lr=1e-4`, both weight decays 1.0, `ema=True`, \(B=2048\) |
| data | `data/sigma_k_10/<k>`; \(n=10\). Profile \(k\) is recorded, never silently substituted. |
| logging/checkpoint | `+log_z_dynamics=True`, `+z_snapshot=False`, `checkpoint_every_eval=False`, `evaluators=[]` |
| compiler | default `torch.compile`; compiler setting/environment captured verbatim |
| run name schema | `perf0_<baseline-or-cell>_k<k>_s<seed>_<runtime-tag>`; runtime tag contains candidate and code/config hash |
| W&B project | new `Sigma_k_perf` only after equivalence canary passes; never overwrite/merge with canonical science runs |
| queue | no production enqueue in this registration; queue prefix and GPU allocation are TODO after review |

### 4.2 Command matrix

All shell commands use the project-required `rtk` prefix. Rows marked **TBD implementation** are design
contracts, not commands that currently exist; they must not be executed before the corresponding code and
independent audit are present.

| stage | command / action | writes / GPU | pass record |
|---|---|---|---|
| M0 provenance | `rtk git status --short`; `rtk scripts/sigma_enqueue.sh --dry-run perf0` | dry-run only; no GPU | emitted job body, worktree state, resolved config fields |
| M0 static gate | `rtk bash -n scripts/sigma_enqueue.sh scripts/queue_run.sh` | no GPU | PASS/FAIL in artifact log |
| M1 hardware capture | `rtk nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader`; record PyTorch/CUDA/compiler env through a reviewed helper | read-only | environment snapshot captured in §4.3; TF32/determinism and performance counters remain evidence pending |
| M2 steady microbenchmark | **TBD implementation:** reviewed timing mode runs 50 warm-up + 200 measured updates with fixed config | GPU; fully enabled EMA/W&B/z performance-only run | raw per-update CSV, compile time separated; off-component diagnostics reported separately |
| M3 PyTorch trace | §4.2.1의 등록 명령: `torch.profiler` wait=10/warmup=10/active=20, CPU+CUDA+memory+shapes/stacks | GPU; profiler perturbation allowed | Chrome/TensorBoard trace + resolved config; not science evidence |
| M4 system/kernel trace | Nsight Systems after M3; Nsight Compute on only top one/two kernels | GPU; separate profiling reservation | launch-gap and kernel report |
| G1 frozen-state gate | **TBD implementation:** fixed-batch forward/backward/step/EMA/probe comparison under baseline vs one candidate | GPU optional; no queued science run | tensor/output comparison and checkpoint load smoke |
| G2 full canary | reviewed job body, full 100000-epoch canonical run, paired baseline/candidate | GPU | complete W&B/checkpoint/equivalence ledger |
| G3 enqueue gate | `rtk scripts/sigma_enqueue.sh --dry-run <approved-prefix>`; independent verifier audits exact job diff | dry-run only | signed PASS; only then actual enqueue command |

#### 4.2.1 Registered M3 canonical-path diagnostic command

The code defaults—wait 1, warmup 1, active 3, `record_shapes=false`, `with_stack=false`
(`config/cfg_pretrain.yaml:46-59`)—are intentionally small **safe-smoke defaults**. They do not satisfy this
pre-registration's M3 protocol and may not be used for its canonical trace. M3 must use every explicit override
below. `rank0_only` is not a config option: `TrainingProfiler` always instantiates and writes on rank 0 only
(`utils/perf_profiler.py:90-96`), while nonzero ranks retain the collective training path.

The following is the registered M3 command. It preserves the canonical per-step tf/z/iter path (B2048,
H3/L6/L2/halt1, BF16, compile default-on, EMA, per-step W&B, z logging) but bounds the campaign to
`epochs=20, eval_interval=20`. It is therefore a **performance-only diagnostic**, not a canonical science run;
its accuracy, phase, or checkpoint must not enter EXP-001/005/007 evidence.

```bash
rtk uv run pretrain.py arch=trm \
  global_batch_size=2048 epochs=20 eval_interval=20 min_eval_interval=0 \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  ema=True ema_rate=0.999 evaluators="[]" checkpoint_every_eval=False \
  arch.mlp_t=False arch.H_cycles=3 arch.L_cycles=6 arch.L_layers=2 \
  arch.halt_max_steps=1 arch.forward_dtype=bfloat16 \
  data_paths="[data/sigma_k_10/6]" seed=1 +k=6 \
  +log_z_dynamics=True +z_snapshot=False \
  +project_name=Sigma_k_perf +run_name=perf0_m3_tf_z_iter_k6_s1 \
  perf_profiler.enabled=True perf_profiler.performance_only=True \
  perf_profiler.output_dir=reports/figures/2026-07-26_experiment-speed-profiling/profiles/m3_tf_z_iter_k6_s1 \
  perf_profiler.wait=10 perf_profiler.warmup=10 perf_profiler.active=20 \
  perf_profiler.repeat=1 perf_profiler.max_steps=40 \
  perf_profiler.record_shapes=True perf_profiler.profile_memory=True \
  perf_profiler.with_stack=True perf_profiler.export_chrome_trace=True \
  perf_profiler.export_tensorboard_trace=True
```

Before queue submission, the exact same overrides plus `--cfg job` must be rendered and saved as a sibling
artifact. The submitted job body and this file must be byte-compared for the registered fields; relying on YAML
defaults is not sufficient.

```bash
rtk mkdir -p reports/figures/2026-07-26_experiment-speed-profiling/profiles/m3_tf_z_iter_k6_s1
rtk uv run pretrain.py arch=trm \
  global_batch_size=2048 epochs=20 eval_interval=20 min_eval_interval=0 \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  ema=True ema_rate=0.999 evaluators="[]" checkpoint_every_eval=False \
  arch.mlp_t=False arch.H_cycles=3 arch.L_cycles=6 arch.L_layers=2 \
  arch.halt_max_steps=1 arch.forward_dtype=bfloat16 \
  data_paths="[data/sigma_k_10/6]" seed=1 +k=6 \
  +log_z_dynamics=True +z_snapshot=False \
  +project_name=Sigma_k_perf +run_name=perf0_m3_tf_z_iter_k6_s1 \
  perf_profiler.enabled=True perf_profiler.performance_only=True \
  perf_profiler.output_dir=reports/figures/2026-07-26_experiment-speed-profiling/profiles/m3_tf_z_iter_k6_s1 \
  perf_profiler.wait=10 perf_profiler.warmup=10 perf_profiler.active=20 \
  perf_profiler.repeat=1 perf_profiler.max_steps=40 \
  perf_profiler.record_shapes=True perf_profiler.profile_memory=True \
  perf_profiler.with_stack=True perf_profiler.export_chrome_trace=True \
  perf_profiler.export_tensorboard_trace=True --cfg job \
  | rtk tee reports/figures/2026-07-26_experiment-speed-profiling/profiles/m3_tf_z_iter_k6_s1/resolved_config.yaml
```

### 4.3 Environment, queue ownership, and CPU-only diagnostic snapshot (2026-07-26)

This subsection is provenance, not a profiler result. **GPU smoke was intentionally not run.** At the snapshot,
queue-owned GPUs 2–7 were all `processing` and showed 98–100% utilization; GPU0 was occupied by another user
at 99%. GPU1 appeared idle in device telemetry, but is not available for this work because the queue ownership
rule reserves it to another user. Therefore no GPU memory, kernel, step-time, W&B, EMA, z-logger, or
train-throughput measurement may be inferred from this snapshot.

| item | observed value | scope / limitation |
|---|---|---|
| GPU fleet | 8 × RTX 6000 Ada, 49,140 MiB each | hardware inventory only; no exclusive device acquired |
| active ownership | GPU2–7 queue `processing`; GPU0 external user; GPU1 reserved despite apparently idle telemetry | protects other jobs; GPU profiling requires a separately queued, approved job |
| accelerator runtime | Torch `2.10.0+cu126`, CUDA `12.6`, cuDNN `91002` | compile backend, TF32 and deterministic flags remain TODO |
| host | 2 × EPYC 7763, 128 CPUs, RAM 1.0 TiB | inventory only; NUMA/pinning and load were not profiled |
| CPU loader: B=2048 warmup batch | 516.753 ms | one warmup observation; includes first-payload/worker effects and is not canonical iterator startup |
| CPU loader: B=2048 measured batches | \(n=19\); mean 30.964, median 29.947, p95 40.128, min 29.190, max 40.128 ms | CPU-only; no H2D overlap/GPU utilization, so neither a step-time share nor a throughput claim |
| CPU loader: B=128 warmup batch | 30.615 ms | one diagnostic warmup observation; B=128 is not canonical baseline |
| CPU loader: B=128 measured batches | \(n=99\); mean 4.928, median 4.825, p95 5.822, min 4.321, max 5.965 ms | diagnostic scaling point only; not an allowed baseline batch-size change |

Source: [`dataloader_smoke.json`](../../reports/figures/2026-07-26_experiment-speed-profiling/dataloader_smoke.json),
generated by [`analysis/dataloader_smoke.py`](../../analysis/dataloader_smoke.py). The JSON records raw batch
timings, nearest-rank p95 convention, environment and dirty-worktree provenance. Reproduction command:

```bash
rtk uv run python analysis/dataloader_smoke.py \
  --output reports/figures/2026-07-26_experiment-speed-profiling/dataloader_smoke.json
```

The diagnostic deliberately uses `epochs_per_iter=20`. The live loader constructs a group order by
concatenating `epochs_per_iter` permutations (`puzzle_dataset.py:206-210`), while the canonical live value is
2000 (10 million group entries for 5,000 train groups). Therefore the raw artifact **does not emulate or
measure canonical startup**, and its warmup value cannot be extrapolated to that startup. The only limited
inference allowed here is that a CPU loader component is worth profiling under the real configuration; it does
**not** establish S-5 or authorize a sampler/worker change.

### 4.4 Measurement protocol

1. **Warm-up and synchronization.** Report compile/startup time separately. For steady-state timings, discard
   the pre-registered 50 updates. CUDA events delimit H2D, forward, backward, optimizer and EMA; synchronize
   only once per measured interval/window. CPU monotonic timing separately measures next-batch wait, W&B,
   checkpoint and total wall time. A per-step `cuda.synchronize()` in production is forbidden.
2. **Repetition and variance.** Each timing condition uses at least 200 post-warm-up updates. Report N,
   median, mean, p10/p90, p95, and coefficient of variation, rather than a single fastest time. Compiler
   first-run is a separate row. A candidate needs a speed improvement larger than the predeclared baseline
   noise envelope; exact threshold is set after M2 baseline repetitions and before candidate data are read.
3. **Profiler layering.** M2 establishes unprofiled throughput; M3 attributes it; M4 is only used when M3
   leaves an actionable kernel/launch ambiguity. No profiler number is used as canonical campaign throughput.
4. **GPU telemetry.** Capture utilization, SM occupancy when available, power/clock state, device memory,
   `torch.cuda.max_memory_allocated`, and `max_memory_reserved`. §4.3 has an environment/ownership snapshot,
   but **no GPU training measurement** exists at registration time. Missing counters are written as
   `evidence_pending`, never inferred from model size.
5. **Timing boundaries.** Evaluation, EMA deepcopy, two z-probe splits, PCA/scatter image production, checkpoint
   serialization, and W&B are reported separately from train updates and also as campaign amortized time.
6. **Component-bound diagnostics.** Any W&B-off, EMA-off, or z-logging-off run isolates a component only. It is
   not the canonical M3 path, cannot supply baseline throughput, and cannot be promoted as a scientific-equivalent
   run without rerunning the fully enabled §4.2.1 command.

### 4.5 Performance metrics

| metric | definition | aggregation / caveat |
|---|---|---|
| `perf/update_ms` | end-to-end wall milliseconds per optimizer update | median/mean/p95 after warm-up; compile excluded and reported separately |
| `perf/samples_per_s` | effective global examples ÷ update wall time | use yielded `global_effective_batch_size`, not nominal B if it differs |
| `perf/input_tokens_per_s` | input examples × input sequence length ÷ update wall time | not interchangeable with supervised tokens |
| `perf/target_tokens_per_s` | non-ignore labels ÷ update wall time | primary token throughput when label masking matters |
| `perf/h2d_ms`, `forward_ms`, `backward_ms`, `optimizer_ms`, `ema_ms` | CUDA-event intervals | sum need not equal wall time because of overlap/asynchrony |
| `perf/data_wait_ms`, `wandb_ms`, `eval_ms`, `zprobe_ms`, `checkpoint_ms` | CPU/wall intervals | report per event and amortized per update |
| `perf/gpu_util_pct`, `perf/gpu_mem_*` | device telemetry and allocator peak | p50/p95 plus sampling interval and tool version |
| `perf/compile_ms`, `perf/graph_break_count` | startup compiler cost / diagnosed graph fragmentation | compiler evidence only; not steady-state speed |

## 5. Figure 계획 및 W&B mapping

No performance W&B keys exist yet. The table distinguishes keys already emitted by the canonical run from
planned keys that require implementation. Planned keys may not be backfilled or treated as present before G1.

| Figure / artifact | content | W&B key / source | filter·aggregation |
|---|---|---|---|
| F1 `step-time-breakdown` | baseline/candidate stacked update decomposition | planned `perf/*` raw CSV + W&B `perf/*` | same GPU/runtime; median and p10/p90 over 200 warm updates |
| F2 `campaign-time-waterfall` | train vs eval/EMA/z-probe/checkpoint amortized cost | planned `perf/eval_ms`, `perf/zprobe_ms`, `perf/checkpoint_ms` | one canonical full run per condition; no scientific pooling |
| F3 `util-memory` | utilization and allocated/reserved memory over time | planned telemetry CSV, `perf/gpu_*` | 1-s samples; report missing intervals |
| F4 `equivalence-trajectory` | baseline vs candidate primary/corroboration metrics | existing `probe/train_exact`, `probe/test_exact`, `phase/index`, `z/eff_rank`, `all/exact_accuracy`; checkpoint recompute table | paired seed/cell; evaluation points aligned by optimizer step |
| A1 profiler trace | operator/launch attribution | local trace under `reports/figures/2026-07-26_experiment-speed-profiling/` | profiling-only, not W&B scientific evidence |
| A2 ledger | config/data/sample/EMA/checkpoint equality | local CSV/JSON under the same figure-data artifact root | fixed candidate, seed, cell, commit |

**Required W&B provenance mapping:** every candidate run logs `runtime/candidate_id`, `runtime/code_hash`,
`runtime/resolved_config_hash`, `runtime/data_manifest_hash`, `runtime/gpu`, `runtime/torch_cuda`,
`runtime/compile_mode`, and `runtime/profiling_mode`. Existing science keys retain their spelling and step axis;
performance keys never overwrite them.

## 6. 결과 기입란 (실험 후)

### 6.1 Candidate priority and gates

| priority | candidate | expected benefit | status now | equivalence class | promotion gate |
|---|---|---|---|---|---|
| P0 | measurement-only timing/profiler | diagnosis only | unrun | no science change if isolated | M0–M4 artifacts complete; no claim about speed before M2 |
| P1 | batch train W&B/metric D2H | S-2이면 host synchronization 감소 가능 | unrun; effect **unknown** | training-equivalent; evidence trajectory changes unless documented | G1 exact fixed-batch state; G2 retains every eval/probe/z key and full final evidence |
| P1 | eliminate redundant z-probe prediction forward | S-4이면 eval time 감소 가능 | unrun; effect **unknown** | measurement implementation candidate | frozen EMA checkpoint: predictions, exact, z tensors/deltas/PR/phase all match; then G2 |
| P1 | semantically matched EMA vectorization | S-3이면 overhead 감소 가능 | unrun; effect **unknown** | numerical implementation candidate | EMA tensors/model outputs/checkpoint reload and full trajectories within predeclared baseline envelope |
| P2 | fix verified `torch.compile` graph pathology | S-6일 때 only | unrun; graph evidence pending | compiler/numerical candidate | 4-cell G1, `_orig_mod` state dict load/save, G2 |
| P2 | sample pipeline optimization | S-5일 때 only | unrun | data-order candidate | complete index trace equality; otherwise reject as baseline speedup |
| P3 | kernel/CUDA graphs/cast rework | only after top-op attribution | unrun | high-risk numerical/capture candidate | full G1/G2 plus capture/RNG/optimizer/EMA audit |
| P3 | multi-GPU | unknown; small model may be communication-bound | unrun | distributed protocol candidate | separate DDP study; no baseline label without full evidence gate |
| reject from this plan | FP64 loss change, AMP/TF32, fused optimizer, batch/accumulation, recurrence/eval/probe reduction | may be fast | not candidates | changed experiment | write a separate preregistration |

### 6.2 Minimum controls and acceptance rule

**G1 — frozen-state gate.** For each `trm`/`trm_singlez` × H3/L6/H1/L1 cell, use recorded fixed batches to
compare baseline and one candidate: sample-index hash; logits/loss/predictions; gradients and post-step state;
EMA tensors; z probe output; saved checkpoint keys and reload. Logging-only candidates require equality of all
model/evaluation outputs. Numerical/compiler candidates use tolerances defined from repeated unchanged baseline
runs **before** candidate outcome inspection.

**G2 — paired full-run canary.** Use exactly one predeclared \(k\), one seed, and all four z×iter cells. The
choice of \(k\), GPU/runtime environment, and candidate is frozen in the dry-run manifest. It is a coverage
canary, not evidence that cycle-type/gcd effects are invariant across \(k\). Require all of:

1. same resolved configuration except the reviewed speed field, same data/sample-index evidence, B, update/LR
   ledger, EMA update count/rate, total optimizer steps, and checkpoint semantics;
2. all primary `probe/*`, `phase/index`, z scalar metrics and `all/exact_accuracy` exist on the same evaluation
   step grid; recompute is run for any threshold-near or discordant cell;
3. no changed G/C or H-collapse classification, and no unpredeclared phase-transition discrepancy;
4. candidate speed gain exceeds the baseline noise envelope, with the full distribution reported;
5. independent verifier reproduces the ledger from disk/local W&B, not merely the implementer’s summary.

**G3 — promotion cohort.** Only candidates passing G2 can run the intended full \(k\) grid. Candidate and
legacy runs are tagged separately; they are never silently pooled for z/iter, cycle-type, or collapse claims.

### 6.3 Expected artifacts (GPU/profile artifacts TODO unless marked present)

| artifact | planned location | completion condition |
|---|---|---|
| CPU-only loader diagnostic (**present**) | `reports/figures/2026-07-26_experiment-speed-profiling/dataloader_smoke.json` | raw timings + generation command and environment/git provenance |
| resolved baseline/candidate manifests | `reports/figures/2026-07-26_experiment-speed-profiling/manifests/` | emitted job, resolved config, git/data/runtime hashes |
| M3 resolved profiler config | `reports/figures/2026-07-26_experiment-speed-profiling/profiles/m3_tf_z_iter_k6_s1/resolved_config.yaml` | exact §4.2.1 overrides rendered with `--cfg job` before queue submission |
| unprofiled timing CSV | `reports/figures/2026-07-26_experiment-speed-profiling/data/steady_state.csv` | 200 post-warm-up rows/condition, timing metadata |
| profiler traces and top-op tables | `reports/figures/2026-07-26_experiment-speed-profiling/profiles/` | M3/M4 export with tool versions |
| equivalence ledger | `reports/figures/2026-07-26_experiment-speed-profiling/data/equivalence_ledger.csv` | G1/G2 checks, pass/fail, tolerances frozen before candidate read |
| figures | `reports/figures/2026-07-26_experiment-speed-profiling/*.png` | generated from corresponding CSV/trace summaries |
| verifier report | `lab/audits/YYYY-MM-DD_perf001-equivalence-verifier.md` | independently recomputed acceptance checklist |

### 6.4 Rollout / rollback

1. **Observe:** M0–M4 only. If no preserved-path bottleneck is actionable (S-7), stop without code change.
2. **One change:** implement one P1/P2 candidate behind an explicit runtime switch; run G1 then G2. Do not
   combine logging, EMA and z-probe changes in one comparison.
3. **Promote:** after independent G2 PASS, create a new queue prefix and `Sigma_k_perf` runtime tag. Original
   canonical job templates remain untouched.
4. **Rollback:** immediately disable the candidate switch and mark the artifact `noncanonical` if any invariant,
   checkpoint compatibility, metric schema, G/C/phase classification, or speed-over-noise gate fails. Preserve
   failed profiling evidence; do not overwrite baseline W&B/checkpoints.
5. **Completion gates:** before enqueue, run `rtk bash -n scripts/*.sh` as applicable and
   `rtk scripts/sigma_enqueue.sh --dry-run <approved-prefix>`; after a candidate checkpoint exists, run the
   relevant `rtk uv run pytest tests/` and a `measure_rho.py` smoke path if the candidate touches checkpoint/model
   semantics. An independent verifier, not the implementer, closes the equivalence gate.

### 6.5 Result fields (leave blank until evidence exists)

- [x] M1 partial environment/ownership provenance captured (§4.3); GPU performance counters pending:
- [ ] M2 baseline timing distribution:
- [ ] Scenario classification (S-0 … S-7):
- [ ] Candidate selected and pre-change threshold frozen:
- [ ] G1 frozen-state result:
- [ ] G2 full canary result:
- [ ] G3 promotion/rollback decision:
- [ ] Independent verifier report:

## §7 Postmortem

> Fill only after the planned work closes. Until then this section is intentionally not a result claim.

### §7a. Hypothesis ↔ result

**Pre-registered prediction:** P-1 is conditional: at least one implementation path may be actionable, but the
identity and size of every bottleneck are unknown. A preserved-path speedup is acceptable only after G1–G3.

**Actual result:** TODO — no profiler or GPU throughput measurement exists at registration time.

**판정:** [ ] supported / [ ] not supported / [ ] inconclusive / [ ] S-0 UNCOVERED

**Mechanism analysis:** TODO. If P-1 misses, distinguish (i) recurrent/FP64 intrinsic compute domination,
(ii) candidate implementation not reducing the measured bottleneck, (iii) profiling/synchronization artifact,
and (iv) numerical/evidence gate failure. Do not restate update-time numbers as explanation.

### §7b. Bug log

| # | bug / discrepancy | root cause | prevention |
|---|---|---|---|
| 1 | TODO | TODO | TODO |

### §7c. Lessons → memory candidates

- 보류: profiler and independent equivalence verification are unrun; no general rule is yet verified.
