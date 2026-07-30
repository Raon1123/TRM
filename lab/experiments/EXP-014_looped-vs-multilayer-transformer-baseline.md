---
id: EXP-014               # 2026-07-30 재번호: 초안은 EXP-010이었으나 main의 EXP-010(fig1-clean-rerun)과 충돌 — 내용 불변
slug: looped-vs-multilayer-transformer-baseline
hypotheses: [H-022, H-023]
parent_exp: EXP-005      # fig1 block ablation — 이 실험은 fig1 grid에 외부 baseline 축을 추가한다
registration_mode: pre-registered
wandb_runs: []           # run-name 스키마는 §4
status: planned
date_designed: 2026-07-22
date_closed: ~
---

# Looped transformer vs multi-layer transformer — TRM 비교 baseline — 2026-07-22

| 작성일 | 상태 | 담당 파이프라인 | 연계 H-번호 | 사전등록 |
|---|---|---|---|---|
| 2026-07-22 | enqueue 대기 (GPU 미지출) | Opus 설계·구현 · 사용자 커밋/기동 | H-022, H-023 | pre-registered (전 cell 미실행) |

> **범위 선언**: 이 문서는 *준비*까지다. 코드·config·그리드·검증은 완료했고 **job 파일 생성과
> GPU 기동은 하지 않았다**. 58 cell 전부가 미실행이므로 `pre-registered` credit은 온전하다.

## 0. 기호 정의

| 기호 | 정의 |
|---|---|
| **순열의 크기 n** | σ 가 작용하는 집합의 크기. 이 실험 전체에서 **n = 10 고정** (`data/sigma_k_10/<k>`). |
| **합성 깊이 k** | 학습 목표가 σ^k 인 지수. k ∈ {3,4,5,6,7,8,10}. |
| **블록 깊이 ℓ** | weight-tie 되는 단위 블록 안의 transformer layer 수 (`arch.H_layers`). |
| **루프 수 T** | 그 블록을 forward 1회에 반복 적용하는 횟수 (`arch.H_cycles`). |
| **유효 깊이 D** | D = ℓ · T. 한 forward에서 적용되는 transformer layer 총 횟수. |
| **untied / tied** | untied = layer마다 독립 파라미터(= multi-layer). tied = 같은 파라미터 재사용(= looped). |
| **full BPTT** | T 회 루프 전체에 gradient가 흐름. TRM의 **1-step gradient**(앞 H_cycles−1 을 `torch.no_grad()` 로) 와 대비. |
| **injection schedule** | input embedding을 매 루프마다 더하는지(every) 첫 루프에만 더하는지(first). |
| **puzzle_emb_len** | puzzle embedding이 차지하는 시퀀스 위치 수. 실제 시퀀스 길이 = `seq_len(11) + puzzle_emb_len`. **`trm.yaml`=16 → 27 위치, `transformers_baseline`/`looped_transformer` 기본(ceil-div)=1 → 12 위치.** |

> ⚠️ D는 **파라미터 수와 독립**이다. 이 실험의 전체 설계는 "D를 맞추면 파라미터가 어긋나고,
> 파라미터를 맞추면 D가 어긋난다"는 사실을 정면으로 다루기 위해 **matched pair**로 짜였다.

## 1. 가설

**주 질문**: TRM의 성능은 (a) *weight-tied recurrence 그 자체*에서 오는가, (b) 단순히 *유효 깊이*에서
오는가, 아니면 (c) TRM 고유 요소(z-carry / 1-step gradient / 매 cycle injection)에서 오는가?

문헌(`lab/literature/2026-07-17_bg_weight-tied-expressivity.md`)은 TRM의 `arch=trm`/`trm_singlez`가
**"looped transformer"라는 잘 연구된 architecture genus의 한 instance**라고 이미 판정했다. 그러나
이 repo에는 **정작 canonical looped transformer가 없었다** — 아래 §3의 발견 참조. 즉 지금까지
TRM은 자기 genus의 표준 구성원과 한 번도 대조된 적이 없다.

- **H-A (tying-buys-depth)**: D를 맞춘 상태에서 tied(looped)가 untied(multi-layer)와 비슷하거나
  더 낫다면, 성능은 깊이에서 오고 tying은 (파라미터를 6배 아끼면서) 그 깊이를 사는 수단이다.
- **H-B (depth-is-what-matters)**: 파라미터를 맞춘 shallow(deep2)보다 looped(loop2x6)가 낫다면
  looping이 실제 계산 깊이를 산 것이고, 차이가 없다면 looping은 파라미터 재사용 그 이상이 아니다.
- **H-C (TRM-specific residual)**: 파라미터·D·시퀀스 길이를 **셋 다** TRM과 맞춘 `loop2x21_pel16` 이 fig1의
  `fig1_tf_z_iter` 와 차이가 없다면, TRM의 z-carry·1-step-grad는 σ^k 에서 무효이고 TRM ≈ looped
  transformer 다. 차이가 있다면 그 잔차가 TRM 고유 기여분이며 tier C가 그것을 한 축씩 분해한다.

> ⚠️ **파라미터 수가 감추는 축 (2026-07-22 감사에서 적발)**: `puzzle_emb` 파라미터 수는
> `num_puzzle_identifiers × puzzle_emb_ndim` 이라 **`puzzle_emb_len` 과 무관**하다. 따라서
> "파라미터 수 일치" 검사는 TRM(27 위치)과 lt_ 그리드(12 위치)의 **시퀀스 기하 차이를 탐지하지
> 못한다**. 이것이 z-carry·1-step-grad·injection 에 이은 **네 번째 차이**다. 대응: lt_ 그리드
> 전체는 12 위치로 통일해 looped↔multi-layer 대조를 깨끗하게 유지하고, **`loop2x21_pel16`
> 한 cell만** 27 위치로 두어 이 축을 *혼입시키지 않고 분리*한다. `scripts/verify_looped_grid.py`
> 가 이 기하를 직접 assert 한다 (파라미터 검사로는 못 잡으므로).

> **modal status**: 위 셋은 전부 **미검증 예측**이다. 이 문서의 어떤 문장도 "establishes"급이 아니다.
> H-023 관련 함의는 `h023-falsified-as-framed` 의 현재 판정(optimization-not-capacity =
> supported-but-not-established)을 넘어서지 않는다.

## 1b. 관련 근거 자료 (web · proof · theory canon)

| 종류 | 식별자·링크 | 뒷받침/도전하는 주장 | 상태 |
|---|---|---|---|
| web | Yang, Lee, Nowak, Papailiopoulos, "Looped Transformers are Better at Learning Learning Algorithms", ICLR 2024, arXiv:2311.12424 | **훈련된** looped transformer가 <10% 파라미터로 standard transformer 성능에 도달 — H-A/H-B의 positive precedent | verified (기존 scout note에서 직접 fetch 확인) |
| web | Fan, Du, Ramchandran, Lee, "Looped Transformers for Length Generalization", ICLR 2025, arXiv:2409.15647 | adaptive step-count looped TF가 알고리즘 과제에서 length-generalize | verified — **단, 그쪽 축은 sequence length, 우리 축은 합성 깊이 k. 직접 예측 근거로 쓰면 category error** (scout note §4 경고) |
| web | Giannou et al., "Looped Transformers as Programmable Computers", ICML 2023, arXiv:2301.13196 | looped TF의 capacity 존재증명(hand-constructed) | verified — capacity-only, 학습 가능성 주장 아님 |
| web | Merrill & Sabharwal, "Exact Expressive Power of Transformers with Padding", arXiv:2505.18948 | O(log^d n) looping ⟺ FO-uniform TC^d | verified (abstract-level). H-022 인용위생 이슈는 `omega-logk-lb-unproven` 참조 — **Ω 하한 미증명, 정리 인용 금지** |
| canon | `lab/literature/2026-07-17_bg_weight-tied-expressivity.md` | TRM = looped transformer genus 라는 판정, 및 "H-023 collapse에 대한 문헌 선례 zero" | 기존 scout note (novelty gate 충족) |
| canon | memory `iter-recurrent-compute-confound` | iter on/off가 compute와 혼입된 전례 → 이 실험이 D를 cell별로 명시하는 이유 | verified rule |

**Novelty gate (CLAUDE.md S2)**: 기존 scout note가 이 문헌 축을 이미 전수 조사했고, "weight-tied
no-z 변형의 train≈1/test≈0 collapse에 대한 선례 없음"을 명시한다. 신규 광역 scout 불필요 —
`scout-triage-before-realarm` 규율에 따라 기존 판정을 선참조하여 재알람하지 않는다.

## 2. 예상 결과 시나리오와 대응

| 시나리오 | 관측 패턴 | 해석 | 대응 |
|---|---|---|---|
| **S1** | loop2x6 ≈ deep12 ≫ deep2 | tying이 깊이를 6배 싸게 산다 (H-A·H-B 동시 지지) | tier B factorization으로 (ℓ,T) 최적점 보고. TRM의 우위는 z/grad 축으로만 주장 가능 |
| **S2** | deep12 ≫ loop2x6 > deep2 | untied 용량이 실제로 필요 — tying은 손해지만 shallow보단 낫다 | "TRM은 깊이를 사되 대가를 치른다"; loop2x21이 그 격차를 메우는지 확인 |
| **S3** | loop2x6 ≈ deep2 (둘 다 낮음) | looping이 σ^k 에서 유효 깊이를 사지 못함 | **H-022 optimization-side 강한 신호**. 용량은 있으나(문헌) 학습이 도달 못함 → C0 teacher-forced 경로로 라우팅 |
| **S4** | loop2x21 ≈ fig1_tf_z_iter | z-carry·1-step-grad가 σ^k 에서 무효 | TRM 서사를 "looped transformer의 재발견"으로 정직하게 축소. **보고서 주장 강도 하향 필요** |
| **S5** | fig1_tf_z_iter ≫ loop2x21 | TRM 고유 기여 실재 | tier C(grad1/noinj)가 어느 축인지 지목. z 축은 trm_singlez cohort와 대조 |
| **S6** | loop2x6_grad1 ≈ loop2x6 | 1-step gradient 근사가 무해 | TRM의 근사는 순수 계산 절약. H-023 논의에서 grad 축 제외 가능 |
| **S7** | loop2x6_grad1 ≪ loop2x6 | 1-step gradient가 성능 손실 요인 | **H-023 재해석**: no-z collapse가 z 부재가 아니라 grad 근사와 z 부재의 상호작용일 수 있음 → H-023 문서 갱신 |
| **S8** | k에 따라 비단조 (fig1 noz_iter 패턴 재현) | 붕괴 축이 architecture-agnostic | `sigmak-cycle-type-collapse-axis` (gcd(k, ord σ)) 와 대조 — k=9/k=11 판별로 라우팅 |

## 3. 코드/설정 수정 내역

> **핵심 발견 (2026-07-22, 코드 확인)**: `transformers_baseline`(`Model_ACTV2`)의 `H_cycles` 는
> **dead code** 다. `Model_ACTV2_Inner.forward` 는 `H_level` 을 **단 한 번** 통과시키고,
> config 필드에도 `# kept for compatibility` 주석이 달려 있다. 결과:
> 1. 기존 ablation의 `abl_tfb_lay{L}_cyc6_*` 18 cell 은 `cyc1` 과 **완전 동일한 run** 이었다.
>    (10.0.12.93 에서 skip 처리된 것은 **사후적으로 옳은 판단**이었고, 이제 그 근거가 코드로 확정됐다.)
> 2. 즉 **multi-layer baseline은 이미 존재**했고(= untied depth stack), **없던 것은 looped 쪽**이었다.
>    이 실험은 그래서 multi-layer를 새로 만들지 않고 looped 를 신규 구현한다.
> 이 성질은 `tests/test_looped_transformer.py::test_transformers_baseline_ignores_cycles` 로
> **회귀 고정**했다 — 이 테스트가 깨지면 93 cohort의 해석이 바뀐 것이므로 ACTIVE_COHORT.md 재검토 신호다.

| 파일:위치 | 수정 | 이유 | 대안과 pros/cons |
|---|---|---|---|
| `models/recursive_reasoning/looped_transformer.py` (신규) | `Model_ACTV3` — tied 블록을 T회 적용, full BPTT, `loop_grad_cycles`/`input_injection_every_cycle` 노브 | canonical looped transformer가 repo에 없었음 | **(택)** 신규 파일: tfb가 byte-identical 유지 → 93에서 도는 cohort 의미 불변. **(기각)** tfb에 flag 추가: 코드 중복은 줄지만 살아있는 cohort의 semantics를 소급 변경할 위험 |
| `config/arch/looped_transformer.yaml` (신규) | `halt_max_steps: 1` 고정, `num_heads: 8` | halt>1 이면 ACT 외부 루프가 **두 번째 혼입 recurrence 축**이 됨. heads=8은 trm과 head_dim=64 일치 | tfb 기본 12 heads는 512//12=42 로 head_dim 불일치 → 축 혼입. 대신 deep arm도 `num_heads=8`로 새로 돌림 (기존 abl_tfb 재사용 포기 = GPU 비용, 대가로 정합성) |
| `scripts/sigma_enqueue.sh` | `STAGES=looped` 스테이지 + `LOOPED_TIER_{A,B,C}` + `emit_looped_tier` | 기존 grid 정의 방식과 동일 패턴 | **기본 STAGES에 넣지 않음** — 넣으면 fig1/ablation 재생성 시 58 job이 조용히 섞임 (기존 `STAGES` 게이트 도입 취지와 동일) |
| `tests/test_looped_transformer.py` (신규) | 6개 회귀 게이트 | T=1 축퇴·weight tying·grad path·tfb dead-cycles | 없으면 "looping이 실제로 도는지" 를 훈련 결과로만 추론하게 됨 |
| `scripts/verify_looped_grid.py` (신규) | 그리드 전 cell을 Hydra 경로로 실제 instantiate → D·파라미터 표 + matched-pair assert | 그리드 주석의 D/파라미터 주장이 코드와 어긋나면 즉시 실패 | 손계산 표: 싸지만 drift 감지 못함 (실제로 `loop2x21` 추가 시 이 스크립트가 커버리지 누락을 잡아냄) |

**`transformers_baseline` 은 한 글자도 수정하지 않았다.** (93 cohort 보호)

## 4. Run 매니페스트

- **run name 스키마**: `lt_<tag>_k<k>_s<seed>`
  - `deep{D}` = untied multi-layer (arch=`transformers_baseline`, H_layers=D)
  - `loop{ℓ}x{T}` = tied looped (arch=`looped_transformer`, H_layers=ℓ, H_cycles=T)
  - 접미사 `_grad1`(1-step gradient), `_noinj`(첫 cycle에만 injection)
- **그리드 (총 58 runs)**: tier A 4 cell × k{3,4,5,6,7,8,10} = 28, tier B 7 cell × k{4,6,8} = 21,
  tier C 3 cell × k{4,6,8} = 9.
- **cohort → 플래그 매핑** (파라미터·D는 `scripts/verify_looped_grid.py` 실제 instantiate 결과):

  | cohort | arch | ℓ (H_layers) | T (H_cycles) | D=ℓ·T | params | seq | tier |
  |---|---|---|---|---|---|---|---|
  | `deep2` | transformers_baseline | 2 | 1 | 2 | 6,828,034 | 12 | A |
  | `deep12` | transformers_baseline | 12 | 1 | 12 | 40,906,754 | 12 | A |
  | `loop2x6` | looped_transformer | 2 | 6 | **12** | **6,828,034** | 12 | A |
  | `loop2x21` | looped_transformer | 2 | 21 | **42** | **6,828,034** | 12 | A |
  | `loop1x12` | looped_transformer | 1 | 12 | 12 | 3,420,162 | 12 | B |
  | `loop3x4` | looped_transformer | 3 | 4 | 12 | 10,235,906 | 12 | B |
  | `loop6x2` | looped_transformer | 6 | 2 | 12 | 20,459,522 | 12 | B |
  | `loop2x3` | looped_transformer | 2 | 3 | 6 | 6,828,034 | 12 | B |
  | `loop2x12` | looped_transformer | 2 | 12 | 24 | 6,828,034 | 12 | B |
  | `deep4` | transformers_baseline | 4 | 1 | 4 | 13,643,778 | 12 | B |
  | `deep6` | transformers_baseline | 6 | 1 | 6 | 20,459,522 | 12 | B |
  | `loop2x6_grad1` | looped_transformer | 2 | 6 | 12 | 6,828,034 | 12 | C (`loop_grad_cycles=1`) |
  | `loop2x6_noinj` | looped_transformer | 2 | 6 | 12 | 6,828,034 | 12 | C (`input_injection_every_cycle=False`) |
  | `loop2x21_pel16` | looped_transformer | 2 | 21 | **42** | **6,828,034** | **27** | C (`puzzle_emb_len=16`) |
  | *(anchor, 재run 안 함)* `fig1_tf_z_iter` | trm | 2 | 3 (×L6) | **42** | **6,828,034** | **27** | fig1 |

  **matched pair 판독법** (이 표의 전부):
  - `loop2x6` vs `deep12` — D 동일(12), params 6배 차 → **tying의 값**
  - `loop2x6` vs `deep2` — params 동일, D 6배 차 → **looping이 깊이를 사는가**
  - `loop2x21_pel16` vs `fig1_tf_z_iter` — **params·D·시퀀스 길이 셋 다 동일** → 남는 차이는
    z-carry·1-step-grad·injection 뿐. **H-C 판정은 `loop2x21`(12위치)이 아니라 이 cell로 해야 한다.**
  - `loop2x21` vs `loop2x21_pel16` — 시퀀스 기하 축 단독 (그 축이 유의한지 자체 검정)
  - `loop{1x12,2x6,3x4,6x2}` — D 고정(12), tying 정도만 변화 → **factorization 곡선**

- **공통 hparam**: `epochs=100000 eval_interval=2000 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0
  puzzle_emb_weight_decay=1.0 +log_z_dynamics=True +z_snapshot=False checkpoint_every_eval=False
  ema=True`, `seed=1`, `hidden_size=512`, `num_heads=8`, `halt_max_steps=1`, `evaluators=[]`
  — fig1 프로토콜과 동일 (§5 metric provenance가 여기 의존).
- **데이터 경로 / wandb project**: `data/sigma_k_10/<k>` (EXP-007 order-filter 수정 후 clean 데이터,
  `sigmak-data-filter-off-disk` 참조) / `Sigma_k_new` (fig1·ablation과 동일 project, `lt_` 접두사로 분리).
- **GPU · queue**: 미정 (§7-1). 58 job.
- [x] `STAGES=looped scripts/sigma_enqueue.sh --dry-run` 로 그리드 검증함 → 58 job, 이름·순서 확인.
- [x] 기본 `STAGES` (fig1 ablation) dry-run 은 **126 job, `lt_` 0건** — 신규 스테이지가 기존 그리드를 오염시키지 않음.
- [x] 전 cell Hydra instantiate + matched-pair assert 통과 (`scripts/verify_looped_grid.py`).
- [x] `rtk uv run pytest tests/test_looped_transformer.py` → **6 passed**.
- [x] **end-to-end 스모크 (2026-07-22, aigpu0918 GPU 0, `WANDB_MODE=offline`)** — 유닛테스트만으로는
  `ACTLossHead` + ACT 루프 + `torch.compile` + bf16 경로를 증명하지 못하므로 실제 `pretrain.py` 로 2 cell 실행:
  - `loop2x6` (D=12): 훈련·eval·EMA 전환·checkpoint 저장 정상, **`probe/test_exact` 로깅 확인**
    (figure primary metric이 신규 arch에서 실제로 나온다는 증거).
  - `loop2x21_pel16` (D=42, seq 27 — **그리드 최대 부하**): exit 0, OOM 없음, checkpoint 저장 정상.
    **피크 메모리 44.9 GiB / 49 GiB (RTX 6000 Ada)** — `global_batch_size=2048` 에서 **들어가지만
    여유가 ~4 GiB뿐**이다. 함의: (a) 이 cell은 GPU를 단독 점유해야 하고 (co-tenant 있으면 OOM),
    (b) 93의 Blackwell 96 GiB에서는 여유롭다, (c) §7-4의 batch 축소 fallback은 48 GiB 카드에서
    **다른 프로세스가 없다면 불필요**하다.
  - 스모크 run은 epochs 30–60의 폐기용이며 `SMOKE_looped` (offline) 로 격리 — `Sigma_k_new` 미오염.

### 실행 상태 스냅샷 (2026-07-22)
| cohort | done (post-hoc) | running | queued (pre-registered) |
|---|---|---|---|
| 전 14 cohort | 0 | 0 | **0 (job 파일 미생성 — 설계만 등록)** |

## 5. Figure 계획 (wandb log 매핑)

| Figure | 내용 | wandb key / 소스 | 필터·집계 |
|---|---|---|---|
| **F1** matched-pair grid | k(가로) × {deep2, deep12, loop2x6, loop2x21, TRM anchor}(세로) 정확도 히트맵 | `probe/test_exact` (fig1 primary metric, `log_z_dynamics=True` 로 게이팅) | project=`Sigma_k_new`, run명 `lt_*` + `fig1_tf_z_iter_*`, config 필드 `k` |
| **F2** tying-vs-depth 산점 | x=params(log), y=test_exact, 점=cohort, 색=D | 동일 | k별 패널. **F1/F2의 핵심 주장이 여기서 읽힘** |
| **F3** factorization 곡선 | D=12 고정, x=ℓ ∈{1,2,3,6}, y=test_exact | 동일 | tier B, k∈{4,6,8} |
| **F4** depth ladder | x=D ∈{6,12,24,42}(ℓ=2 고정), y=test_exact | 동일 | loop2x3/2x6/2x12/2x21 |
| **F5** grad·injection ablation | loop2x6 vs _grad1 vs _noinj 막대 | 동일 | tier C, k∈{4,6,8} |
| **F6** TRM 잔차 | loop2x21 vs loop2x21_pel16 vs fig1_tf_z_iter | 동일 | k∈{4,6,8}. **H-C 판정 figure** |

**metric provenance (필수)**: `test-exact-metric-provenance` 규율에 따라 **`probe/test_exact`
단일 계통만** 쓴다 (EMA 가중치, 512 샘플). `all/exact`(online-500) 와 섞지 말 것. 모든 figure
캡션에 `probe/test_exact (EMA, n=512)` 를 명시한다. figure는 `reports/figures/` 정본 +
`lab/figures/<날짜>_looped-baseline/` 사본 양쪽에 저장 (`figures-copy-to-lab`).

## 6. 결과 기입란 (실험 후)

- [ ] post-hoc cell: **없음** (전 cell 사전등록)
- [ ] 시나리오 판정: S?
- [ ] 실제 값 vs 예측:
- [ ] wandb run / checkpoint step:
- [ ] 후속 결정:

## 7. 미결 사항 (⏸ PI 결정 대기)

이 항목들은 **준비 범위 밖**이라 의도적으로 미결로 남긴다. 어느 쪽이든 §1–§5는 유효하다.

1. **실행 호스트**: 93은 현재 ablation 38 job 대기(4 GPU 점유, 완료까지 상당 시간). 로컬
   aigpu0918은 fig1 cohort 진행 중. 58 job을 (a) 93 큐에 append, (b) fig1 완료 후 로컬,
   (c) 제3 호스트 중 어디에 넣을지 미정. 93 이라면 §8의 코드 배포 경로가 선행돼야 한다.
2. **seed 수**: 현재 s=1 단일 (fig1과 동일 관행). tier A만이라도 s∈{1,2,3} 으로 밴드를 볼지 —
   그러면 28→84 job. **권고: 우선 s=1로 신호를 보고, S1/S4 처럼 판정이 미세차에 걸릴 때만 증seed.**
3. **k=9, k=11 포함 여부**: `sigmak-cycle-type-collapse-axis` 의 판별 k. 이 baseline에도 넣으면
   붕괴 축이 architecture-agnostic 인지 직접 볼 수 있다 (tier A +8 job). 별도 결정 사항.
4. **loop2x21 메모리**: full BPTT로 ~42 layer 활성값 보존. **스모크 결과 48 GiB 카드에서
   44.9 GiB로 통과** (위 §4 체크리스트) — 따라서 batch 축소는 기본적으로 **불필요**하다. 다만
   여유가 ~4 GiB뿐이므로 **해당 GPU 단독 점유가 전제**다. 그럼에도 OOM이 나면 그 cell만
   `global_batch_size=1024` 로 낮추고 프로토콜 이탈로 기록 (다른 cell과 batch 불일치 = 혼입 주의).

## 8. 93 서버 배포 경로 (실행 결정 시 선행 조건)

93의 코드는 `/mnt/ayp/trm/bootstrap/code-20260721-worktree.tar.gz` 스냅샷 + 미러에서 덮어쓴
`scripts/{sigma_enqueue,queue_run}.sh` 다. 신규 arch는 **tarball에도 미러에도 없다.** 따라서 93에서
돌리려면 최소한 아래가 NAS 미러로 전달돼야 한다:

- `models/recursive_reasoning/looped_transformer.py`
- `config/arch/looped_transformer.yaml`
- `scripts/sigma_enqueue.sh` (looped 스테이지 포함본)
- (권장) `tests/test_looped_transformer.py`, `scripts/verify_looped_grid.py` — 93에서도 게이트 실행

`mnt-bootstrap-refresh-20260721` 규율: **bootstrap payload는 날짜별 불변**이므로 기존 tarball을
덮지 말고 신규 날짜 payload를 만들거나 docs-mirror 쪽 파일만 in-place 갱신할 것.
그리고 93에서 `STAGES=looped` 로 enqueue 하기 전에 **그 호스트에서 pytest·verify 스크립트를 먼저
통과**시킬 것 — Blackwell(sm_120) 환경은 이 파이프라인에서 여전히 신규 코드 미검증이다.

## §7 Postmortem

(실험 종료 후 작성)
