---
id: EXP-013
slug: fig1-ablation-multiseed-replication
hypotheses: [H-022, H-023, H-025]
parent_exp: EXP-010            # fig1 seed=1 그리드의 직접 확장(같은 SEEDS 배열이 ablation도 구동 — EXP-011도 자매로 확장됨, 아래 §0 참조)
registration_mode: pre-registered   # SEEDS=(2 3) 셀은 이 문서 작성 시점에 미실행(enqueue() idempotent skip으로 seed=1 재실행 없음, §3 확인) — EXP-010/011과 달리 진짜 사전등록
wandb_runs: []          # run-name 스키마는 §4 참조
status: planned
date_designed: 2026-07-29
date_closed: ~
---

# fig1 + ablation 그리드 다중 시드 확장 (seed 2, 3) — 2026-07-29

> 실험 보고서 규율: 모든 실험은 착수 전 본 템플릿으로 보고서를 작성한다.
> figure는 wandb log만으로 그릴 수 있어야 한다 — 필요한 지표가 로깅되는지 §5에서 보장할 것.

| 작성일 | 상태 | 담당 파이프라인 | 연계 H-번호 | 사전등록 |
|---|---|---|---|---|
| 2026-07-29 | planned (enqueue 대기, 사용자 승인 후 launch) | 설계 Opus(Claude 메인 세션) · 실행 사용자 tmux 큐 · 커밋/launch는 사용자 | H-022, H-023, H-025 | **pre-registered** |

## 0. 기호 정의

`lab/reports/2026-07-22_fig1-experiment-explainer.md` §0 및 `EXP-010`/`EXP-011` §0의 기호를
그대로 재사용한다(재정의 금지, 링크만). 본 문서 고유 기호만 추가:

| 기호 | 정의 |
|---|---|
| `s_m` | model seed (`seed=` hydra 필드). data pool·config는 seed에 따라 변하지 않음(CLEAN-R1의 `s_d` data-seed 축과 무관 — 이 문서는 model seed만 다룬다) |
| G4/H3/A0 | 7개 k에 대한 `classify()` 결과 카운트 표기, 예: G가 4개·H가 3개·A가 0개 |
| `노이즈 vs 이동` | 관측된 collapse-k 집합의 변화가 seed 분산 내인지, 데이터/메커니즘의 실제 차이인지 |

## 1. 가설

`EXP-010`/`EXP-011`은 fig1(56 cell)·ablation(70 cell 계획)을 **seed=1 단일 표본**으로만
관측했다. 두 문서의 §7a가 공통으로 지목한 미해결 지점:

1. `EXP-010` §7a 후보 2: "clean-data ord-filter 수정이 `noz_iter`의 비단조 collapse-k
   집합(legacy k={5,7} → clean k={5,6,7})을 이동시켰다"는 **seed=1 노이즈와 구분되지
   않는다**.
2. `EXP-010` §7c lesson 1/2: `z_iter` k=3 dagger(peak G→final A), `noz_noiter` k=10 단독
   A 이상 — 둘 다 seed=1 단일 관측이라 memory 승격 기준(fail→investigate→verify→distill)
   미충족.
3. `lab/audits/2026-07-29_fig1-56cell-recovery.md` §4.2 (⭐ 신규 논거): `tf_noz_iter`의
   k∈{8,10} 비단조 회복에 `mlp_noz_iter` 대응물이 없다(mlp는 k=5..8,10 전부 H) — C4(gcd/
   cycle-type 판독 거부)를 강화하는 근거이나, 이 역시 **seed=1 단일 관측**이다.

본 실험은 위 세 관측이 **seed 분산 내의 우연**인지 **재현 가능한 신호**인지 model-seed
2개(2, 3)를 추가해 판별한다. 새로운 가설을 도입하지 않는다 — H-022/H-023/H-025의 기존
관측을 강건화(robustify)하는 보조 grid다(EXP-011이 EXP-010에 대해 그랬던 것과 동일한
위상).

## 1b. 관련 근거 자료 (web · proof · theory canon)

| 종류 (web/proof/canon) | 식별자·링크 | 뒷받침/도전하는 주장 | 상태 |
|---|---|---|---|
| canon | `lab/experiments/EXP-010_fig1-clean-rerun-sigma-k-new.md` §7a | seed=1 단일 표본이라 collapse-k 이동의 원인(데이터 vs seed 노이즈)을 판별 불가 | 검증됨(문서 내) — 본 실험의 직접 동기 |
| canon | `lab/experiments/EXP-011_module-ablation-sigma-k-new.md` | 동일 SEEDS 배열이 구동하는 자매 grid, 같은 seed=1 한계 공유 | 검증됨(문서 내) |
| canon | `lab/audits/2026-07-29_fig1-56cell-recovery.md` §4.2 | tf/mlp `noz_iter` 비대칭이 C4(cycle-type 판독 거부) 강화 논거이나 seed=1 단일 | 검증됨(문서 내) — 재현되면 논거가 강화, 안 되면 약화 |
| canon | [[iter-recurrent-compute-confound]] | iter 축은 H3L6 vs H1L1(~6× recurrent-compute)과 혼입 — 본 실험은 이 confound를 해소하지 않는다(범위 밖) | 검증됨(memory), 미해결 유지 |

## 2. 예상 결과 시나리오와 대응

각 seed=1 단일 관측(O1: noz_iter collapse-k 이동, O2: z_iter k=3 dagger, O3: tf/mlp
noz_iter 비대칭)에 대해 독립적으로 판정한다. seed 2, 3의 값이 seed 1과 **정성적으로
동일한 classify() 결과**(G/H/A)를 내면 "재현", 아니면 "seed-불안정"으로 기록한다.

| 시나리오 | 관측 패턴 | 해석 | 대응 (다음 액션) |
|---|---|---|---|
| R1 | O1/O2/O3 전부 3개 seed에서 동일 classify() | 세 관측 모두 강건 | `sigmak-collapse-k-set-shifts-with-cleandata.md` 등 EXP-010 §7c 보류 lesson을 memory로 승격 검토 |
| R2 | O1만 seed-불안정(k별 G/H가 seed마다 뒤집힘), O2/O3는 강건 | collapse-k 집합의 정확한 위치는 노이즈이나 비단조성 자체·tf/mlp 비대칭은 실재 | EXP-010 §4.1 개정 시 "정확한 k 집합"은 인용하지 말고 "비단조 패턴 존재"만 인용 |
| R3 | O3(tf/mlp 비대칭)가 seed-불안정 | `lab/audits/2026-07-29_fig1-56cell-recovery.md` §4.2의 C4 강화 논거가 약화 — 단일 seed 우연일 수 있음 | C4 강화 주장 철회, 원래 강도(정황적)로 롤백 |
| R4 | 셋 다 seed-불안정 | seed=1 단일 관측 전체가 노이즈 지배 | fig1/ablation 해석을 전면 재검토, 최소 3-seed가 기본값이어야 한다는 규율 제안 |

각 시나리오는 상호 배타적이지 않다(O1/O2/O3 독립 판정) — 위 표는 대표 조합이며, 실제
판정은 §6에서 O1/O2/O3 개별로 기입한다.

## 3. 코드/설정 수정 내역

| 파일:위치 | 수정 | 이유 | 대안과 pros/cons |
|---|---|---|---|
| `scripts/sigma_enqueue.sh` `SEEDS=(1)` → `SEEDS=(1 2 3)` (fig1 grid, §1 loop) | fig1 grid만 이 배열을 그대로 씀. `enqueue()`가 **idempotent**(run_name이 `jobs/`·`processing/`·`done/`·`failed/` 어디든 있으면 skip)이고 **이 호스트의 로컬 큐에 fig1 seed=1 56/56이 이미 `done/`으로 존재**함을 실측 확인(아래) — seed=1 재순회는 전부 skip되고 seed 2/3만 신규 추가된다 | — |
| `scripts/sigma_enqueue.sh` 신규 `ABLATION_SEEDS=(2 3)` 추가, ablation 루프(§2, TRM_ABLATIONS·TFB 두 곳)만 `SEEDS`→`ABLATION_SEEDS`로 교체 | **당초 계획(fig1과 동일하게 SEEDS=(1 2 3) 공유)은 반증됨**: ablation seed=1(70 cell, EXP-011)은 **별도 호스트(`10.0.12.93`)**에서 실행됐고, 이 호스트의 로컬 `scripts/queue/done`·`jobs`에는 `abl_*` 항목이 **0개**(dry-run 실측, 아래 §4.1). idempotency 검사는 호스트-로컬 파일시스템만 보므로 `SEEDS=(1 2 3)`을 ablation에도 그대로 적용했다면 seed=1 70개가 **새 job으로 재emit·재훈련**되어 (a) GPU 낭비, (b) 두 호스트의 wandb 산출물이 같은 run_name으로 충돌해 M23 tie-break가 정확히 하나를 못 골라 `duplicate-retry-superseded`로 quarantine — 오늘 아침 fig1에서 고친 것과 같은 결함을 ablation에 새로 만드는 결과였다 | **대안 A(기각)**: SEEDS를 그대로 공유(당초 AskUserQuestion 답변대로 "1 2 3") — dry-run에서 이 위험이 실측 확인되어 기각. **대안 B(기각)**: 큐 상태를 두 호스트 간 동기화한 뒤 SEEDS 공유 — 이번 작업 범위를 넘는 인프라 변경이라 보류. **채택**: ablation 전용 `ABLATION_SEEDS=(2 3)`으로 분리 — 코드 변경 최소, 위험 원천 차단, fig1은 원안(`SEEDS=(1 2 3)`) 그대로 유지 |
| `scripts/sigma_enqueue.sh` line ~66-68 주석 | "Single seed (=1)..." → "Multi-seed (1,2,3) as of 2026-07-29 (EXP-013)... ABLATION_SEEDS 분리 사유" 로 갱신 | 주석이 stale해지는 것 방지, 향후 재실행자가 왜 두 배열이 분리됐는지 알 수 있게 | — |

## 4. Run 매니페스트

- run name 스키마: fig1 `fig1_{mlp|tf}_{z|noz}_{iter|noiter}_k{k}_s{seed}`; ablation
  `abl_{tag}_k{k}_s{seed}` / `abl_tfb_lay{L}_cyc{C}_k{k}_s{seed}`
- 그리드 (신규분만, seed=1은 fig1은 idempotent skip / ablation은 `ABLATION_SEEDS`로 애초
  제외):
  - fig1: 8 cohort × k∈{3,4,5,6,7,8,10}(7) × `SEEDS`={1,2,3}, seed=1은 이 호스트
    `done/`에 이미 존재해 skip → **112 신규**(seed 2,3)
  - ablation: (`TRM_ABLATIONS` 4 + `TFB_LAYERS`×`TFB_CYCLES` 3×2=6 = 10 axis) × k(7) ×
    `ABLATION_SEEDS`={2,3}(seed=1은 별도 호스트 `10.0.12.93` 소유, 아래 §4.1 참조하여
    아예 배열에서 제외) = **140 신규**
  - 합계 **252 신규 run** (기존 seed=1: fig1 56 완주, 이 호스트 재실행 없음 / ablation
    70 계획·52 완주·2 permanently-crashed는 다른 호스트 소유, 이 실험이 건드리지 않음 —
    `EXP-010`/`EXP-011` §6 참조)
- **cohort → 플래그 매핑** (fig1, `COHORTS` 배열 그대로 인용):

  | cohort | arch | H_cycles | L_cycles | mlp_t |
  |---|---|---|---|---|
  | tf_z_iter | trm | 3 | 6 | False |
  | tf_z_noiter | trm | 1 | 1 | False |
  | tf_noz_iter | trm_singlez | 3 | 6 | False |
  | tf_noz_noiter | trm_singlez | 1 | 1 | False |
  | mlp_z_iter | trm | 3 | 6 | True |
  | mlp_z_noiter | trm | 1 | 1 | True |
  | mlp_noz_iter | trm_singlez | 3 | 6 | True |
  | mlp_noz_noiter | trm_singlez | 1 | 1 | True |

  ablation axis는 baseline(`tf_z_iter`, `arch.L_layers=2 arch.halt_max_steps=1`) 주변
  one-factor-at-a-time — `TRM_ABLATIONS`(halt8/halt16/H6/L3) + `TFB_LAYERS`(1,2,6) ×
  `TFB_CYCLES`(1,6) transformers_baseline 축. EXP-011 §4와 동일, 재정의하지 않음.
- 공통 hparam: `epochs=100000 eval_interval=2000 lr=1e-4 puzzle_emb_lr=1e-4
  weight_decay=1.0 puzzle_emb_weight_decay=1.0 +log_z_dynamics=True +z_snapshot=False
  checkpoint_every_eval=False ema=True`(EXP-010/011과 동일 프로토콜, 변경 없음)
- 데이터 경로 / wandb project: `data/sigma_k_10/<k>` / `Sigma_k_new`(기존 seed=1과 동일
  project — 같은 스냅샷·같은 acquire 파이프라인으로 재추출됨)
- GPU · queue: 기존 `scripts/queue_run.sh` FIFO, `GPUS="4 5 6 7"` 기본값 — 별도 host 분리
  없음(EXP-011의 `10.0.12.93` 분리는 이번엔 적용하지 않음, 단일 큐에 append)
- [x] `scripts/sigma_enqueue.sh --dry-run` 로 그리드 검증함 — **252개 신규 헤더 출력,
  seed=1 168개는 전부 `skip (already in queue lifecycle)`로 확인** (아래 §4.1)

### 4.1 dry-run 검증 스냅샷 (2026-07-29)

**1차 시도 (`SEEDS=(1 2 3)`을 fig1·ablation 공유, 반증됨):** `rtk bash scripts/sigma_enqueue.sh
--dry-run` → 총 322 헤더, `skip (already in queue lifecycle)` 56건(전부 `fig1_*_s1`).
ablation seed=1(`abl_*_s1`) 헤더가 **skip 없이 신규로 322건 중 일부 포함되어 나타남** —
이 호스트의 `scripts/queue/done`(55개) + `scripts/queue/jobs`(1개 stale)를 직접 확인한 결과
`abl_` 접두 항목이 **0개**임을 재확인(`ls scripts/queue/done | grep -c abl_` → 0). 즉
ablation seed=1은 이 호스트 관점에서 "한 번도 실행된 적 없는 셀"로 보여 그대로 두면
70개가 재훈련될 뻔했다 — §3의 `ABLATION_SEEDS` 분리 결정의 직접 증거.

**2차 시도 (`ABLATION_SEEDS=(2 3)` 분리 적용 후, 채택안):**
```
skip 56건 (전부 fig1_*_s1)
신규 fig1_* 112건 (seed 2,3)
신규 abl_*  140건 (seed 2,3)
합계 252 (dry run, nothing written)
```
`bash -n scripts/sigma_enqueue.sh` / `bash -n scripts/queue_run.sh` 모두 통과.
전체 실행 로그: `/tmp/claude-1002/.../scratchpad/dryrun_v2.txt`(세션 로컬, 비영구 —
필요시 재실행으로 재생성 가능, 이 문서에는 요약 수치만 정본으로 남김).

## 5. Figure 계획 (wandb log 매핑)

| Figure | 내용 | wandb key / 소스 | 필터·집계(seed) |
|---|---|---|---|
| fig-01a/b 확장 (min/max band) | 기존 PEAK/FINAL classify() 표에 seed 1/2/3 band 추가 | `probe/test_exact`, `probe/train_exact`(local datastore, `test-exact-probe` protocol) — 기존과 동일 채널, seed 태그만 추가 | `s∈{1,2,3}` 개별 표시 + median/min/max band. **주의**: 렌더 파이프라인(`lab/figure_pipeline`)이 현재 seed=1 단일 표만 지원 — 다중 seed 집계 로직은 이 실험의 범위 밖(별도 task로 추적, §6에 기록) |
| 없음(신규 figure 미계획) | O1/O2/O3 판정은 §6 표만으로 충분, 별도 figure 불필요 | — | — |

## 6. 결과 기입란 (실험 후)

- [ ] 실제 값 vs 예측 (O1/O2/O3 개별):
  - O1 (noz_iter collapse-k 집합 seed 안정성):
  - O2 (z_iter k=3 dagger 재현):
  - O3 (tf/mlp noz_iter 비대칭 재현):
- [ ] wandb run / checkpoint step:
- [ ] 후속 결정:

## §7 Postmortem

> ⏸ 실험 종료 후 작성.

### §7a. 가설 ↔ 결과 불일치

**사전등록 예측:** (§1·§2에서 복사)

**실제 결과:** (§6에서 복사)

**판정:** [ ] 일치 / [ ] 불일치

**불일치 원인 (mechanism):**

### §7b. 버그 로그

| # | 버그 내용 | 근본 원인 | 재발 방지 조치 |
|---|---|---|---|
| 1 | | | |

### §7c. Lessons → memory 후보

- (lesson 1): → 파일 후보 `<slug>.md`
