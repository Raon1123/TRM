---
title: "Handoff Index (server-local, self-contained)"
updated: 2026-07-30T03:00
---

# Handoff Index — /home/ayp/project/trm

> `/handoff` / `/handoff-resume` 전용 INDEX. Track별 newest-first. Resume:
> `/handoff-resume <code|date|path|keyword>`.
> 이 서버에는 레거시 `_handoff-index.md`(수기 항목, code VGR/AFA)가 실재하지 않는다 —
> 2026-07-22 최초 실행 시 `lab/handoffs/` 자체가 없어 이 INDEX부터 생성했다.
> three-letter code 네임스페이스는 이 INDEX가 단독 관리한다.

### harness-sync / laptop↔server relay

- `[HSR]` [2026-07-30 02:45] **needs-decisions** — 2-hop 릴레이(랩탑→gdrive→서버) 라운드 3·4: harness 에이전트 정의의 죽은 vault 참조 **33 치환**(전부 `count(old)==1` 단언, `tomllib` 8/8). 미해결 4건 종결 — `Tags System`=`80_Wiki/_tag-registry.md` · `4 pillars`=Research Relevance Framework 4영역(`challenge.md.off:27`이 동격 진술, 정황 아닌 등식) · README 고아 확정 · attention-literature 완료취급. **서버가 07-28에 만든 오배선 1건 되돌림**(claim-registry `C1–C6`→실제는 `D/P/H` 코드계 — 경로가 열려서 precondition이 통과하는, 부재보다 잡기 어려운 종류). 랩탑 미검출 2건 보고(`80_Wiki/researchers/` 부재 / `10_Analysis/landscape/` 부재 = `SCOPED_WRITE_ROOTS` 유일 허용구가 허공). 코드 작업 완료, 남은 건 사용자 결정 5건 → `2026-07-30_harness-sync-relay-round4-0245.md` · next: `rclone lsl gdrive:harness-inbound/2026-07-28/ 로 REPLY4 확인 → 없으면 Decision 1·2를 사용자에게`

### fig1-multiseed / EXP-013

- `[FAM]` [2026-07-30 02:45] **ready-for-resume** — 사용자 요청("grid 시드 다르게")으로 fig1+ablation 다중시드(seed 2,3) 재현 실험 EXP-013 사전등록 + `scripts/sigma_enqueue.sh` 수정(`SEEDS`/`ABLATION_SEEDS` 분리 — ablation seed=1은 별도 호스트 `10.0.12.93` 소유라 공유하면 70개 재훈련 위험, dry-run으로 실측 확인 후 분리). dry-run 252 신규 job(fig1 112+ablation 140) 검증, launch는 사용자 보류. 부수: `.gitignore` `lab/*`가 신규 `lab/experiments/*.md`를 추적 누락시키는 문제 발견(FGP 트랙과 동일 문제, `git add -f` 필요); FGP 트랙의 오래된 handoff 스냅샷을 근거로 `mv`를 실행했다가 daily log의 최신 정정을 뒤늦게 발견해 즉시 원복(net 영향 없음, memory 갱신) → `2026-07-30_fig1-ablation-multiseed-exp013-0245.md` · next: `git add -f로 EXP-013 문서 커밋 여부 결정 → queue_run.sh launch 판단`

### figure-pipeline / fig1 provenance

- `[FGP]` [2026-07-30 02:42] **needs-decisions** — `0217_fig1_mlp_noz_iter_k8_s1`이 `done/`→`jobs/`로 되돌아간 뒤 **02:44:39 gpu2로 claim되어 재훈련 시작**, 그러나 **step 478/244,140에서 자연 정지**(02:48:56) → dir 미완주 → M23이 무시 → **fig1 56/56 유지, 후퇴 없음**. 개입 권고는 **철회**: `scripts/queue/stop`(07-29 12:47:35)이 `do not launch or recover claims pending audit` hold이고, 불변 대장 `2026-07-29_clean-r1-auto-claim/incident_manifest.json`이 0217을 `legacy_excluded`로 명시 제외한다. **진짜 문제는 hold가 걸린 상태에서 claim이 통과한 것**(`queue_run.sh:119-122`는 hard stop으로 취급) → 사건 소유자 보고 대상. coverage 게이트는 `pending 29`+`failed 2`로 재폐쇄 → `2026-07-30_fig1-0217-requeue-regression-risk.md` · next: `CLEAN-R1 사건 소유자에게 hold 실효성 보고 → 그 다음 task #22`
- `[FGP]` [2026-07-29 12:48] **superseded** (↑ 위 문서로 이어짐) — fig1 acquisition **48/56 → 56/56** 복구(스냅샷 `ds-sigma-k-new-20260729-7c3f44b48859547c`, QA 16/16, 651 passed); 지시받은 `config.yaml` 판별은 **반증**(ablation 4/8만 판별)되어 사전등록 완주 기준 **M23**(`terminal_step=244100 ∧ eval_rows=50`, 완주 후보 유일할 때만 선택)으로 대체. 새 결과 2건: C4에 mlp 대응물 부재라는 실증 논거, EXP-007 tf/mlp 상충이 k=8·10에서도 재현되는 **일반 패턴**(⏳k=11 전 문서화 필요). PREVIEW 잔존 사유는 `source_commit_dirty` 하나 — coverage 게이트는 큐 drain으로 열림. NAS 3호스트 중 **1/3만 이행**(93 완료 / ml7 코호트 미기동+heartbeat 찢어진 쓰기 / imil0 NVML 불일치) → `2026-07-29_fig1-56cell-recovery-provenance-1248.md` · next: `Read lab/audits/2026-07-29_fig1-56cell-recovery.md §7 → 보고서 §6.1·§7 개정을 k=11 arm 전에 착수`
  - code 주의: `FGE`(explainer 산문 트랙)와 별개 트랙이다 — predecessor 아님.

### perf-optimization / PERF-001

- `[PFO]` [2026-07-29 07:25] **ready-for-resume** — PERF-001 계측 하네스+flame chart 도구 PR #1 merge(`98e6c64`); py-spy가 `unique()`를 wall time 37.6%로 지목했으나 후보 P1-A는 bit-exact 동등하되 이득 0 → **기각**(스텝이 GPU-bound라 sync 제거 시 대기가 다음 sync로 이동). 다음은 실제 GPU 작업량(metrics `torch.stack`, CUDA 13.4%) → `2026-07-29_perf-flamechart-p1a-refuted.md` · next: `Read lab/reports/2026-07-29_perf002-p1a-refuted.md §5 → metrics 로깅 주기 후보 사전등록 후 A/B 측정`

### fig1-architecture / explainer

- `[FGE]` [2026-07-22 11:10] **needs-decisions** — fig1 3축(z / token-mixer / iter) explainer 작성(`lab/reports/2026-07-22_fig1-experiment-explainer.md`); full-MLP 질문 답변 후 같은 날 07:30 선행 감사를 뒤늦게 발견해 §9·§2.2 보정(mlp_t=이미 완전 MLP / 빈 4번째 칸=스택형 MLP-Mixer / RoPE·길이 혼입) → `2026-07-22_fig1-axes-explainer-1110.md` · next: `Read lab/audits/2026-07-22_arch-axis-audit.md §4-§6 → tfb cyc no-op(비용 0, 원격 큐 40% 절감) 먼저 처리`

### mcp-server / trm-mcp

- `[MCS]` [2026-07-30 02:53] **needs-decisions** — 소비자 매뉴얼(codex-work 초안→codex-verify FAIL 3 WRONG→수정, `trm-mcp/docs/MANUAL.md`+사본) + **첫 라이브 배포**(`10.20.22.130:8765`, whitelist=10.20.23.74+서버4+자기자신, env=`.env.production` 단일 위치). 라이브 첫 검증이 4중 감사가 못 본 **SDK transport-security 421**(FastMCP loopback 자동보호) 적발→명시적 allowlist 배선+회귀 3종, **353 passed**, codex 독립 감사 7/7 SAFE(revert 실증 포함). token rotation(07-30, 사용자 요청으로 세션 내 전달). 미결: D-s6 해석 확인·방화벽 sudo·Q3/Q5/Q6·저장소 무커밋 → `2026-07-30_mcp-live-deploy-manual-0253.md` · next: `소비자 머신(예: 10.20.23.74)에서 MANUAL §5d claude mcp add로 원격 연결 실증 → 방화벽 ufw 적용 요청`
- `[MCS]` [2026-07-22 10:57] **superseded** (↑ 위 문서로 이어짐) — 내부망 Research MCP 서버 trm-mcp 설계·구현·활성화 완료(별도 프로젝트 ~/project/trm-mcp, v1 wave 1~7 + T10b whitelist 게이트, 350 passed, 다중 적대 감사 통과); 실 라이브 기동은 배포 행위로 남김, 열린 결정 Q3/Q5/Q6·research_mcp 제거(D1) 대기 → `2026-07-22_trm-mcp-build-complete-1057.md` · next: `Read ~/project/trm-mcp/docs/DESIGN.md + docs/DECISIONS.json → 활성화 또는 열린 결정 진행`

### roadmap-automation / orchestrator

- `[RMA]` [2026-07-22 04:20] **needs-decisions** — 메인 오케스트레이터 세션 종료·tick 모니터링 별도 세션 이관; A2(EXP-007 k9) verifier 게이트까지 완주해 ⏸PI 대기, tick 원장 내구화, ROADMAP §0 A3–A5 고아 트리거 적발 → `2026-07-22_orchestrator-monitor-handover.md` · next: `ACTIVE_COHORT + ROADMAP §0 정독 후 §0 트리거 표 재정의 판단`
