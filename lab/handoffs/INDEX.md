---
title: "Handoff Index (server-local, self-contained)"
updated: 2026-07-29T07:25
---

# Handoff Index — /home/ayp/project/trm

> `/handoff` / `/handoff-resume` 전용 INDEX. Track별 newest-first. Resume:
> `/handoff-resume <code|date|path|keyword>`.
> 이 서버에는 레거시 `_handoff-index.md`(수기 항목, code VGR/AFA)가 실재하지 않는다 —
> 2026-07-22 최초 실행 시 `lab/handoffs/` 자체가 없어 이 INDEX부터 생성했다.
> three-letter code 네임스페이스는 이 INDEX가 단독 관리한다.

### perf-optimization / PERF-001

- `[PFO]` [2026-07-29 07:25] **ready-for-resume** — PERF-001 계측 하네스+flame chart 도구 PR #1 merge(`98e6c64`); py-spy가 `unique()`를 wall time 37.6%로 지목했으나 후보 P1-A는 bit-exact 동등하되 이득 0 → **기각**(스텝이 GPU-bound라 sync 제거 시 대기가 다음 sync로 이동). 다음은 실제 GPU 작업량(metrics `torch.stack`, CUDA 13.4%) → `2026-07-29_perf-flamechart-p1a-refuted.md` · next: `Read lab/reports/2026-07-29_perf002-p1a-refuted.md §5 → metrics 로깅 주기 후보 사전등록 후 A/B 측정`

### fig1-architecture / explainer

- `[FGE]` [2026-07-22 11:10] **needs-decisions** — fig1 3축(z / token-mixer / iter) explainer 작성(`lab/reports/2026-07-22_fig1-experiment-explainer.md`); full-MLP 질문 답변 후 같은 날 07:30 선행 감사를 뒤늦게 발견해 §9·§2.2 보정(mlp_t=이미 완전 MLP / 빈 4번째 칸=스택형 MLP-Mixer / RoPE·길이 혼입) → `2026-07-22_fig1-axes-explainer-1110.md` · next: `Read lab/audits/2026-07-22_arch-axis-audit.md §4-§6 → tfb cyc no-op(비용 0, 원격 큐 40% 절감) 먼저 처리`

### mcp-server / trm-mcp

- `[MCS]` [2026-07-22 10:57] **needs-decisions** — 내부망 Research MCP 서버 trm-mcp 설계·구현·활성화 완료(별도 프로젝트 ~/project/trm-mcp, v1 wave 1~7 + T10b whitelist 게이트, 350 passed, 다중 적대 감사 통과); 실 라이브 기동은 배포 행위로 남김, 열린 결정 Q3/Q5/Q6·research_mcp 제거(D1) 대기 → `2026-07-22_trm-mcp-build-complete-1057.md` · next: `Read ~/project/trm-mcp/docs/DESIGN.md + docs/DECISIONS.json → 활성화 또는 열린 결정 진행`

### roadmap-automation / orchestrator

- `[RMA]` [2026-07-22 04:20] **needs-decisions** — 메인 오케스트레이터 세션 종료·tick 모니터링 별도 세션 이관; A2(EXP-007 k9) verifier 게이트까지 완주해 ⏸PI 대기, tick 원장 내구화, ROADMAP §0 A3–A5 고아 트리거 적발 → `2026-07-22_orchestrator-monitor-handover.md` · next: `ACTIVE_COHORT + ROADMAP §0 정독 후 §0 트리거 표 재정의 판단`
