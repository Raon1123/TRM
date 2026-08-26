---
title: Goal Graph Orchestration — Sol, Terra, Luna Recursive Workflow
type: operating-manual
status: active
version: 1.1.0
created: 2026-08-02
updated: 2026-08-02
scope: Codex global workflow, Personal Research, code, research, mathematical proof
canonical: true
implementation_status: implemented-and-verified
---

# Goal Graph Orchestration

## 0. 목적과 핵심 원칙

이 문서는 복잡한 목표를 Sol, Terra, Luna 계층으로 분해하고 병렬 실행하는 정본 운영 매뉴얼이다. 모델 이름은 OpenAI의 GPT-5.6 모델군을 가리키지만, 아래의 관리자 역할 배치는 이 문서가 정한 로컬 운영 정책이다.

- **Sol**: 전체 목표를 책임지는 선임 관리자이자 최종 통합자
- **Terra**: 보조정리, 하위 프로젝트, 작업 묶음을 책임지는 중간 관리자
- **Luna**: 하나의 원자 작업을 직접 읽고, 쓰고, 실행하고, 증거를 남기는 실행자

핵심 원칙은 다음과 같다.

1. 하나의 목표에는 하나의 Sol 원장이 있다.
2. 하나의 하위 프로젝트에는 하나의 Terra 원장이 있다.
3. 하나의 원자 작업에는 하나의 불변 업무 지시서와 실행별 결과가 있다.
4. 각 파일에는 한 명의 원장 작성자만 있다.
5. 의존 작업은 `PASS` 판정을 받아야만 충족된다.
6. 실제 경과시간과 에이전트 누적 작업시간을 모두 제한한다.
7. Luna의 자체 검사는 최종 증거가 아니다.
8. 실패한 작업만 최소 범위로 다시 실행한다.
9. Sol은 Luna의 원시 로그를 관리하지 않는다. Terra가 증거를 압축해 보고한다.
10. 상세함은 장문이 아니라 누락 없는 구조화 필드와 직접 증거로 확보한다.

이 문서는 [[research-collaboration-contract]]의 atomic work contract, 단일 작성자, 독립 검증, 증거 기반 판정 규칙을 상속한다. 충돌 시 사용자의 현재 지시, 적용되는 `AGENTS.md`, `CLAUDE.md`, 해당 공통 계약의 순서로 우선한다.

## 1. 언제 사용하는가

다음 중 하나를 만족하면 이 워크플로를 사용한다.

- 세 개 이상의 실질적인 작업 단위와 의존 관계가 있다.
- 여러 단계의 병렬 실행과 후속 통합이 필요하다.
- 한 시간 이상 걸릴 가능성이 있다.
- 한 문맥에 모든 중간 출력과 로그를 담으면 품질 저하가 예상된다.
- 보조정리 그래프, 증명·반증 경쟁, 다중 실험, 다중 문서 작성이 포함된다.
- 사용자가 “재귀 원장”, “Luna 병렬”, “Sol–Terra–Luna”를 요청한다.
- 작업이 중단되더라도 원장에서 재개할 수 있어야 한다.

다음에는 사용하지 않는다.

- 한두 번의 도구 호출로 끝나는 단순 작업
- 의존 관계가 없는 두 개의 작은 작업
- 파일을 바꾸지 않는 짧은 질문
- 이미 더 구체적인 도메인 워크플로가 전체 과정을 소유하는 작업
- 병렬화 비용이 실제 실행 시간보다 큰 작업

## 2. 용어

| 용어 | 정의 |
|---|---|
| 목표 원장 | Sol이 소유하는 전체 목표, 하위 프로젝트, 예산, 최종 판정 기록 |
| 하위 원장 | Terra가 소유하는 자기 하위 그래프, Luna 작업, 단계, 증거, 재실행 기록 |
| 업무 지시서 | Luna 한 명에게 전달되는 변경 불가능한 원자 작업 계약 |
| 실행 결과 | Luna의 한 번의 시도에서 생성된 파일 변경, 명령 결과, 체크리스트 판정 |
| 진행 관찰 | 정해진 간격마다 새 증거, 남은 일, 범위 이탈을 기록하는 짧은 보고 |
| 의존성 그래프 | 선행 작업에서 후행 작업으로 이어지는 순환 없는 방향 그래프 |
| 단계 | 현재 의존성이 모두 충족되어 동시에 실행할 수 있는 작업 묶음 |
| 형식 검증 | 문법, 파일, 스키마, 경로, 명령 성공 여부 같은 기계적 검사 |
| 내용 검증 | 산출물이 주장하는 목적을 실제로 달성했는지 확인하는 의미 검사 |
| 실제 경과시간 | 여러 작업이 병렬 실행되더라도 현실에서 흐른 시간 |
| 에이전트 누적 작업시간 | 동시에 실행된 모든 에이전트의 작업시간을 합한 값 |
| agent thread | 하위 에이전트가 독립된 문맥에서 작업하는 실행 단위 |

## 3. 권한과 재귀 구조

### 3.1 기본 구조

```text
Sol: 전체 목표 원장
└─ Terra: 하위 프로젝트 원장
   ├─ Luna: 원자 작업 지시서 + 실행 결과
   ├─ Luna: 원자 작업 지시서 + 실행 결과
   └─ Terra: 필요한 경우 한 단계 더 세분화
      ├─ Luna: 원자 작업 지시서 + 실행 결과
      └─ Luna: 원자 작업 지시서 + 실행 결과
```

기본적으로 Terra 계층은 최대 두 단계다. 즉, Sol 아래의 Terra가 한 단계의 자식 Terra를 만들 수 있다. 더 깊은 관리자 계층은 다음을 모두 만족할 때만 Sol이 승인한다.

- 현재 하위 프로젝트를 Luna 원자 작업으로 안전하게 자를 수 없다.
- 추가 Terra가 맡을 독립된 성공 조건과 쓰기 범위가 있다.
- 추가 계층의 시간 및 누적 작업시간 비용이 부모 예산 안에 있다.
- 보고 압축으로 Sol의 부담이 실제로 감소한다.

### 3.2 역할별 권한

| 역할 | 소유 | 할 수 있는 일 | 할 수 없는 일 |
|---|---|---|---|
| Sol | 목표 원장, 전역 예산, 최종 통합 | 목표 해석, 하위 그래프 승인, Terra 배정, 전역 게이트 판정 | 모든 Luna 원시 로그 추적, 개별 Luna 파일의 일상적 편집 |
| Terra | 자기 하위 원장과 하위 쓰기 범위 | Luna 생성·관찰·중단·검증·재실행, 내부 예산 재분배, 부모 보고 | 부모 목표·기한·금지 경로 확장, 다른 Terra 원장 수정 |
| Luna | 자기 실행 결과와 지정된 대상 파일 | 정확한 입력 읽기, 지정 파일 쓰기, 명령 실행, 증거 기록 | 원장 수정, 범위 확장, 자기 기한 연장, 공유 색인·기록 수정 |

Terra는 전달자 역할이 아니라 배정된 **중간 프로젝트의 완결 책임자**다. 자기 계약 안의
분해, 단계 실행, 검증 판정, 국소 통합, 실패 범위 재실행은 Terra가 끝낸다. 일반적인
자식 실패를 Sol로 넘기지 않으며, `NARROW → REASSIGN → 최소 범위 재실행`을 먼저 적용한다.
Sol fallback은 부모 목표·권한·기한·Terra 예산·금지 경로·전역 그래프·공유 파일 소유권을
바꾸어야 하거나, 실행 한도 소진 뒤에도 루트 수준 판단이 남을 때만 허용한다. 이때도
`DECISION_REQUIRED` 보고에는 Sol이 정할 최소 결정만 남기고 Terra가 해결 가능한 일은
계속 자기 범위에 유지한다.

### 3.3 부모-자식 불변 조건

자식은 부모가 준 다음 값을 넓힐 수 없다.

- 목표와 성공 조건
- 입력 및 출력 범위
- 쓰기 허용 경로
- 금지 경로
- 실제 경과시간 예산
- 에이전트 누적 작업시간 예산
- 동시 실행 수
- 승인 없이 가능한 외부 동작

부모 계약보다 넓은 권한이 필요하면 `BLOCKED`로 보고하고 부모의 새 지시를 기다린다.

## 4. 실행 생명주기

### G0 — 목표 수신과 안전 경계

Sol은 실행 전에 다음을 고정한다.

- 목표와 산출물
- 가정과 알려지지 않은 항목
- 사용자 승인 범위
- 전체 쓰기 범위와 금지 경로
- 전체 제한시간과 누적 작업시간
- 검증·통합용 예약 시간
- 완료 판정 규칙

필수 입력, 권한, 경로가 없으면 실행하지 않고 `BLOCKED`로 끝낸다.

### G1 — 하위 그래프 동결

Sol은 목표를 보조정리 또는 하위 프로젝트 노드로 분해한다. 각 노드는 다음을 가진다.

- 하나의 측정 가능한 목적
- 선행 노드
- 담당 Terra
- 시간 및 동시 실행 예산
- 출력과 증거
- 다음 노드에 제공하는 보장

하위 그래프의 노드 식별자, 의존성, 성공 조건을 동결한 뒤 Terra를 생성한다. 이후 구조 변경은 기록된 수정으로만 허용한다.

### G2 — Terra 하위 원장 생성

각 Terra는 부모 계약 안에서 자기 하위 원장을 작성한다. Luna를 만들기 전에 기계적 검사를 통과해야 한다.

- 식별자 고유성
- 의존 대상 존재
- 순환 의존성 부재
- 쓰기 범위 비중첩
- 자식 예산이 부모 예산 안에 있음
- 모든 성공 조건이 참·거짓으로 판정 가능함
- 모든 성공 조건에 명령 또는 증거 경로가 있음

### G3 — 병렬 단계 실행

Terra는 다음 조건을 모두 만족하는 자식만 `READY`로 바꾼다.

- 모든 의존 작업이 `PASS`
- 입력이 존재하고 읽을 수 있음
- 쓰기 범위가 현재 실행 중 작업과 겹치지 않음
- 시간과 누적 작업시간 예산이 남아 있음
- 독립 검증용 예약을 침범하지 않음

같은 단계의 `READY` 작업은 동시 실행할 수 있다.

### G4 — 진행 관찰과 개입

Luna는 정해진 간격마다 진행 관찰을 남긴다. Terra는 원시 로그가 아니라 새 증거와 성공 조건의 이동을 본다. 사소한 일에 빠졌다고 판정되면 범위를 줄이거나 작업을 중단한다.

### G5 — 독립 검증과 판정

Luna의 결과는 형식 검증과 내용 검증을 모두 통과해야 한다. 검증자는 원래 Luna와 다른 문맥에서 입력 파일과 결과를 직접 확인한다.

### G6 — 통합과 부모 보고

Terra는 자기 하위 범위의 통합을 완료하고 네 종류의 보고 중 하나를 부모에게 보낸다.

- 시작 보고
- 단계 통과 보고
- 부모 결정이 필요한 예외 보고
- 최종 결과 보고

### G7 — 종료 또는 재개 가능 상태

Sol은 모든 필수 하위 프로젝트가 `PASS`이고 전역 검증이 통과했을 때만 목표를 완료한다. 그렇지 않으면 원장을 보존하고 `FAIL`, `BLOCKED`, `INCONCLUSIVE` 중 하나로 종료해 재개 가능하게 한다.

## 5. 원장 저장 구조

```text
<goal-state-root>/<goal-id>/
├─ goal-ledger.yaml
├─ subprojects/
│  └─ <terra-id>/
│     ├─ ledger.yaml
│     ├─ tasks/
│     │  └─ <task-id>/
│     │     ├─ order.yaml
│     │     └─ attempts/
│     │        └─ 001/
│     │           ├─ checkpoints/
│     │           └─ result.yaml
│     └─ reports/
└─ evidence-index.yaml
```

소유권은 파일 단위로 분리한다.

- `goal-ledger.yaml`: Sol만 수정
- Terra의 `ledger.yaml`, `order.yaml`, `reports/`: 해당 Terra만 수정
- 각 실행의 `checkpoints/`, `result.yaml`: 해당 Luna만 수정
- 실제 산출물: 업무 지시서에 적힌 단일 Luna만 수정
- `evidence-index.yaml`: Sol만 최종 승인된 증거를 등록

같은 파일을 둘 이상의 에이전트가 수정해야 한다면 병렬 실행하지 않는다. staging 파일을 따로 만들거나 부모가 직렬 통합한다.

## 6. 식별자 규칙

식별자는 원장 안에서 의미와 계층을 드러낸다.

```text
G-20260802-proof-workflow        전체 목표
SP-01                           Sol이 배정한 하위 프로젝트
SP-01-L03                       SP-01의 세 번째 Luna 작업
SP-01-T02                       SP-01의 두 번째 자식 Terra
SP-01-L03-A02                   Luna 작업의 두 번째 실행
```

식별자는 한 번 발행한 뒤 재사용하거나 다른 의미로 바꾸지 않는다. 재실행은 작업 식별자를 유지하고 실행 번호만 증가시킨다.

## 7. Sol 목표 원장 스키마

```yaml
schema_version: 1
identity:
  goal_id:
  owner: sol
  parent_ledger_id: null
  created_at:
  updated_at:

contract:
  objective:
  assumptions: []
  inputs: []
  outputs: []
  write_scope: []
  forbidden_paths: []
  authorization_scope: []

schedule:
  planned_start_at:
  deadline_at:
  wall_clock_budget_minutes:
  agent_effort_budget_minutes:
  verification_reserve_minutes:
  max_concurrency:
  observation_interval_minutes:

graph:
  nodes:
    - subproject_id:
      owner_terra:
      objective:
      dependencies: []
      status: DRAFT
      allocation:
        wall_clock_budget_minutes:
        agent_effort_budget_minutes:
        concurrency_quota:
      ledger_path:
      latest_report_path:

global_gates:
  - id:
    predicate:
    verification_method:
    evidence_path:

status:
verdict:
unresolved: []
```

Sol 목표 원장은 Luna별 명령과 로그를 담지 않는다. Terra의 최신 보고와 전역 게이트만 담는다.

## 8. Terra 하위 원장 스키마

```yaml
schema_version: 1
identity:
  ledger_id:
  parent_ledger_id:
  owner:
  role: terra
  depth:

contract:
  objective:
  assumptions: []
  inputs: []
  outputs: []
  write_scope: []
  forbidden_paths: []
  dependencies: []
  provides: []

schedule:
  phase:
  wall_clock_budget_minutes:
  agent_effort_budget_minutes:
  deadline_at:
  checkpoint_interval_minutes: 15
  concurrency_quota:
  verification_reserve_minutes:

children:
  - task_id:
    type: luna
    objective:
    dependencies: []
    order_path:
    write_scope: []
    status: DRAFT
    current_attempt: 0
    max_attempts: 2

verification:
  form_checks: []
  intent_checks: []
  independent_verifier:
  verdict:
  evidence: []

recovery:
  failed_children: []
  escalations: []

reporting:
  report_to:
  latest_report:
  next_report_trigger:
```

## 9. Luna 업무 지시서

업무 지시서는 발행 후 변경하지 않는다. 수정이 필요하면 새 실행이 아니라 Terra가 새 지시서 버전을 명시적으로 발행한다.

```yaml
schema_version: 1
identity:
  task_id:
  parent_ledger_id:
  issued_by:
  assigned_role: luna-worker

objective:
  measurable_outcome:
  parent_criterion:

inputs:
  exact_paths: []
  required_records: []

scope:
  write_scope: []
  forbidden_paths: []
  allowed_tools: []

dependencies:
  required_passes: []

outputs:
  exact_paths: []
  required_schema:

success_criteria:
  - id:
    predicate:
    verification_command:
    evidence_path:

time_contract:
  mode: read-only | write-execute
  timebox_minutes:
  checkpoint_interval_minutes: 15
  deadline_at:
  max_extension_count: 1
  timeout_action:

execution_rules:
  - Read every existing file before editing it.
  - Do not edit outside write_scope.
  - Do not create new objectives or dependencies.
  - Record direct evidence for every checklist item.
  - Stop and report on missing input, scope drift, or timeout.

return_contract:
  result_path:
  required_verdicts: [PASS, FAIL, BLOCKED, INCONCLUSIVE]
```

## 10. 진행 관찰 스키마

Luna는 기본 15분 단위로 다음을 기록한다.

```yaml
observed_at:
elapsed_minutes:
completed_predicates: []
new_evidence: []
files_touched: []
failed_approaches: []
remaining_work: []
estimated_remaining_minutes:
scope_drift: false
active_operation:
next_interval_objective:
```

다음은 진척으로 인정하지 않는다.

- “계속 조사 중” 같은 근거 없는 서술
- 파일 수나 도구 호출 수만 늘어난 상태
- 같은 실패 접근을 이름만 바꾸어 반복한 상태
- 성공 조건과 무관한 표현·정리·미관 개선
- 확인되지 않은 추측의 개수 증가

## 11. 시간 계획

### 11.1 기본 계획 단위

| 계층 | 기본 단위 | 최대 단위 |
|---|---:|---:|
| Sol | 60분 목표 구간 | 사용자 또는 목표 원장이 정한 전체 기한 |
| Terra | 30분 실행 단계 | 부모가 배정한 하위 프로젝트 기한 |
| Luna | 15분 원자 구간 | 45분; 초과 시 작업 분할 |

제한시간이 주어지지 않으면 Sol은 첫 60분 실행 구간을 계획한다. 구간 종료 시 새 작업 생성을 멈추고 진행을 재평가한다. 무기한 실행은 허용하지 않는다.

### 11.2 두 종류의 예산

예산은 다음 두 값을 모두 가진다.

1. **실제 경과시간 예산**: 병렬 작업과 무관하게 현실에서 흐른 시간
2. **에이전트 누적 작업시간 예산**: 동시에 실행한 모든 에이전트 시간을 합한 값

예를 들어 Luna 열 명이 30분씩 동시에 작업하면 실제 경과시간은 30분, 에이전트 누적 작업시간은 300분이다.

### 11.3 검증 예약

전체 예산의 최소 20%를 검증과 통합에 예약한다. Terra와 Luna는 탐색이나 작성 지연을 이유로 이 예약을 사용할 수 없다. 예약이 남지 않으면 새 작업을 만들지 않고 기존 산출물을 검증한다.

### 11.4 혼합형 제한시간

읽기·탐색 작업은 제한시간에 다음 순서로 처리한다.

1. 진행 관찰 기록
2. 목적을 최소 성공 조건으로 축소
3. 새 증거가 있을 때만 15분 구간 1회 연장
4. 추가 구간에도 진척이 없으면 중단하고 `INCONCLUSIVE` 또는 `BLOCKED`

파일 수정·도구 실행 작업은 제한시간에 다음처럼 처리한다.

1. 실행을 중단
2. 부분 산출물과 명령 출력을 보존
3. 실패 피드백 패킷 작성
4. Terra가 새 실행 여부를 판정

Luna는 자기 연장을 승인할 수 없다. Terra는 Luna에 한 번만 연장을 허용한다. Terra 하위 프로젝트의 연장은 Sol만 승인한다.

## 12. 사소한 일에 빠짐 감지

다음 중 하나가 관찰되면 Terra가 개입한다.

- 한 관찰 구간 동안 성공 조건이 하나도 전진하지 않음
- 같은 실패 접근을 새 근거 없이 두 번 반복
- 업무 지시서에 없는 파일, 정의, 보조정리로 범위 확대
- 핵심 산출물 전에 선택적 문서화나 표현 정리에 두 구간 이상 사용
- 남은 시간 예상이 연속 두 번 증가
- 동일한 명령 실패를 원인 진단 없이 반복
- 새 의존 작업이 필요하지만 원장 변경 요청을 하지 않음
- 읽은 자료가 늘었지만 어떤 결정이 바뀌었는지 설명하지 못함
- 실행 중인 Luna 수가 늘어도 병목 경로가 짧아지지 않음

Terra의 개입은 다음 중 하나여야 한다.

- `CONTINUE`: 직접 증거가 있고 다음 구간 목표가 명확함
- `NARROW`: 최소 성공 조건으로 범위를 축소
- `STOP_AND_REPORT`: 부분 결과를 보존하고 중단
- `REASSIGN`: 새 Luna에 더 명확한 지시서로 재배정
- `ESCALATE`: 부모 Terra 또는 Sol의 결정 요청

## 13. 상태와 단계 전이

```text
DRAFT → READY → RUNNING → VERIFYING → PASS
                           ├→ FAIL → REPAIR_READY → RUNNING
                           ├→ BLOCKED → ESCALATED
                           └→ INCONCLUSIVE → REDESIGN_REQUIRED
```

규칙:

- `PASS`: 판정 조건과 직접 증거가 모두 충족됨
- `FAIL`: 직접 증거가 조건 위반을 보임
- `BLOCKED`: 입력, 권한, 의존성, 환경이 없어 판정 불가
- `INCONCLUSIVE`: 증거가 결론들을 구분하지 못함
- `SKIPPED`: 선택 작업이며 Terra가 생략 근거와 영향 없음의 증거를 기록함

`PASS`만 의존성을 충족한다. `FAIL`, `BLOCKED`, `INCONCLUSIVE`가 하나라도 있는 필수 노드는 다음 단계를 열지 않는다.

## 14. 단계 계산과 병렬 실행

Terra는 선행 작업이 없는 노드를 첫 단계에 둔다. 이후 노드는 모든 선행 노드가 통과한 가장 이른 단계에 둔다.

다음 조건에서는 같은 단계여도 직렬 실행한다.

- 쓰기 범위가 겹침
- 동일한 외부 상태를 변경함
- 한 작업의 실제 출력이 다른 작업의 입력을 결정함
- 동일한 공유 색인, 기록, 원장을 수정함
- 한 작업의 실패가 다른 작업의 필요성을 없앨 수 있음

동시 실행 수는 부모가 배정한 한도와 현재 환경의 실제 한도 중 작은 값을 사용한다.

## 15. 검증 계약

### 15.1 형식 검증

- 필수 파일 존재
- 구조화 데이터 문법과 스키마
- 출력 경로와 쓰기 범위
- 명령 종료 상태
- 테스트 및 빌드 결과
- 식별자와 의존성 일치
- 시간·실행 횟수 한도

### 15.2 내용 검증

- 산출물이 목표를 실제로 달성하는가
- 주장과 증거가 직접 연결되는가
- 숨은 가정이나 결론 강화가 없는가
- 부분 성공을 전체 성공으로 보고하지 않았는가
- 반례나 실패 사례를 누락하지 않았는가
- 변경이 요청된 동작을 실제로 수행하는가

내용 검증이 없으면 검증 결과에 `form-only`를 명시하고 최종 `PASS`로 사용하지 않는다.

### 15.3 독립성 등급

| 대상 | 최소 검증자 |
|---|---|
| 단순 Luna 결과 | 다른 문맥의 Luna 검증자 |
| 파일 수정과 실행 결과 | Luna 검증자 + Terra 판정 |
| 핵심 알고리즘·보조정리 | 별도 Terra 검증자 |
| 전체 정리·최종 연구 결론 | Sol의 최종 증거 연결 감사 |

검증자는 작성자의 요약을 정답으로 받지 않는다. 원본 입력, 산출물, 명령 결과, 판정 규칙을 직접 본다.

## 16. 실패 피드백과 재실행

모든 비통과 결과는 다음 패킷을 만든다.

```yaml
failed_item:
observed:
expected:
evidence:
likely_cause:
minimal_repair:
rerun_scope:
rerun_command:
owner:
```

재실행 규칙:

1. 기존 업무 지시서와 결과를 덮어쓰지 않는다.
2. 동일 작업 식별자 아래 실행 번호를 증가시킨다.
3. 첫 실패가 국소 오류이면 같은 thread에 최소 수리 지시를 한 번 보낼 수 있다.
4. 명제 오인, 범위 이탈, 두 번째 실패이면 새 Luna thread에서 시작한다.
5. 기본 Luna 실행 한도는 두 번이다.
6. 한도 소진 시 Terra가 지시서를 재설계하거나 부모에게 승격한다.
7. 같은 성공 조건을 다시 검사하고 다른 쉬운 조건으로 바꾸지 않는다.
8. 부분 결과는 명시적 입력으로만 재사용하며 숨은 문맥으로 전달하지 않는다.

## 17. Terra의 부모 보고

Terra는 Luna 원시 로그를 Sol에게 보내지 않는다. 다음 네 종류의 보고만 사용한다.

### 17.1 시작 보고

```yaml
report_type: STARTED
subproject_id:
objective:
child_count:
planned_phases:
allocated_budget:
first_gate:
```

### 17.2 단계 통과 보고

```yaml
report_type: PHASE_PASS
subproject_id:
phase:
passed_nodes: []
evidence_index: []
elapsed_minutes:
remaining_critical_path: []
```

### 17.3 결정 요청 보고

```yaml
report_type: DECISION_REQUIRED
subproject_id:
verdict:
failed_or_blocked_items: []
dependency_impact: []
budget_remaining:
options: []
recommended_option:
decision_needed_from_parent:
```

### 17.4 최종 보고

```yaml
report_type: COMPLETE
subproject_id:
verdict:
outputs: []
evidence_index: []
verification_summary:
elapsed_minutes:
agent_effort_minutes:
unresolved: []
downstream_guarantees: []
```

최상위 Terra는 Sol에게 직접 보고한다. 자식 Terra는 부모 Terra에게 보고하고, 부모 Terra가 압축된 결과를 Sol 보고에 포함한다.

## 18. 수식 증명 워크플로

### 18.1 Sol의 책임

Sol은 증명 작업을 시작하기 전에 다음을 확정한다.

- 정리의 정확한 명제
- 변수의 영역과 모든 정량화
- 허용되는 가정
- 결론의 정확한 강도
- 보조정리 목록과 의존성
- 각 보조정리가 다음 단계에 제공하는 보장
- 증명 성공 조건과 반례 조건

Sol은 보조정리의 구조와 탐색 논지를 동결한다. Terra는 명시적 수정 요청 없이 보조정리의 의미를 바꾸지 않는다.

### 18.2 Terra의 책임

각 보조정리를 담당하는 Terra는 두 경쟁 경로를 설계한다.

```text
보조정리 Terra 원장
├─ 증명 경로
│  ├─ 직접 증명
│  ├─ 경우 분해
│  ├─ 기존 정리 적용 조건 확인
│  └─ 형식·계산 검산
└─ 반증 경로
   ├─ 경계 사례
   ├─ 작은 차원·유한 사례
   ├─ 가정 제거 실험
   └─ 반례 탐색
```

Terra는 증명과 반증을 모두 시도하고, Luna 작업을 의존성과 쓰기 범위가 겹치지 않게 배정한다.

### 18.3 Luna의 책임

Luna는 다음과 같은 원자 작업을 직접 수행한다.

- 정의와 기존 정리의 정확한 파일·절 위치 확인
- 증명 조각 작성
- 작은 사례 계산
- 경계 조건과 반례 후보 생성
- 형식 증명 도구 실행
- 실패한 전술과 오류 출력 기록
- 지정된 증명 파일 수정

### 18.4 판정 규칙

- 완전한 증명과 독립 내용 검증이 모두 통과해야 보조정리가 `PASS`
- 유효한 반례가 검증되면 보조정리는 `FAIL`
- 반례를 찾지 못한 것은 증명의 근거가 아님
- 증명도 반례도 확정되지 않으면 `INCONCLUSIVE`
- 증명과 반례가 동시에 보고되면 반례를 우선하고 증명의 오류 위치를 조사
- 모든 보조정리가 통과해도 Sol이 전체 의존성 연결과 최종 정리 도출을 별도로 감사

### 18.5 보조정리 업무 지시서 예시

```yaml
task_id: SP-LEMMA-02-L01
objective:
  measurable_outcome: Lemma 2의 n=1,2 경계 사례를 검산한다.
inputs:
  exact_paths:
    - <theorem-source>
    - <lemma-registry>
scope:
  write_scope:
    - <task-specific-result>
  forbidden_paths:
    - <canonical-theorem-file>
success_criteria:
  - id: C1
    predicate: 모든 허용 입력의 계산 결과가 명제와 일치하거나 정확한 반례를 제공한다.
    verification_command: <bounded-command>
    evidence_path: <task-specific-result>
time_contract:
  mode: write-execute
  timebox_minutes: 30
  checkpoint_interval_minutes: 15
```

## 19. 일반 연구와 코딩 적용

### 19.1 연구 문헌 작업

Sol은 연구 질문과 주장 구조를 동결한다. Terra는 주장별 또는 문헌군별 원장을 관리한다. Luna는 논문 파일 읽기, 정확한 인용 위치 추출, 표 작성, 반대 증거 탐색을 수행한다. 인용 키는 `D:/raw/bibtex/references.bib`에서 확인하며 발명하지 않는다.

### 19.2 소프트웨어 기능 구현

Sol은 사용자 행동, 공개 인터페이스, 통합 조건을 확정한다. Terra는 모듈 또는 기능 묶음을 관리한다. Luna는 겹치지 않는 파일 범위에서 테스트, 구현, 문서 작업을 수행한다. 공유 파일과 통합 브랜치는 Terra 또는 Sol의 직렬 작성으로 제한한다.

### 19.3 실험 작업

Sol은 사전등록된 결정 규칙과 전체 실험 그래프를 고정한다. Terra는 실험군 또는 분석 단계를 관리한다. Luna는 설정 검증, 로그 분석, 산출물 수집, 계산을 담당한다. 제출·취소·삭제 같은 외부 변경은 별도의 사용자 승인과 실험 계약을 따른다.

## 20. 사용 방법

### 20.1 새 목표 시작

```text
이 목표를 Sol–Terra–Luna 재귀 원장으로 실행해. 총 제한시간은 3시간이다.
```

시간을 생략하면 첫 60분 구간이 자동 계획된다.

### 20.2 수식 증명

```text
이 정리를 보조정리 그래프로 분해하고, 각 보조정리에 대해 증명과 반증을 병렬 시도해. Luna가 관련 파일을 직접 읽고 쓰며, Terra가 보조정리별 원장을 관리하게 해.
```

### 20.3 현재 상태 확인

```text
현재 목표 원장을 읽고 단계별 상태, 사용 시간, 남은 병목 작업, 부모 결정이 필요한 항목만 보고해.
```

### 20.4 실패 작업 재실행

```text
실패한 작업만 피드백 패킷을 반영해 재실행해. 기존 성공 조건과 증거 기준은 유지해.
```

### 20.5 안전한 중단

```text
현재 실행을 중단하고 검증된 부분 결과, 미완료 작업, 재개 조건을 원장에 보존해.
```

### 20.6 재개

```text
<goal-id> 원장을 다시 읽고, 의존성이 PASS이며 예산이 남은 READY 작업부터 재개해.
```

### 20.7 범위 축소

```text
남은 시간 안에 최종 목표에 직접 필요한 작업만 유지하고 선택 작업을 근거와 함께 SKIPPED로 닫아.
```

## 21. 운영자 체크리스트

### 시작 전

- [ ] 목표, 가정, 입력, 출력이 정확하다.
- [ ] 전체 쓰기 범위와 금지 경로가 고정되었다.
- [ ] 실제 경과시간과 누적 작업시간 예산이 있다.
- [ ] 검증 예약이 전체 예산의 20% 이상이다.
- [ ] Sol의 하위 그래프에 순환이 없다.
- [ ] 각 Terra의 목적과 보고 경로가 분명하다.

### Luna 생성 전

- [ ] 하나의 측정 가능한 목적만 있다.
- [ ] 정확한 입력·출력 경로가 있다.
- [ ] 쓰기 범위가 다른 실행 작업과 겹치지 않는다.
- [ ] 모든 의존성이 `PASS`다.
- [ ] 성공 조건이 이진 판정 가능하다.
- [ ] 제한시간과 중단 동작이 있다.
- [ ] 결과 및 증거 경로가 있다.

### 단계 종료 전

- [ ] 형식 검증과 내용 검증을 분리했다.
- [ ] 독립 검증자가 원본 산출물을 직접 확인했다.
- [ ] 모든 필수 작업이 `PASS`다.
- [ ] 실패 작업은 피드백 패킷을 갖는다.
- [ ] 다음 단계의 쓰기 범위와 예산이 유효하다.
- [ ] Terra 보고가 원시 로그 없이 필요한 증거를 포함한다.

### 전체 종료 전

- [ ] Sol이 보조정리 또는 하위 프로젝트 연결을 감사했다.
- [ ] 전역 성공 조건에 직접 증거가 있다.
- [ ] 미해결 항목이 명시되었다.
- [ ] 실제 경과시간과 누적 작업시간이 기록되었다.
- [ ] 재개 또는 재실행에 필요한 원장이 보존되었다.
- [ ] Personal Research 산출물은 정본 경로와 미러 상태를 확인했다.

## 22. 문제 해결

### Terra가 너무 많은 원시 정보를 Sol에 보낸다

보고를 네 가지 형식 중 하나로 다시 작성한다. 완료 노드 목록, 증거 색인, 시간, 남은 병목, 부모 결정만 유지한다.

### Luna가 범위를 넓힌다

즉시 중단하고 `scope_drift: true`를 기록한다. Terra가 새 의존 작업 또는 업무 지시서 수정이 필요한지 판정한다.

### 여러 Luna가 같은 파일을 수정해야 한다

병렬 실행하지 않는다. 별도 staging 파일을 만들거나 한 Luna가 순차 통합한다.

### 탐색이 끝나지 않는다

추가 탐색이 어떤 결정을 바꿀지 적는다. 답할 수 없으면 중단한다. 새 증거가 있을 때만 15분 구간을 한 번 추가한다.

### 증명과 반례가 충돌한다

반례를 독립 검산한다. 유효하면 보조정리를 `FAIL`로 두고 증명 조각의 최초 오류를 찾는 새 작업을 발행한다.

### 검증자가 작성자 요약만 반복한다

검증을 무효로 한다. 원본 입력, 결과 파일, 명령, 판정 규칙만 제공한 새 문맥의 검증자를 생성한다.

### 재실행이 전체 작업을 반복한다

실패 피드백 패킷의 `minimal_repair`와 `rerun_scope` 밖의 작업을 금지한다. 이미 `PASS`인 노드는 다시 실행하지 않는다.

### 시간은 남았지만 누적 작업시간을 소진했다

새 Luna 생성을 멈춘다. 현재 결과를 검증하고, 추가 에이전트 비용이 필요한 경우 부모에게 예산 변경을 요청한다.

## 23. 안전 경계

- 사용자 요청 범위를 넘어서는 외부 쓰기, 제출, 취소, 삭제, 구매를 하지 않는다.
- 보호 경로는 별도 사용자 허가 없이 수정하지 않는다.
- Personal Research에서는 iCloud 투영 경로를 직접 수정하지 않고 `D:/raw/` 정본에서 작업한다.
- 공유 색인, daily log, handoff index는 해당 직렬화 도구를 통한다.
- 비밀, 토큰, 비밀번호, 개인 키를 원장이나 보고에 기록하지 않는다.
- 원격 실행과 실험 변경은 별도의 승인 및 실험 생명주기 계약을 따른다.
- BibTeX 키를 발명하지 않는다. 미해결 인용은 `[NEEDCITE: author year topic]`으로 남긴다.

## 24. Codex 구현 표면

승인된 설계는 이 서버의 내구성 있는 프로젝트 로컬 Codex 표면에 다음 파일로 구현한다.
Windows 전역 경로와 `~/.codex/`는 이 서버의 sync 신뢰 경계 밖이므로 사용하지 않는다.

```text
<repo>/
├─ .agents/skills/goal-graph-orchestrator/
│  ├─ SKILL.md
│  ├─ agents/openai.yaml
│  ├─ scripts/ledger.py
│  └─ references/ledger.schema.json
├─ .codex/agents/
   ├─ terra-manager.toml
   ├─ terra-verifier.toml
   ├─ luna-worker.toml
   └─ luna-verifier.toml
└─ lab/goal-state/<goal-id>/
```

프로젝트 `AGENTS.md`에는 이 문서를 가리키는 짧은 적용 규칙만 둔다. 상세 규칙을 복제하지
않는다. 동시 하위 에이전트 수는 원장 한도와 실제 환경의 노출 한도 중 작은 값을 사용한다.
현재 하위 에이전트 표면에 Luna 모델이 노출되지 않으면 Luna는 capability seat로만 사용하고,
결과에 실제 runtime model과 agent/thread ID를 기록한다.

## 25. 구현 및 배포 검증 기준

구현은 다음이 모두 통과해야 완료된다.

### 기계적 검사

- 원장 스키마 검증
- 중복 식별자 차단
- 존재하지 않는 의존성 차단
- 순환 의존성 차단
- 동시 쓰기 범위 충돌 차단
- 부모 예산 초과 차단
- 허용되지 않은 상태 전이 차단
- 검증 증거 없는 `PASS` 차단

### 독립 에이전트 시험

- 시간 압박 때문에 원장을 생략하지 않는다.
- Terra가 성공 조건 없이 Luna를 만들지 않는다.
- 여러 Luna가 같은 파일을 동시에 수정하지 않는다.
- 증명 실패를 반례로 잘못 판정하지 않는다.
- 제한시간 후 사소한 탐색을 계속하지 않는다.
- 실패한 작업만 재실행한다.
- Terra가 원시 로그를 Sol에 과도하게 전달하지 않는다.
- 새 문맥의 검증자가 작성자의 판단을 그대로 반복하지 않는다.

### 완료 판정

모든 필수 검사가 `PASS`이고 미해결 구현 항목이 없을 때만 `implementation_status`를 `implemented-and-verified`로 바꾼다.

## 26. 정본과 참고 자료

- 공통 연구 협업 계약: `lab/research-collaboration-contract.md`
- Claude/Codex parity 분석: `lab/claude-harness-analysis-20260715.md`
- OpenAI GPT-5.6 모델 지침: https://developers.openai.com/api/docs/guides/latest-model
- Codex 하위 에이전트 문서: https://learn.chatgpt.com/docs/agent-configuration/subagents
- Codex 사용자 설정 문서: https://learn.chatgpt.com/docs/customization/overview

## 27. 변경 기록

### 1.1.0 — 2026-08-02

- 프로젝트 로컬 Codex skill, Sol–Terra–Luna 역할, 상태 엔진 및 JSON Schema 구현
- Terra 중간 프로젝트와 한 단계 재귀 Terra, 실패 피드백·immutable 재발행 구현
- 실제 runtime model/agent binding, 기준별 독립 검증, 사용량·검증 reserve 강제
- task/Terra의 Sol fallback을 root-contract escalation packet으로 제한
- global gate와 최종 `finish` 증거 검증 구현
- 적대적 회귀 테스트 25개 및 fresh-context `gpt-5.6-sol` 독립 감사 PASS

### 1.0.0 — 2026-08-02

- Sol–Terra–Luna 재귀 계층 승인본 작성
- 상세 재귀 원장과 단일 작성자 계약 확정
- 실제 경과시간 및 에이전트 누적 작업시간 통제 추가
- 읽기 작업과 쓰기·실행 작업의 혼합형 제한시간 확정
- 수식 증명의 보조정리별 증명·반증 패턴 추가
- 사용, 상태 확인, 중단, 재개, 재실행 절차 통합
