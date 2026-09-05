#!/usr/bin/env bash
# Machine gate for Track-A (Lean) lemmas: assert every audited theorem depends
# ONLY on Lean's three standard axioms and never on `sorryAx`.
#
# Why this exists: `#print axioms` reports to stdout and `lake env lean` exits 0
# even when a theorem is proved by `sorry`. Check.lean was therefore advisory,
# not a gate. This wrapper turns it into an exit code so it can be dropped into
# the goal-graph task template as `verification_command`
# (lab/11_governance/11_lab-root/goal-graph-orchestration.md §18.5).
#
# Usage: scripts/lean_axiom_gate.sh <lean-project-dir> [check-file]
# Exit:  0 = all audited theorems sorry-free and axiom-clean
#        1 = sorryAx or a non-standard axiom found, or no theorems audited
#        2 = build/toolchain failure (inconclusive, NOT a pass)

set -uo pipefail

PROJ="${1:?usage: lean_axiom_gate.sh <lean-project-dir> [check-file]}"
CHECK="${2:-Check.lean}"

ALLOWED='propext|Classical.choice|Quot.sound'

cd "$PROJ" || { echo "GATE-ERROR: no such project dir: $PROJ" >&2; exit 2; }
[ -f "$CHECK" ] || { echo "GATE-ERROR: no check file: $PROJ/$CHECK" >&2; exit 2; }

# shellcheck disable=SC1090
[ -f "$HOME/.elan/env" ] && . "$HOME/.elan/env"
export PATH="$HOME/.elan/bin:$PATH"
command -v lake >/dev/null || { echo "GATE-ERROR: lake not on PATH" >&2; exit 2; }

out=$(lake env lean "$CHECK" 2>&1)
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "GATE-ERROR: '$CHECK' failed to elaborate (exit $rc) — inconclusive, not a pass" >&2
  echo "$out" >&2
  exit 2
fi

# Lines look like: 'Ns.thm' depends on axioms: [propext, Classical.choice, Quot.sound]
mapfile -t lines < <(printf '%s\n' "$out" | grep "depends on axioms")

# A theorem proved with no axioms at all prints "does not depend on any axioms".
mapfile -t freelines < <(printf '%s\n' "$out" | grep "does not depend on any axioms")

total=$(( ${#lines[@]} + ${#freelines[@]} ))
if [ "$total" -eq 0 ]; then
  echo "GATE-FAIL: $CHECK audited 0 theorems — an empty gate is not a pass" >&2
  printf '%s\n' "$out" >&2
  exit 1
fi

fail=0
for l in "${lines[@]}"; do
  name=${l%%\' depends*}; name=${name#\'}
  axioms=${l##*: [}; axioms=${axioms%]}
  bad=""
  IFS=',' read -ra parts <<< "$axioms"
  for a in "${parts[@]}"; do
    a="${a// /}"
    [ -z "$a" ] && continue
    if ! printf '%s' "$a" | grep -qE "^($ALLOWED)$"; then
      bad="$bad $a"
    fi
  done
  if [ -n "$bad" ]; then
    echo "GATE-FAIL: $name depends on non-standard axiom(s):$bad"
    fail=1
  fi
done

if [ "$fail" -ne 0 ]; then
  echo "GATE-FAIL: $PROJ/$CHECK — $total theorem(s) audited, at least one unsound for Track-A 'proven(formal)'" >&2
  exit 1
fi

echo "GATE-PASS: $total theorem(s) audited in $PROJ/$CHECK; axioms ⊆ {propext, Classical.choice, Quot.sound}; no sorryAx."
exit 0
