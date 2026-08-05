#!/usr/bin/env bash
# Surface tests that FAILED and then PASSED within the same ctest run.
#
# CI runs ctest with `--repeat until-pass:N`, which is the right call for a
# gate — a genuine flake should not block a merge on someone else's bad luck.
# But it also means a test can fail for real and still leave the job green,
# with the only trace buried in LastTest.log. That is how a cr_detached_spawn
# failure on macos-14 nearly went unnoticed (issue #919): the check list said
# pass, and the failure was 15 s of daemon-startup breakage.
#
# So: still let the retry hold the gate, but never let it hide the event. Any
# test with both a failing and a passing attempt is reported as a GitHub
# ::warning:: annotation (visible on the PR without opening logs) and written
# to the step summary. Exit status is always 0 — this reports, it does not gate.
#
# Usage: report_flaky_tests.sh <path-to-LastTest.log> <label>
set -uo pipefail

LOG="${1:?usage: report_flaky_tests.sh <LastTest.log> <label>}"
LABEL="${2:-tests}"
SUMMARY="${GITHUB_STEP_SUMMARY:-/dev/null}"

if [ ! -f "$LOG" ]; then
  echo "## Flaky-test report ($LABEL)" >> "$SUMMARY"
  echo "(LastTest.log not found at $LOG)" >> "$SUMMARY"
  exit 0
fi

# ctest writes one line per attempt:
#   Test #114: cr_detached_spawn ......***Failed   15.36 sec
#   Test #114: cr_detached_spawn ......   Passed    1.41 sec
# Collect the set of test names seen failing and seen passing, then intersect.
failed=$(grep -oE 'Test +#[0-9]+: +[A-Za-z0-9_.-]+ .*\*\*\*(Failed|Exception|Timeout)' "$LOG" \
         | sed -E 's/Test +#[0-9]+: +([A-Za-z0-9_.-]+) .*/\1/' | sort -u)
passed=$(grep -oE 'Test +#[0-9]+: +[A-Za-z0-9_.-]+ .* Passed' "$LOG" \
         | sed -E 's/Test +#[0-9]+: +([A-Za-z0-9_.-]+) .*/\1/' | sort -u)

flaky=$(comm -12 <(printf '%s\n' "$failed") <(printf '%s\n' "$passed") | sed '/^$/d')

echo "## Flaky-test report ($LABEL)" >> "$SUMMARY"
if [ -z "$flaky" ]; then
  echo "No test both failed and passed in this run." >> "$SUMMARY"
  exit 0
fi

echo "These tests FAILED and then PASSED on retry. The job is green because" >> "$SUMMARY"
echo "\`--repeat until-pass\` did its job, but each of these is a real failure:" >> "$SUMMARY"
echo '```' >> "$SUMMARY"
while IFS= read -r t; do
  [ -z "$t" ] && continue
  echo "::warning title=Flaky test ($LABEL)::$t failed and passed within the same run — green via ctest retry, not because it is healthy"
  grep -E "Test +#[0-9]+: +$t " "$LOG" | sed 's/^/  /' >> "$SUMMARY"
done <<< "$flaky"
echo '```' >> "$SUMMARY"
exit 0
