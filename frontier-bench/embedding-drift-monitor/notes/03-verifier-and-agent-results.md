# Verifier Evolution and Agent Repair Results

## Scope and Attribution

This project is an educational micro-reproduction inspired by Frontier-Bench's embedding-drift-monitor task.

It is not an exact reproduction of the official benchmark environment.

The task implementation, candidate repairs, verifier versions, shortcut experiment, and Codex trial described below are part of our simplified reproduction and experimental design.

## Research Questions

This experiment studies:

- meaningful implementation defects,
- weak-verifier false acceptance,
- honest but incomplete repair,
- verifier-targeted shortcuts,
- partial credit,
- verifier hardening,
- and visible-verifier agent repair.

## Known Defects

The broken implementation contains six known component defects:

1. unsafe zero-vector normalization,
2. Euclidean distance used instead of cosine distance,
3. transposed pairwise distance output,
4. global averaging instead of nearest-reference aggregation,
5. mean-based thresholding instead of quantile calibration,
6. stale shift streaks in the alert debouncer.

## Candidate Implementations

### Broken implementation

The original candidate contains all six known defects.

### Honest partial repair

This candidate correctly repairs zero-safe normalization but leaves the other five known defects unchanged.

### V0 normalization shortcut

This candidate is designed to pass the weak V0 normalization test without implementing general row-wise normalization.

It normalizes only the first row and replaces all remaining rows with zeros.

### Reference implementation

The reference implementation repairs all six known component defects.

### Codex Agent repair

Codex CLI 0.145.0 with GPT-5.6 Sol was started from the broken implementation.

The agent could inspect the visible V2 tests and modify implementation files under `src/`.

It was instructed not to modify tests, datasets, Git metadata, or other solution branches.

## Experiment Results

| Candidate | V0 | V1 | V2 | Semantic status |
|---|---:|---:|---:|---|
| Broken implementation | 5/6 | 6/7 | 2/8 | All six known defects remain |
| Honest partial repair | 6/6 | 7/7 | 3/8 | Only normalization is repaired |
| V0 normalization shortcut | 6/6 | 6/7 | 2/8 | Exploits the weak V0 normalization test |
| Reference implementation | 6/6 | 7/7 | 8/8 | Repairs all six known component defects |
| Codex Agent repair | 6/6 | 7/7 | 8/8 | Repairs all six known component defects |

## Verifier Evolution

### V0: Weak Component Verifier

V0 contains six tests, but several assertions use weak properties or convenient inputs.

V0 falsely accepts both:

- the honest partial repair,
- and the verifier-targeted normalization shortcut.

A V0 score of 6/6 therefore does not imply that the monitor is semantically correct.

### V1: Targeted Shortcut Closure

V1 adds a multi-row normalization test.

This new test rejects the known normalization shortcut because the shortcut zeros valid rows after the first row.

The honest partial repair still passes because its normalization implementation is genuinely correct.

However, V1 still does not precisely test the other five component contracts.

### V2: Hardened Component Verifier

V2 contains eight tests covering:

- zero-safe normalization,
- general row-wise normalization,
- cosine-distance geometry,
- pairwise axis orientation,
- nearest-reference aggregation,
- requested quantile calibration,
- basic alert triggering and clearing,
- and shift-streak reset behavior.

V2 detects all six known component defects in the broken implementation.

It also distinguishes the honest partial repair from the complete repairs.

## Codex Agent Trial

The visible-verifier trial started with the following V2 result:

- 6 failed
- 2 passed

The agent then:

1. inspected the implementation and visible verifier,
2. identified the six implementation defects,
3. modified only `src/monitor.py`,
4. reran the verifier,
5. and achieved 8/8 V2 tests passing.

An independent regression check across V0, V1, and V2 produced:

- 21 passed
- 0 failed

## Interpretation

The Codex trial demonstrates that a coding agent can use visible verifier feedback to diagnose and repair multiple implementation defects.

It does not demonstrate hidden-test generalization.

The 8/8 result also does not prove complete task correctness because V2 remains a component-level verifier.

The reference implementation and Codex repair are indistinguishable under V2, but this does not prove that their implementations are identical or equally robust outside V2's tested contracts.

## Partial Credit Interpretation

The V2 test count provides simple component-level partial credit.

For example:

- the broken implementation receives 2/8,
- the honest partial repair receives 3/8,
- and the complete repairs receive 8/8.

However, these fractions should not be interpreted as exact percentages of system correctness.

Each test has equal numerical weight, while the real-world importance of different defects may not be equal.

## Remaining Verifier Gaps

The current verifier does not yet check:

- complete end-to-end monitor behavior,
- alert behavior over the generated stream,
- reference-data immutability,
- held-out calibration integrity,
- command-line interface behavior,
- test or data tampering,
- hidden randomized inputs,
- or broader metamorphic properties.

## Main Lessons

1. A passing test suite is meaningful only relative to the wrong solutions it excludes.
2. Weak property checks can accept both incomplete repairs and deliberate shortcuts.
3. Concrete false-acceptance examples provide useful guidance for verifier hardening.
4. Non-square inputs expose axis-transposition bugs that square inputs can hide.
5. Stateful components require interruption and reset sequences, not only happy paths.
6. Visible-verifier success should not be reported as hidden-test benchmark performance.
7. Test-count partial credit is useful but should not be confused with exact semantic correctness.
