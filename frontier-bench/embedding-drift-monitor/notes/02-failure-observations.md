# Failure Observations

## Scope

This document records observations from our simplified educational reproduction inspired by the Frontier-Bench embedding-drift-monitor task.

These are not official Frontier-Bench experiment results.

## Observation 1: Unsafe L2 Normalization

### Broken Implementation

The initial monitor normalizes each embedding using:

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

This implementation does not handle embeddings whose L2 norm is zero.

The bug is intentionally preserved in the initial broken implementation.

### Minimal Experiment

The normalization function was tested with two vectors:

    [3.0, 4.0]
    [0.0, 0.0]

The first vector has an L2 norm of 5 and was normalized correctly:

    [0.6, 0.8]

The zero vector has an L2 norm of 0. Dividing it by its norm produced:

    [nan, nan]

The program also emitted:

    RuntimeWarning: invalid value encountered in divide

The final finite-value check returned:

    all finite: False

### Stream Experiment

The same normalization function was applied independently to all nine public stream windows.

The observed results were:

    w01: finite=True, nan_values=0
    w02: finite=True, nan_values=0
    w03: finite=False, nan_values=384
    w04: finite=True, nan_values=0
    w05: finite=True, nan_values=0
    w06: finite=True, nan_values=0
    w07: finite=True, nan_values=0
    w08: finite=True, nan_values=0
    w09: finite=True, nan_values=0

The only corrupted input was located at window 3, position 5.

The embedding dimension was 384, so the single zero embedding produced 384 NaN values after unsafe normalization.

### Failure Chain

    zero embedding
    -> zero L2 norm
    -> division by zero
    -> NaN normalized embedding
    -> possible NaN distance
    -> possible NaN drift score
    -> unreliable threshold comparison
    -> unreliable alert behavior

### Why This Matters

The program did not terminate with an exception.

Instead, it emitted a runtime warning and continued returning an array containing invalid numerical values.

A direct crash would be visible and relatively easy to diagnose. Silent numerical corruption is more dangerous because downstream components may continue executing while making invalid monitoring decisions.

A successful process exit therefore does not prove that the monitor produced valid results.

### Verifier Implications

A verifier should not check only whether the program runs without crashing.

It should also verify that:

- normalized embeddings remain finite;
- zero embeddings do not crash the monitor;
- zero embeddings do not produce NaN or infinite values;
- downstream drift scores remain finite;
- one corrupted embedding does not contaminate unrelated windows;
- alert decisions remain meaningful after corrupted input is processed.

These checks test observable task properties rather than requiring one exact implementation.

### Current Design Decision

The unsafe normalization bug remains intentionally present in the initial broken implementation.

The Human Agent should first observe and diagnose the failure before implementing a zero-safe normalization rule.

This observation will later inform the design of the initial verifier and its hidden robustness cases.
