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

## Observation 2: Euclidean Distance Is Not Cosine Distance

### Component Contract

In this simplified reproduction, `cosine_distance(a, b)` receives two
finite, non-zero, one-dimensional embeddings with matching shapes.

Both embeddings have already been L2-normalized, so their norms are
approximately equal to 1.

The diagnostic inputs below therefore satisfy the component contract.

### Broken Implementation

The implementation intentionally uses Euclidean distance instead of cosine
distance:

```python
def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return the distance between two normalized embeddings."""

    return float(np.linalg.norm(a - b))
```

Euclidean distance is a plausible embedding metric, but it is not the metric
required by this component.

### Diagnostic Experiment

The observed results were:

```text
same: 0.0
orthogonal: 1.4142135623730951
opposite: 2.0
```

The expected cosine distances are:

```text
same: 0.0
orthogonal: 1.0
opposite: 2.0
```

For identical unit vectors, both Euclidean distance and cosine distance return
0.

For opposite unit vectors, both metrics return 2.

These two cases therefore cannot distinguish the incorrect Euclidean
implementation from the required cosine-distance implementation.

For orthogonal unit vectors, cosine distance is 1, while Euclidean distance is
the square root of 2, approximately 1.414.

The orthogonal case is a contract-valid input that exposes the metric mismatch.

### Verifier Implication

A weak verifier that checks only identical and opposite vectors could accept
this incorrect implementation.

An orthogonal case provides stronger evidence that the function implements
cosine distance rather than another plausible embedding metric.

This does not yet define the final Verifier V0. It records one observed system
failure and one potential verifier blind spot.

### Current Scope

This observation concerns only scalar distance between two normalized
embeddings.

Pairwise distance, window scoring, calibration, thresholds, debouncing, and
final alert behavior have not yet been implemented.
