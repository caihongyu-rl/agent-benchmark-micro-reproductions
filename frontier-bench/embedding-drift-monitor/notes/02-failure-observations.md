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

## Observation 3: Pairwise Distance Matrix Is Transposed

### Component Contract

In this simplified reproduction, `pairwise_cosine_distances(current, reference)`
receives two matrices of L2-normalized embeddings.

If the input shapes are:

```text
current.shape = (n_current, dimension)
reference.shape = (n_reference, dimension)
```

the required output shape is:

```text
(n_current, n_reference)
```

Each output row represents one current embedding, and each output column
represents one reference embedding.

### Broken Implementation

The implementation correctly calculates the pairwise cosine-distance values,
but intentionally transposes the result:

```python
def pairwise_cosine_distances(
    current: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Return pairwise cosine distances to the reference embeddings."""

    similarities = current @ reference.T
    distances = 1.0 - similarities
    return distances.T
```

### Diagnostic Experiment

The input shapes were:

```text
current shape: (2, 2)
reference shape: (3, 2)
```

The observed output was:

```text
distance shape: (3, 2)

[[0. 1.]
 [1. 0.]
 [2. 1.]]
```

The component contract instead requires an output shape of `(2, 3)`:

```text
[[0. 1. 2.]
 [1. 0. 1.]]
```

The individual distance values are present, but the meanings of the two matrix
axes have been reversed.

### Downstream Risk

A downstream scoring function may assume that each row represents one current
embedding and reduce each row to its nearest reference distance.

With the transposed matrix, the same operation instead processes one row per
reference embedding. This can change the resulting window score even though the
individual pairwise values were calculated correctly.

### Verifier Implication

A weak verifier using equal-sized current and reference sets may fail to expose
this error because both the correct and transposed outputs have the same shape.

If the test matrix is also symmetric, transposition does not change its visible
values.

Using different numbers of current and reference embeddings produces a
non-square output and makes the two axes distinguishable.

This does not yet define the final Verifier V0. It records one observed system
failure and one potential verifier blind spot.

### Current Scope

This observation concerns only the shape and axis meaning of the pairwise
distance matrix.

Window scoring, calibration, thresholds, debouncing, and final alert behavior
have not yet been implemented.

## Observation 4: Window Score Uses the Global Mean

### Simplified Scoring Contract

In this educational reproduction, the window drift score is defined as the
mean nearest-reference distance.

For each current embedding, the scoring function should first select its
smallest distance to any reference embedding. It should then average those
nearest-reference distances across the current window.

This is our simplified scoring design, not an official benchmark formula.

### Broken Implementation

The implementation intentionally averages every value in the pairwise distance
matrix:

```python
def window_drift_score(distances: np.ndarray) -> float:
    """Return the mean nearest-reference distance for a current window."""

    return float(np.mean(distances))
```

This includes distances to reference embeddings that are not the nearest match
for a current embedding.

### Diagnostic Experiment

The contract-valid distance matrix was:

```text
[[0.1 0.9 1.2]
 [0.2 0.8 1.1]]
```

The nearest reference distance for each current embedding was:

```text
[0.1 0.2]
```

The observed score from the broken implementation was:

```text
0.7166666666666668
```

The expected mean nearest-reference score was:

```text
0.15000000000000002
```

Even though both current embeddings have a close reference match, the global
mean is increased by the distances to unrelated reference embeddings.

### Downstream Risk

A falsely elevated window score can make normal traffic appear shifted.

The same scoring mistake would also affect calibration scores, so it may distort
both the normal-score distribution and the threshold derived from that
distribution.

The final effect depends on the later calibration and threshold design, which
has not yet been implemented.

### Verifier Implication

A weak verifier that checks only whether the score is a finite scalar could
accept this implementation.

A test must distinguish the required row-wise nearest-reference aggregation
from plausible alternatives such as:

- the global mean of all distances;
- the maximum distance;
- the nearest distance across the entire matrix.

The verifier should check scoring behavior rather than require a particular
NumPy implementation.

This does not yet define the final Verifier V0. It records one observed system
failure and several plausible shortcut surfaces.

### Current Scope

This observation concerns only aggregation of a valid pairwise distance matrix
into one window score.

Calibration, threshold selection, debouncing, and final alert behavior have not
yet been implemented.

## Observation 5: Calibration Uses the Mean as the Threshold

### Simplified Calibration Contract

In this educational reproduction, an alert threshold is derived from held-out
normal window scores.

The simplified contract uses the requested quantile of the calibration-score
distribution. With the default argument, this is the 0.95 quantile.

This is our educational calibration design, not an official benchmark formula.

### Broken Implementation

The implementation intentionally returns the arithmetic mean and ignores the
`quantile` argument:

```python
def calibrate_threshold(
    calibration_scores: np.ndarray,
    quantile: float = 0.95,
) -> float:
    """Return an alert threshold from held-out normal scores."""

    return float(np.mean(calibration_scores))
```

The mean represents the center of the normal-score distribution rather than
its intended upper boundary.

### Diagnostic Experiment

The held-out normal scores were:

```text
[0.10, 0.12, 0.14, 0.16, 0.30]
```

The observed mean threshold was:

```text
0.164
```

Using NumPy's default quantile calculation, the expected 0.95 quantile was:

```text
0.27199999999999996
```

A high but normal score of `0.25` produced different decisions:

```text
0.25 > observed mean threshold: True
0.25 > expected quantile threshold: False
```

The broken threshold therefore classifies this normal diagnostic score as
shifted.

### Downstream Risk

A threshold near the center of the normal-score distribution may produce false
alerts for ordinary windows whose scores are above the mean.

This error can then propagate into the debouncing state, where repeated false
window classifications may eventually activate an alert.

Debouncing has not yet been implemented, so that propagation is currently a
design inference rather than a directly observed result.

### Verifier Implication

A weak verifier that checks only whether the threshold is finite and falls
inside the calibration-score range could accept the mean-based implementation.

The verifier must distinguish the requested quantile from plausible alternatives
such as the mean, median, maximum, or a hard-coded constant.

For small calibration samples, the quantile convention must also be defined
consistently. Numerical comparisons should allow ordinary floating-point
tolerance rather than require an exact decimal representation.

This does not yet define the final Verifier V0. It records one observed system
failure and potential verifier blind spots.

### Current Scope

This observation concerns threshold calibration from an already valid array of
normal window scores.

Debouncing and final alert behavior have not yet been implemented.

## Observation 6: Stable Windows Do Not Reset the Shift Streak

### Simplified Debouncing Contract

In this educational reproduction, an alert is activated after two consecutive
shifted windows and cleared after two consecutive stable windows.

A stable window must interrupt the shifted-window streak. Similarly, a shifted
window must interrupt the stable-window streak.

This is our simplified debouncing design, not an official benchmark parameter.

### Broken Implementation

The implementation intentionally fails to reset `shift_streak` when a stable
window is observed:

```python
class AlertDebouncer:
    """Track sustained shifted and stable window decisions."""

    def __init__(
        self,
        trigger_after: int = 2,
        clear_after: int = 2,
    ) -> None:
        self.trigger_after = trigger_after
        self.clear_after = clear_after
        self.shift_streak = 0
        self.stable_streak = 0
        self.alert = False

    def update(self, shifted: bool) -> bool:
        """Update and return the current alert state."""

        if shifted:
            self.shift_streak += 1
            self.stable_streak = 0

            if self.shift_streak >= self.trigger_after:
                self.alert = True
        else:
            self.stable_streak += 1

            if self.stable_streak >= self.clear_after:
                self.alert = False

        return self.alert
```

The stable branch is missing:

```python
self.shift_streak = 0
```

### Diagnostic Experiment

The first sequence was:

```text
True, True, False, False
```

The observed alert behavior appeared correct:

```text
step 1: alert=False
step 2: alert=True
step 3: alert=True
step 4: alert=False
```

However, after recovery, the internal shifted streak incorrectly remained at 2:

```text
step=4 shift_streak=2 stable_streak=2 alert=False
```

The second sequence was:

```text
True, False, True
```

The observed result was:

```text
step=1 shift_streak=1 alert=False
step=2 shift_streak=1 alert=False
step=3 shift_streak=2 alert=True
```

The stable window did not interrupt the shifted streak. Two separated shifted
windows were therefore treated as consecutive.

### Downstream Risk

The monitor may activate an alert after isolated shifted windows that are
separated by stable traffic.

Even after an alert is cleared, the stale shifted streak remains in memory.
A later shifted window may therefore reactivate the alert too early.

### Verifier Implication

A weak verifier that checks only a consecutive trigger sequence and a
consecutive recovery sequence could accept this implementation because the
visible alert outputs appear correct.

An interleaved sequence such as `shifted, stable, shifted` is needed to test
whether stable traffic actually resets the shifted streak.

The verifier may also need to test behavior after recovery, rather than checking
only the alert value at the moment it is cleared.

This does not yet define the final Verifier V0. It records one observed state
failure and potential verifier blind spots.

### Current Scope

This observation concerns only debouncing of already classified shifted and
stable windows.

The complete end-to-end monitor and command-line alert behavior have not yet
been implemented.

## Observation 7: End-to-End Failures Can Propagate or Be Masked

### Integrated Pipeline

The simplified components were connected into the following pipeline:

```text
raw window
-> normalization
-> pairwise distance
-> window score
-> threshold comparison
-> debouncer
-> alert
```

This integration does not add a new intentional bug. It exposes how the
existing component failures interact.

### Observed Calibration State

The broken pipeline produced these calibration scores:

```text
[0.81721246, 0.84710920, 0.81330371, 0.81308401]
```

The mean-based calibration implementation produced:

```text
threshold = 0.822677344083786
```

These values already depend on the broken global-mean window score and should
not be interpreted as a correct normal-score distribution.

### Zero-Vector Failure Propagation

Window 3 contained the injected zero embedding and produced:

```text
score=nan
finite=False
shifted=False
alert=False
```

The observed propagation was:

```text
zero embedding
-> unsafe normalization
-> NaN values
-> NaN window score
```

The comparison `NaN > threshold` evaluated to `False`, so a numerically invalid
window was silently classified as not shifted.

### Observed Alert Sequence

Windows 6 and 7 were classified as shifted:

```text
w06 shifted=True alert=False
w07 shifted=True alert=True
```

This activated the alert after two consecutive shifted decisions.

Window 9 was the first stable decision after the alert:

```text
w09 shifted=False alert=True
```

The alert correctly remained active because the simplified contract requires
two consecutive stable decisions before recovery.

### Masked and Unreached Failures

The pairwise transpose failure was masked by the broken global-mean scoring
function because a matrix and its transpose have the same global mean.

The broken scalar `cosine_distance` function was not exercised by this
end-to-end path because the monitor directly called the pairwise implementation.

The stale debouncer streak was also not fully exposed by this stream sequence.
It required the separate interleaved-sequence diagnostic.

### Verifier Implication

End-to-end checks are necessary for observing failure propagation, but they are
not sufficient for validating every component contract.

A meaningful verifier will need both:

- component-level evidence for local numerical, shape, and state properties;
- end-to-end evidence for stable traffic, shifted traffic, invalid numerical
  inputs, and alert behavior.

An apparently reasonable final alert sequence does not prove that all internal
components are correct.

### Current Scope

The intentionally broken monitoring pipeline is now connected and runnable.

Command-line behavior and the verifier have not yet been implemented.
