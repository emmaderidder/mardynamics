"""The simulation study.

Generates synthetic data only. No empirical estimate, empirical residual,
sample length, country or indicator enters any part of it, and the
appearance of a ``(5, 5)`` panel in the dimension grid is one point of a
pre-specified raster, not a reference to the empirical application.

The study has four branches and one sensitivity extension:

Design and protocol
    Has no simulated cell. Its outputs are the constructed populations,
    the change directions, the two signal measures, the realised
    departure sizes, the path verification and the seed registry, all
    produced before any replication runs.
Calibration
    Finite-sample calibration and numerical reliability: the null grid,
    the fixed-regressor wild-bootstrap comparison, canonical recovery
    under stability and the numerical multiple-start check.
Detection
    Detection, conditional mode-specific attribution and influence-path
    signatures.
Matrix restriction
    The value and the limits of the matrix restriction.
Sensitivity
    The population-and-direction sensitivity over three pre-specified
    population configurations.

Every cell is defined by its full design coordinates: dimensions, sample
length, persistence, population configuration, innovation design, path,
changing block, departure size, replication and bootstrap draw counts.
Each of them keys the cell's identity, its seed stream, its checkpoint
and its reuse key.

Modules
-------
``sim_config``
    Every run-wide constant.
``seeds``
    Reproducible generators from the single master seed and a draw's
    logical identity, addressed by index so that worker count cannot
    change a result.
``design``
    The design layer: the canonical population, the change directions in
    both their block and their embedded coordinates, the population
    information, score covariance, sandwich covariance and signal
    metric, the exact exponential map on the unit sphere, the designed
    paths and their verification, the theoretical signatures, the
    induced separable transition direction and the non-separable one.
``innovations``
    The four innovation designs, all with the same per-entry variance.
``dgp``
    Forward simulators: constant, designed-path and unrestricted, all
    with the same burn-in under the initial parameter and the initial
    innovation scale.
``estimators``
    Thin calls into the shared estimator with the simulation's numerical
    controls, the perturbed starts of the multiple-start check, and the
    numerical-reliability record.
``metrics``
    Canonical recovery losses, the conditional attribution events and the
    Monte Carlo summaries, conditional and unconditional.
``registry``
    The typed cell registry and the reuse keys.
``design_audit``
    Construction and verification of every designed object, before
    anything runs, plus the seed registry.
``runner``
    Threading, chunked checkpointing, worker dispatch and failure
    accounting.
``sim_io``
    Run directory, manifest, atomic checkpoints, tables, the output
    index and cross-run reuse.
``figures``
    The influence-path figure data and the signature agreement.
``study``
    The orchestrator of one run and the assembly of every table.
``designs``
    One module per branch that has executable cells.
"""
