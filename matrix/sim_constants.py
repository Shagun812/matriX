"""Frozen simulator constants for the MatriX benchmark.

The companion ``sim_constants.yaml`` records the same values plus provenance
metadata.  Keep the two files synchronized whenever simulator constants change.

These are benchmark constants, not claims of a validated digital twin.
Changing them requires regenerating simulator-derived artifacts.
"""

from __future__ import annotations

SIMULATOR_VERSION = "1.0"

# Reference material/evidence values.
REFERENCE_PHASE = "anatase"
REFERENCE_COMPOSITION = 1.0
REFERENCE_GAP_EV = 3.20
REFERENCE_MAIN_PEAK_2THETA = 25.28

# Synthetic reference peak locations.
ANATASE_PEAKS = (
    25.28,
    37.80,
    48.05,
    53.89,
    55.06,
    62.69,
)

RUTILE_PEAKS = (
    27.45,
    36.08,
    39.20,
    41.23,
    44.05,
    54.32,
    56.64,
)

# Screen/refined relationship.
K_SCREEN = 3.0

# Refined measurement noise scales.
NOISE_LEVELS = {
    "sigma1": 0.010,
    "sigma2": 0.025,
    "sigma3": 0.050,
}

# Magnitude bands: B1 weakest ... B5 strongest.
MAGNITUDE_BANDS = {
    "B1": (0.10, 0.20),
    "B2": (0.20, 0.35),
    "B3": (0.35, 0.50),
    "B4": (0.50, 0.70),
    "B5": (0.70, 0.90),
}

# Frozen screen tolerances.
COMPOSITION_TOL = 0.05
STRUCTURE_RMS_TOL = 0.10
STRUCTURE_UNMATCHED_TOL = 0.20
OPTICAL_GAP_TOL = 0.08

# Fault-response coefficients.
#
# H1: secondary polymorph fraction is the sampled magnitude.
H1_RUTILE_GAP_EV = 3.00

# H2 is deliberately a SMALL composition perturbation.  The fault magnitude
# is dimensionless benchmark strength, not a literal composition fraction.
H2_COMPOSITION_SHIFT_PER_MAG = 0.05
H2_XRD_SHIFT_PER_MAG = 0.20
H2_OPTICAL_SHIFT_PER_MAG = 0.04

# H3: constant 2theta zero/displacement offset.
H3_OFFSET_PER_MAG = 0.35

# H4: optical baseline/defect-tail distortion.
H4_OPTICAL_SHIFT_PER_MAG = 0.20

# C5 repeat model.
H3_REPEAT_OFFSET_SCALE = 0.10
H4_REPEAT_SCALE = 0.35
REPEAT_NOISE_FACTOR = 0.50

# C3/C4 numerical safeguards.
MIN_PHASE_FRACTION = 0.0
MAX_PHASE_FRACTION = 1.0

# Oracle defaults.
TAU_ID = 0.80
ORACLE_BINS = 3
ORACLE_DRAWS = 500
