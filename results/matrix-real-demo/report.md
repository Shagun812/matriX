# TiO2 measurement evidence report

> Evidence-first output. This report does not assign an H1-H4 fault diagnosis.

- Sample: `synthetic-test-fixture`
- Material: `tio2` (TiO2 anatase/rutile scope)
- Spec version: `1.0`
- Diagnosis: **UNRESOLVED**

## Checks

### C1: AVAILABLE

```json
{
  "status": "AVAILABLE",
  "measured_O_to_Ti": 2.003003003003003,
  "nominal_O_to_Ti": 2.0,
  "difference": 0.0030030030030028243,
  "standard_uncertainty": 0.015029447754215698,
  "uncertainty_type": "standard_uncertainty",
  "standardized_difference": 0.19980794052532597,
  "uncertainty_assumption": "independent supplied standard uncertainties; covariance not provided",
  "tolerance": null
}
```

### C2: AVAILABLE

```json
{
  "status": "AVAILABLE",
  "reason": null,
  "sample_displacement_deg": 2.007382593762568e-08,
  "shift_uncertainty_deg": 0.20563958260985965,
  "fit_quality": {
    "rmse_normalized_intensity": 0.0032349864812728176,
    "r_squared": 0.961330984781976,
    "degrees_of_freedom": 7,
    "jacobian_rank": 4,
    "shift_uncertainty_method": "local least-squares covariance from residuals; excludes instrument/systematic uncertainty"
  },
  "relative_crystalline_fraction": {
    "anatase": 0.7420099569471375,
    "rutile": 0.25799004305286255,
    "interpretation": "relative crystalline fraction; not absolute wt%"
  },
  "reference": {
    "path": "tests/fixtures/real_data/xrd_reference.csv",
    "sha256": "20632448c4baf009f46bd6db3a87f8d239773db1459e53845677be79f9e25a88",
    "citation": "Synthetic test fixture (not ICDD)",
    "source": "tests/fixtures/real_data/xrd_reference.csv",
    "version": "fixture-v1"
  },
  "shift_interpretation": "global profile alignment; not by itself evidence of lattice or composition change"
}
```

### C3: AVAILABLE

```json
{
  "status": "AVAILABLE",
  "anatase": 0.7420099569471375,
  "rutile": 0.25799004305286255,
  "interpretation": "relative crystalline fraction; not absolute wt%",
  "fit_residual": 0.0032349864812728176,
  "sample_displacement_deg": 2.007382593762568e-08
}
```

### C4: AVAILABLE

```json
{
  "status": "AVAILABLE",
  "measurement_type": "synthetic test absorbance",
  "signal_type": "absorbance",
  "x_unit": "eV",
  "energy_conversion": "x is photon energy in eV",
  "alternate_method": "linear baseline fitted over caller-supplied alternate_baseline_window_ev, clipped at zero before Tauc transform",
  "defect_tail_model": "not specified; no physical defect-tail model is fitted",
  "alternate_baseline_window_ev": [
    1.0,
    1.75
  ],
  "alternate_baseline_fit": {
    "slope": 0.008000000000000004,
    "intercept": 0.0019999999999999953
  },
  "models": {
    "anatase": {
      "transition": "indirect_allowed",
      "exponent": 0.5,
      "primary": {
        "band_gap_ev": 1.9999999999808875,
        "unconstrained_intercept_ev": 1.9999999999808875,
        "rmse": 6.174651280245475e-11,
        "r_squared": 1.0,
        "n_points": 6,
        "fitting_window_ev": [
          2.25,
          3.5
        ],
        "exponent": 0.5,
        "status": "AVAILABLE"
      },
      "alternate_baseline": {
        "band_gap_ev": 2.0983426864120296,
        "unconstrained_intercept_ev": 2.0983426864120296,
        "rmse": 0.016654832562287877,
        "r_squared": 0.9986347987401601,
        "n_points": 6,
        "fitting_window_ev": [
          2.25,
          3.5
        ],
        "exponent": 0.5,
        "status": "AVAILABLE"
      },
      "delta_band_gap_ev": 0.09834268643114208
    },
    "rutile": {
      "transition": "direct_allowed",
      "exponent": 2.0,
      "primary": {
        "band_gap_ev": 2.4836223506955086,
        "unconstrained_intercept_ev": 2.4836223506955086,
        "rmse": 0.8028050732069237,
        "r_squared": 0.802009415392745,
        "n_points": 6,
        "fitting_window_ev": [
          2.25,
          3.5
        ],
        "exponent": 2.0,
        "status": "AVAILABLE"
      },
      "alternate_baseline": {
        "band_gap_ev": 2.490770206091364,
        "unconstrained_intercept_ev": 2.490770206091364,
        "rmse": 0.7516790625799774,
        "r_squared": 0.7912208356986392,
        "n_points": 6,
        "fitting_window_ev": [
          2.25,
          3.5
        ],
        "exponent": 2.0,
        "status": "AVAILABLE"
      },
      "delta_band_gap_ev": 0.007147855395855274
    }
  }
}
```

### C5: AVAILABLE

```json
{
  "status": "AVAILABLE",
  "measurements_compared": [
    "composition_O_to_Ti_delta"
  ],
  "deltas": {
    "composition_O_to_Ti_delta": -0.008991026955098569
  },
  "interpretation": "differences between supplied independent measurements; no repeat was simulated"
}
```

## Limitations

- No real-data-calibrated likelihood/prior is available; H1-H4 diagnosis is not made.
- Synthetic simulator likelihood tables and priors are not loaded or reused.
- C2 and C3 require a user-supplied reference profile with provenance; no reference profiles are bundled.
- C1 reports deviation and propagated supplied uncertainty without an invented tolerance.
- C3 fractions are relative crystalline fractions, not absolute weight fractions.
- C4 is an evidence-level Tauc fit; model selection and real-data validation are not established.
- C2 uncertainty is residual-based fit covariance and excludes instrumental/systematic uncertainty.
