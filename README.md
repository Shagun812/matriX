# MatriX

**A cost-aware investigator for diagnosing conflicting materials-characterization evidence.**

> **OSI Hackathon · 21–22 October 2026 · submission deadline: Thu 22 Oct, 7:00 PM CT (UTC−5)**
> **Status:** prototype in development. No performance results are claimed until the frozen benchmark has been run.
> **Additional track:** [Valency MCP Bond](https://docs.valency.io/) (literature-search MCP server), used for constant sourcing and prior-art verification. See Section 17.1.

MaterialFalsify investigates why *simulated* materials-characterization evidence disagrees. It starts from a coarse screening measurement, picks from a fixed catalogue of executable refined checks, observes numerical outputs, updates competing explanations, and either reports a supported diagnosis or abstains when the available checks cannot separate the causes.

**This is not a materials-science chatbot.** The goal is to test whether adaptive, cost-aware diagnosis beats simpler strategies on a transparent, reproducible benchmark.

## At a glance

- **Domain:** simulated powder XRD, composition, and optical-absorption evidence.
- **System:** anatase/rutile TiO₂ polymorphs.
- **Method under test:** LLM-derived hypothesis prior + deterministic Bayesian planner.
- **Arms:** full-check reference, random, cheapest-first, greedy information gain (two priors), LLM-only, hybrid.
- **Key control:** greedy IG with a non-LLM learned prior, trained on the same agent-visible features as the LLM and on far more simulated data, so it is a strong comparator.
- **Limitation:** results describe this simulator and fault catalogue, not real laboratory samples.

---

## 1. Research question

> On a controlled, simulated benchmark, does combining an LLM-derived prior with a deterministic Bayesian planner improve the accuracy–cost trade-off and the handling of unresolvable cases, compared with simpler diagnostic policies?

This is a hypothesis, not a result. Planned comparisons (the only ones tested, to limit multiplicity):

| # | Comparison | Question |
|---|---|---|
| P1 | B3 vs B1 | Does a planner beat random selection? |
| P2 | B3b vs B3 | Does *any* informative prior help? |
| P3 | **B5 vs B3b** | Does the **LLM** prior add value beyond a feature-matched learned prior? |
| P4 | B5 vs B4 | Does deterministic planning beat an LLM choosing checks itself? |

**Primary endpoint:** area under the accuracy–cost curve (Section 8.2). **Co-primary descriptive endpoint:** abstention F1 on unresolved cases at the frozen operating point, because the research question concerns unresolvable cases too. Negative or inconclusive outcomes will be reported as such.

### Investigation loop

1. Inspect the screening evidence and the deterministic conflict summary.
2. Form a prior over competing hypotheses.
3. Select one permitted check.
4. Execute it and record the numerical result.
5. Update the posterior.
6. Stop with `CONFIRMED`, `UNRESOLVED`, or `BUDGET_EXCEEDED`.

---

## 2. Scope

**In scope**

- One system: anatase/rutile TiO₂.
- XRD, composition, optical absorption, plus structured literature metadata (reference phase, composition, band gap). **No literature search by the agent at runtime.** Valency MCP Bond is a builder-side tool only (Section 17.1) and is not part of any arm's information set.
- Five deterministic or seeded checks with normalized costs.
- Seeded **single-fault** cases with hidden labels.
- Baselines, a replayable trace, and a compact demo.

**Out of scope**

DFT reruns, real laboratory data or control, autonomous synthesis, unrestricted RAG, automatic repair, multi-agent systems, multi-material generalization, multi-fault cases (until single-fault results are complete), and any claim of real-world diagnostic validity.

**Feasibility rule:** a smaller working benchmark beats a larger unvalidated design.

---

## 3. Prior art and positioning

Cost-aware sequential diagnosis, LLM-driven XRD analysis, and programmatic physical checks already have prior art. MaterialFalsify does not claim to invent them.

| Work | Overlap | Intended distinction |
|---|---|---|
| [SDBench / Sequential Diagnosis with Language Models](https://arxiv.org/abs/2506.22405) (Nori et al., Microsoft AI) | Sequential test selection, costs, and stopping in clinical diagnosis | Materials-characterization cases with executable simulated checks and a computed unresolved label |
| [AutoXRD / XRDBench](https://arxiv.org/abs/2609.00070) | LLM-assisted powder-XRD analysis with deterministic checks | Cross-modal diagnosis (XRD, composition, optical) with explicit check costs and unresolvable cases |
| [MatPCR](https://arxiv.org/abs/2609.12181) | Programmatic physical oracles for materials reasoning | Evaluates sequential diagnostic *decisions*, not only physical consistency of an output |
| A-Lab characterization work (A. Jain, MRS Fall 2025 talk) | Ranking phase interpretations; detecting when no good interpretation exists | Closest on abstention; not a cost-aware benchmark |

**Not used:** SciPhys OS appeared in early planning. The project was not verified, so it is neither cited as prior art nor a dependency.

**Novelty statement (to be earned, not assumed).** *Unlike SDBench, which diagnoses clinical cases, this benchmark diagnoses disagreements among XRD, composition, and optical evidence using executable checks whose outcomes and unresolvable cases derive from a physical simulator.* No direct prior art for this exact combination was found in the sources examined. That is not proof that none exists.

Before submission, re-run the prior-art check with Valency MCP Bond (Section 17.1) and update this table and the novelty statement with anything found. Do not claim a Bond-assisted search was done until it has been.

The contribution holds only if cases are valid, comparisons are fair, and results are informative. This README specifies a planned benchmark and protocol. It does not establish that the simulator is scientifically valid or that the method is novel.

---

## 4. Simulator, faults, and conflict

### 4.0 Locked design decisions (fix before writing `sim.py`)

Record these as comments at the top of `sim.py`. Changing them later means rebuilding the oracle, checks, and baselines.

| # | Decision |
|---|---|
| D1 | **The initial evidence is a coarse screen.** Screen measurements use a degraded view of the same simulated sample: a low-resolution, higher-noise XRD pattern, a single higher-noise composition measurement, and one standard-baseline optical fit. Screen noise is `k_screen` times the refined noise (default `k_screen = 3`, set in `sim_constants.yaml`). **Checks C1–C4 are refined analyses** (full-resolution pattern, replicate-averaged composition, alternate optical baseline). A check that returns the same number as a screen test is a bug. |
| D2 | **C5 repeat model.** Sample-intrinsic faults (H1, H2) persist on repeat. Artefact faults are partly or fully re-drawn: the H3 offset is re-drawn near zero (sample re-mounted), and the H4 distortion is re-drawn at reduced magnitude (re-processed). Repeat noise is lower than refined noise. C5 therefore separates artefacts from intrinsic faults, at cost 10. |
| D3 | **Oracle information set.** The oracle outcome `y` is the three binary screen-test flags plus the five 3-bin check scores: 2³ × 3⁵ = 1,944 cells (Section 6.2). |
| D4 | **Timing before scaling.** In hour 1, time one full simulation of all five checks. If it exceeds ~10 ms per case, vectorize or use the fallback in Section 6.2 before generating anything. |

### 4.1 Faults and cross-channel coupling

The simulator is *physically motivated*, not a validated digital twin. Each fault perturbs several channels, and the shared effects make cases confusable.

| ID | Fault | XRD | Composition | Optical | Intended confusion |
|---|---|---|---|---|---|
| H1 | Secondary/wrong phase | Added reflections from the other polymorph, scaled by mixture fraction | none | Gap shifts toward the other polymorph in proportion to fraction | H1 vs H4 (both move the apparent gap) |
| H2 | Composition discrepancy | Small lattice-parameter shift (phenomenological) | Measured composition offset | Small defect-related gap change (phenomenological) | H2 vs H3 (both shift peak positions) |
| H3 | XRD instrumental artefact | Constant 2θ zero / displacement offset on all peaks | none | none | H3 vs H2 |
| H4 | Optical-processing artefact | none | none | Baseline or defect-tail distortion that shifts the fitted Tauc gap | H4 vs H1 |

Rules:

- Every constant (lattice parameters, reference gaps, peak lists, resolution, noise levels, `k_screen`) lives in `sim_constants.yaml` with a source. No unreferenced hardcoded numbers. Sources may be found with Valency MCP Bond (Section 17.1). Each constant records the paper identifier and the date it was retrieved, and is checked by a human against the paper itself.
- H2's lattice shift and optical change are **phenomenological perturbations**, not derived physics. Label them that way in the write-up.
- Remove any fault that cannot be modelled defensibly.
- **Single faults only** for now, so ground truth is unambiguous.

### 4.2 Screen tests and conflict definition

The agent-visible conflict is computed deterministically at case creation from three screen tests on the coarse measurements:

| Test | Definition |
|---|---|
| T1 composition | Normalized deviation of the screen composition vs nominal exceeds tolerance |
| T2 structure | RMS 2θ residual or fraction of unmatched reference reflections (coarse pattern) exceeds tolerance |
| T3 optical | Screen-fitted gap differs from the literature gap by more than δ |

The conflict is the vector of test values (coarse magnitudes) and pass/fail flags. **A generated case is kept only if at least one test fails.** Tolerances are fixed in the config before generation. Store the continuous pre-threshold magnitudes as well as the flags.

### 4.3 Seed streams

Three independent seed streams, never shared:

1. **case stream:** generates benchmark cases;
2. **dev-likelihood stream:** planner likelihoods and the B3b prior training set (Section 7);
3. **oracle stream:** identifiability tables and labels (Section 6).

Each case stores seed, simulator version, fault, magnitude, noise level, evidence, permitted checks, hidden configuration, resolvability label, the label-generation rule version, and the frozen oracle/binning configuration identifier.

---

## 5. Diagnostic checks

Checks are bounded and independently unit-tested. C1–C4 are deterministic given their input data. C5 is reproducible under its explicit repeat seed but generates a fresh noise realization under the D2 repeat model, so it is stochastic at the case level and reproducible at the implementation level.

| ID | Check | Output | Relative cost |
|---|---|---|---:|
| C1 | Composition consistency (replicate-averaged) | **Primary score:** normalized absolute composition deviation; also return tolerance flag | 0.5 |
| C2 | XRD zero-shift fit (full resolution) | **Primary score:** signed fitted 2θ offset; also return fit residual | 1 |
| C3 | Two-phase fit (NNLS) | **Primary score:** fitted secondary-phase fraction; also return phase fractions and residual | 2 |
| C4 | Alternate optical baseline | **Primary score:** signed change in fitted optical gap | 2 |
| C5 | Higher-resolution repeat (D2 model) | **Primary score:** change in conflict score; also return repeat measurements | 10 |

Costs are **normalized decision costs**, not laboratory prices. Runtime is logged separately. Run a cost-sensitivity analysis (Section 8.4).

**C5 conflict score.** Calculate it from the continuous, pre-threshold deviation magnitudes for T1–T3: normalize each by its frozen tolerance, cap each at a predeclared maximum of 3, then take their arithmetic mean. The C5 primary score is `repeat_conflict_score − original_conflict_score`. Negative values indicate reduced conflict on the repeat, which is the signature of an artefact under D2. Freeze this definition and all tolerances before generating test cases.

The agent receives each check's description, cost, valid arguments, and output schema. It does **not** receive hidden labels, injection parameters, or any table stating which check separates which hypothesis.

---

## 6. Benchmark design and ground truth

### 6.1 Cells and splits

Each case falls in a cell defined by **magnitude band** (B1 weakest … B5 strongest) and **noise level** (σ1 < σ2 < σ3).

| Split | Cells | Purpose |
|---|---|---|
| dev | bands {B1, B2, B4, B5} × noise {σ1, σ2} | Planner likelihoods, B3b training, prompt tuning, threshold freezing |
| test, in-range | same cells as dev, fresh case seeds | Standard performance |
| test, held-out | band B3 (all σ) **and** noise σ3 (all bands) | Interpolation and extrapolation to unseen conditions |

**Targets:** dev 40 cases, test 60 (30 in-range, 30 held-out). **Floor if time runs short:** dev 24, test 40. Claims on this test size are exploratory. Report in-range and held-out results **separately**, because dev-estimated likelihoods will be miscalibrated outside their range.

The 40 dev *cases* are used for prompt tuning, threshold freezing, and the kill gate. Planner likelihoods and the B3b prior use many more simulations from the dev-likelihood stream over the same dev cells.

### 6.2 Identifiability rule (resolvable vs unresolved)

`UNRESOLVED` is a decision target, not a physical fault. Labelling is **case-level and independent of any agent's policy**.

**Oracle tables.** For each noise level σ, use the **oracle stream** to simulate N cases per hypothesis, drawing fault magnitude from the predeclared magnitude distribution and applying the case-acceptance rule (at least one screen test fails). Default N = 50,000 per hypothesis per σ if one simulation takes ≤ 2 ms. Per D4, **fallback** if simulation is slower: N = 10,000 with 2 bins per check score (2³ × 2⁵ = 256 cells).

**Outcome y.** Each check contributes exactly **one predeclared scalar primary score** (Section 5). Auxiliary fields stay available to the agent but are excluded from this label statistic. Per D3:

```
y = (flag_T1, flag_T2, flag_T3, bin(C1), bin(C2), bin(C3), bin(C4), bin(C5))
    2^3 x 3^5 = 1,944 possible values
```

For each check score, set two bin edges at its 1/3 and 2/3 quantiles using oracle-stream simulations only, pooling the four hypotheses and magnitudes at that noise level. Freeze the edges. Estimate the joint likelihood with add-one (Laplace) smoothing:

```
L_h(y | sigma) = integral of P(y | h, m, sigma, accepted) p(m | accepted) dm
```

**Identifiability.** For a case c with true fault h*, true magnitude m_c, and noise sigma_c:

```
I(c) = E over y ~ P(y | h*, m_c, sigma_c, accepted)  of  [ L_h*(y | sigma_c) / sum over h of L_h(y | sigma_c) ]

resolvable   if I(c) >= tau_id     (default tau_id = 0.8)
UNRESOLVED   otherwise
```

The expectation is estimated from 500 fresh oracle-stream draws at the case's fixed fault magnitude and noise level, **rejection-sampled to satisfy the same acceptance rule** (log the acceptance rate). The four hypotheses get a uniform prior. If a score is constant or has tied quantiles, use deterministic tie handling specified in code and log the effective bin edges. Use the same frozen score definitions and bin edges for every case.

**Interpretation.** I(c) estimates the expected posterior mass on the true fault under this benchmark's binned observation model, assuming all five checks are run and the noise level is known. `UNRESOLVED` means the case falls below the predeclared threshold under that model. It does **not** show that the fault is fundamentally unidentifiable in real materials science.

**Information-set note.** Agents also see the coarse screen magnitudes, which the oracle bins only as pass/fail flags. The labels are therefore conservative relative to what agents can use. **Diagnostic:** on cases labelled `UNRESOLVED`, report the accuracy of B3b after all five checks (the strongest information set any arm has). If it is high on many of these cases, the label is too conservative, and the write-up must say so.

**Requirements**

- Fix tau_id, N, bins, and smoothing **before** the test run. Do not tune them on test.
- Report results at tau_id ∈ {0.7, 0.8, 0.9} as a sensitivity check.
- **Prevalence target:** aim for 20–40% unresolved in dev as a design check. If dev falls outside 10–50%, adjust the magnitude sampling range on dev only, document the change, and regenerate dev. Do not alter the test distribution to force a desired rate; report the naturally resulting test prevalence.
- **Sanity gate:** inspect 10 dev cases (five resolvable, five unresolved, if available) before building agents. Check that labels follow the stated model and threshold, and document the review. If the mix is not available, report the actual counts rather than silently relabelling.

**Limitation.** I(c) conditions on a known noise level, compresses each check to one scalar score, and uses a finite Monte Carlo estimate. It is a benchmark-specific operational label, not a theorem about real-world identifiability. Report the 500-draw standard error for cases near tau_id.

### 6.3 Cost reference

- **Non-adaptive reference:** for each resolvable case, the cheapest subset S of the five checks whose expected posterior on h*, computed with the oracle tables marginalized to S (and the flags), reaches tau_id. Enumerating 2^5 = 32 subsets is cheap.
- This is **not** the adaptive optimum. An adaptive policy can beat it, so regret may be negative, and we report it with that caveat.
- If time permits, estimate an adaptive reference by dynamic programming over discretized outcomes on a small case subset.

### 6.4 Kill gate (before any LLM work)

Verify on dev that:

- initial evidence does not trivially reveal every label (a simple classifier on the screen conflict vector does not reach ~100%);
- some cases need more than one check;
- **every check is selected by the planner on at least some cases.** A check that is never picked is redundant (D1) or has a flat likelihood (D2);
- **C5's likelihood tables differ across hypotheses:** the conflict-score change separates H3/H4 from H1/H2;
- some cases are genuinely confusable (H1/H4, H2/H3);
- costs change the preferred policy on some cases;
- simple baselines do not solve everything;
- the sanity gate in 6.2 passes.

If any condition fails, treat it as a generator bug and fix the generator first. Typical fix: weaker, more confusable faults, or a screen that is genuinely coarser than the refined checks, not more features.

---

## 7. Methods and arms

| Arm | Strategy | Question answered |
|---|---|---|
| B0 | Run all checks, then MAP diagnosis under a uniform prior (full-check reference) | What does a run-all policy cost and how does it perform? |
| B1 | Random valid check selection | Does structure help? |
| B2 | Cheapest-first selection | Is cost ordering alone enough? |
| B3 | Greedy IG per cost, uniform prior | Does a classical planner suffice? |
| **B3b** | Greedy IG per cost, **learned prior** (logistic regression on the same encoded agent-visible features as B5) | Is any informative prior enough? |
| B4 | LLM-only: LLM chooses checks and reports a distribution after each (required arm) | What happens without deterministic planning? |
| B5 | **Hybrid:** LLM prior + deterministic planner | Does the LLM prior add value? |

B0 is a **full-check reference**, not a theoretical ceiling. Imperfect likelihoods or a different decision rule could do better.

**B1 and B2** use a uniform prior and the same planner posterior update **for stopping only**. Check *selection* is random or cheapest-first.

### Planner model

- The planner's check likelihoods are the **joint binned table** over check outcomes, estimated from the dev-likelihood stream, with unobserved checks marginalized out. It does **not** assume the checks are independent given the hypothesis, because several are computed from the same data.
- Estimate from at least 20,000 simulated dev-cell cases per hypothesis, marginalized over the dev magnitude and noise distribution (agents do not know the hidden magnitude or noise). Bin edges come from the dev stream, not the oracle stream.
- **Known approximation.** The prior carries the screen evidence, and check likelihoods are unconditional on it. Because checks refine the same underlying sample, residual dependence can double-count evidence. Report posterior calibration on dev (reliability of the top posterior) for B3 and B3b, and flag any marked overconfidence.

### Hybrid investigator

1. The LLM sees the **agent-visible evidence summary**: the numeric screen conflict vector, key peak and gap values, and structured literature metadata. It returns a validated probability distribution over H1–H4. Sample k = 3 times and average; log per-sample distributions and dispersion.
2. B3b receives the exact same information through a deterministic numeric feature encoder (`features.py`): the same conflict values, peak/gap values, and literature metadata, with categorical fields one-hot encoded and missing values handled by a documented fixed rule.
3. **Training data for B3b.** Train its logistic regression on at least 5,000 simulated cases from the dev-likelihood stream over the dev cells, not on the 40 dev cases. Do not compare B5 with a B3b trained only on the conflict vector or on 40 cases. If the LLM prompt includes few-shot examples, draw them from dev cases only and disclose the count. This favors the control, and that is intended.
4. Mix the LLM prior with a small uniform component (ε = 0.05).
5. The planner updates the posterior with the joint table above.
6. Choose the legal check with the highest expected information gain per cost, within budget.
7. Run the trajectory until a stop condition (below). Apply the τ-sweep by truncation (Section 8.2).

**Stop conditions for one trajectory:** all checks used; budget spent (`BUDGET_EXCEEDED`); or every remaining check's expected gain per cost is below ε_gain.

**Decision at a given τ_stop:** `CONFIRMED` at the first step where the top posterior ≥ τ_stop; if none, `UNRESOLVED` (or `BUDGET_EXCEEDED` if the budget caused the stop).

### B4 (LLM-only)

The LLM sees the same screen evidence and, after each check, the output. It chooses the next check from the catalogue (or requests to stop) and reports a probability distribution over H1–H4 after every step. It receives no likelihood tables. Its trajectory is run once; each τ_stop is applied to its **self-reported top probability** by truncation, so B4 also yields a curve. Record that its probabilities are self-reported and uncalibrated.

The LLM cannot invent checks, alter outputs, see hidden labels, or run code. A malformed output triggers a documented fallback (neutral prior for B5; the previous distribution for B4), logged.

### Leakage prevention

- The evidence summary contains **no** fault names, generator fields, or case IDs encoding the family. The feature encoder uses only agent-visible fields.
- `tests/test_no_leak.py` scans every agent-visible string for forbidden tokens and fails the build on a hit.
- Case files separate `agent_visible` and `hidden` sections.

### Fair comparison

- Identical screen evidence, legal checks, and budget across arms.
- B3, B3b, and B5 use the **same** planner tables and the same encoded information. They differ only in how the prior is produced.
- B4 sees the same evidence and check outputs and uses the same catalogue and budget, but no likelihood tables.
- Thresholds (τ_stop grid, ε_gain) are chosen on dev and **frozen** before the test run.
- Cache all LLM responses; record model, provider, prompt version, and parameters; report API failures and fallbacks.

---

## 8. Evaluation

### 8.1 Metric definitions

| Metric | Definition |
|---|---|
| Resolvable accuracy | Fraction of resolvable cases with `CONFIRMED` and the correct fault |
| Abstention P / R / F1 | Positive class = unresolved case; predicted positive = `UNRESOLVED` status |
| False-diagnosis rate | Fraction of all cases ending `CONFIRMED` with a wrong fault **or** on an unresolved case |
| Premature commitment | `CONFIRMED` and wrong while at least one legal in-budget check remained unrun |
| Mean cost / checks | Total normalized cost and count at decision |
| Cost regret | Cost minus non-adaptive reference cost (Section 6.3); may be negative |
| Reliability | Schema-valid output rate, budget exhaustion, fallback events, trace completeness |

### 8.2 Accuracy–cost curve and primary endpoint

**Curve construction.** Run each arm once per case to produce a full trajectory. For each τ_stop in {0.6, 0.7, 0.8, 0.9, 0.95}, derive that arm's decision by truncating the trajectory (Section 7). Plot resolvable accuracy against mean normalized cost. The grid is fixed and not tuned on test; the test curve is computed on the same grid. B0 contributes a single point.

**Primary endpoint.** For each comparison in P1–P4, compute the area under each arm's accuracy–cost curve by the trapezoid rule over the **common cost interval**: the overlap of the two compared arms' observed mean-cost ranges. Do not extrapolate. Fix this rule in code before the test run. If the overlap is degenerate, report fixed-operating-point metrics only and say so. Higher AUC is better.

Because the test set is small, also report the underlying curves and the fixed-operating-point metrics. AUC is a summary, not a substitute for inspecting false diagnoses and abstention.

**Operating point.** Choose it on dev (for example the cheapest τ_stop reaching ≥ 90% resolvable accuracy) and freeze it. Abstention F1 is reported at this point.

### 8.3 Statistics and multiplicity

- For each of P1–P4, report the paired difference in primary AUC, a paired bootstrap 95% CI, and an effect size. Resample test cases jointly across arms.
- **P-values:** use a paired permutation test (swap the two arms' per-case outcomes at every τ_stop, recompute the AUC difference; 10,000 permutations; two-sided). Apply the **Holm step-down correction** across the four tests. Report raw and adjusted p-values, and treat adjusted values as descriptive given the exploratory test size.
- At the frozen operating point, report paired differences in resolvable accuracy, false-diagnosis rate, abstention F1, and mean cost as secondary outcomes with 95% intervals.
- Use McNemar's test only for paired binary correctness on the same eligible cases; it is not the test for AUC or mean-cost differences.
- Report in-range and held-out strata separately, alongside pooled results. Do not treat multiple thresholds, metrics, or strata as extra confirmatory tests.
- With 60 test cases (or the floor of 40), power for modest effects is low. Emphasize estimates and intervals over binary significance claims.

### 8.4 Sensitivity

Vary C5's cost (5 / 10 / 20) and the budget (5 / full), and report whether the preferred policy changes. Vary tau_id as in 6.2. Sensitivity runs are secondary and must not be used to select or retune the primary result.

### 8.5 Results

Populate only after the frozen evaluation. No anticipated or illustrative numbers.

| Strategy | Resolvable acc. | Mean cost | Abstention F1 | False-diagnosis | AUC | In-range / held-out |
|---|---:|---:|---:|---:|---:|---|
| B0 Full-check | TBD | TBD | TBD | TBD | n/a (single point) | |
| B1 Random | TBD | TBD | TBD | TBD | TBD | |
| B2 Cheapest-first | TBD | TBD | TBD | TBD | TBD | |
| B3 IG, uniform prior | TBD | TBD | TBD | TBD | TBD | |
| B3b IG, learned prior | TBD | TBD | TBD | TBD | TBD | |
| B4 LLM-only | TBD | TBD | TBD | TBD | TBD | |
| B5 Hybrid | TBD | TBD | TBD | TBD | TBD | |

**Interpretation rules**

- If B3b matches or beats B5 within uncertainty, report that the benchmark provides no clear evidence that the LLM prior adds value beyond the feature-matched learned prior.
- If B3b is not completed, P3 is unavailable. B5 vs B3 may be reported descriptively, but any difference **must not** be attributed to the LLM prior.
- If simple baselines solve every case, the benchmark is too easy. Report it as a limitation.
- If B4's self-reported probabilities are poorly calibrated, say so; its curve reflects stated confidence, not true accuracy.

---

## 9. Held-out simulation consistency

Each case's simulated pattern includes a **held-out window** (a 2θ range and an optical energy range) excluded from all checks and from the agent-visible evidence. After a `CONFIRMED` diagnosis, fit the diagnosed fault's parameters on the used data, predict the held-out windows under a fresh noise draw, and report whether the residual is within tolerance and smaller than under the competing hypotheses.

Call this **held-out simulation consistency**, not independent experimental verification. It cannot compensate for an invalid simulator.

---

## 10. Architecture

```text
Screen evidence + conflict vector (coarse)
          |
          v
Prior over H1–H4: LLM (B5) or feature-matched logistic regression (B3b)
          |
          v
Deterministic Bayesian planner (joint binned dev-estimated likelihoods)
          |
          |  choose legal check under budget
          v
Refined check (C1–C5) -> numerical result + provenance
          |
          +--> posterior update -> trajectory -> tau-truncation decision
                                         |
                CONFIRMED / UNRESOLVED / BUDGET_EXCEEDED
                      |
                      v
            Held-out simulation consistency
                      |
                      v
              Replayable JSON trace
```

**Stack:** Python 3.11+, NumPy, SciPy, pandas, Pydantic, pytest, matplotlib, scikit-learn (B3b), marimo (demo), a provider-neutral LLM wrapper with structured output and caching. LangGraph is optional and not recommended for this sprint. A notebook or CLI with clear plots is safer than a web app.

---

## 11. Repository structure and quickstart

```text
materialfalsify/
├── materialfalsify/
│   ├── sim.py              # screen + refined evidence simulation, fault injection (D1–D4 at top)
│   ├── sim_constants.yaml  # sourced constants, noise levels, k_screen
│   ├── checks.py           # C1–C5
│   ├── features.py         # shared agent-visible feature encoder (B3b and LLM summary)
│   ├── schemas.py          # cases, checks, traces
│   ├── oracle.py           # oracle tables, I(c), labels, cost reference
│   ├── likelihoods.py      # dev-only joint binned planner likelihoods
│   ├── priors.py           # B3b logistic-regression prior
│   ├── planner.py          # posterior, IG, stopping, trajectory truncation
│   ├── agents.py           # B0–B5
│   └── evaluation.py       # metrics, AUC, bootstrap, permutation test, Holm, figures
├── cases/                  # dev.json, test.json
├── traces/
├── figures/
├── tests/                  # determinism, schemas, checks, oracle sanity, no-leak
├── demo.py                 # marimo notebook, offline replay
├── pyproject.toml
├── LICENSE
└── README.md
```

This is the **target** structure, not a claim that the modules exist yet.

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy pandas pydantic pytest matplotlib scikit-learn marimo

python -m materialfalsify.sim time                              # D4: time one simulation
python -m materialfalsify.sim generate --split dev  --n 40 --seed 0
python -m materialfalsify.oracle build  --n 50000 --seed 1      # oracle stream
python -m materialfalsify.likelihoods fit --n 20000 --seed 3    # dev-likelihood stream
python -m materialfalsify.priors train --n 5000 --seed 4        # B3b, simulated dev cells
pytest -q                                                       # incl. determinism + no-leak
python -m materialfalsify.evaluation run --arms B0 B1 B2 B3 B3b --split dev
python -m materialfalsify.evaluation run --arms B4 B5 --split dev   # needs LLM key; cached
# freeze thresholds, then:
python -m materialfalsify.sim generate --split test --n 60 --seed 2
python -m materialfalsify.evaluation run --arms all --split test
marimo run demo.py
```

The commands show the intended interface. Adjust names to match the code.

---

## 12. Build plan with time boxes

H0 = your start time on Oct 21. **The deadline is fixed:** Oct 22, 7:00 PM CT. Convert to your local time now and **submit at least 2 hours early**.

| Hours | Work | Gate |
|---|---|---|
| 0–1 | Lock D1–D4 in `sim.py`; time one simulation | Decisions written; per-sim runtime known |
| 1–4 | `sim.py`, C1–C5, oracle tables, labels, B0–B3 | Kill gate (6.4) and sanity gate (6.2) pass |
| 4–10 | `features.py`, B3b on simulated cases, dev likelihoods, calibration check, threshold selection | Baselines reproducible; no-leak test green |
| 10–15 | LLM wrapper, B4, B5, caching | ≥ 95% schema-valid LLM outputs on dev |
| 15–19 | **Freeze** thresholds and config; one test run | Results table populated once |
| 19–24 | Figures, failure inspection, marimo replay | Demo runs offline from cached traces |
| 24–30 | Video, 1-page write-up, post, repo cleanup | Submitted ≥ 2 h before deadline |

### Tiers and cut order

- **Tier A (must):** simulator (D1–D4), checks, oracle labels, B0–B3, accuracy–cost plot.
- **Tier B (should):** B3b, B4, B5, abstention metrics, held-out consistency.
- **Tier C (stretch):** adaptive DP reference, repeated-sample LLM analysis, cost sensitivity.

**Cut order when behind:** (1) Tier C; (2) shrink dev/test to the floor sizes; (3) lower oracle N or use the 256-cell fallback; (4) reduce k from 3 to 1; (5) simplify the held-out check to XRD only; (6) drop C5 and its cost sensitivity **only if the kill gate shows it adds nothing**. **Never cut B3b without also dropping the LLM-benefit claim.**

---

## 13. Demo (target 3:30; the cap is 4:00)

1. **Conflict (30 s):** the screen evidence and the failed consistency tests.
2. **Prior and hypotheses (30 s):** LLM prior vs learned prior.
3. **Action (45 s):** candidate checks, costs, the chosen check, and its numeric output.
4. **Update (40 s):** posterior shift and why the agent continues, confirms, or abstains.
5. **Comparison (45 s):** accuracy–cost plot including B3b.
6. **Limits (20 s):** simulated data; what real validation would need.

Show one resolvable and one unresolved case. The centrepiece is the decision trace. If the API fails, replay a saved trace and label it a replay.

---

## 14. Team and review

| Role | Owns | Deliverable |
|---|---|---|
| Simulation and oracle | `sim.py`, `checks.py`, `oracle.py` | Validated cases, check API within ~3 h |
| Planner and agents | `planner.py`, `agents.py`, `features.py`, `priors.py`, `schemas.py` | B1–B5, structured LLM output |
| Evaluation and demo | `evaluation.py`, tests, `demo.py`, write-up | Fair comparisons, figures, trace, video |

Names: ____________________ (fill in)

**Domain review:** by hour 6, post the fault coupling table (4.1) and parameter ranges in the hackathon Slack and ask a materials-literate mentor to sanity-check them. Reviewer: ____________________. If no review is obtained, say so in the write-up.

Agree on schemas in the first hour and integrate continuously.

---

## 15. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Synthetic faults implausible | Sourced constants, constrained ranges, Slack review, label H2 as phenomenological |
| Checks repeat screen information | D1 (coarse screen, refined checks); kill gate requires every check to be selected sometimes |
| C5 cannot discriminate | D2 repeat model; kill gate checks that C5 likelihoods differ across hypotheses |
| Initial evidence leaks the answer | Kill gate; no-leak test; hidden/visible case split |
| `UNRESOLVED` labels circular | Case-level I(c) from an independent oracle stream (6.2) |
| Oracle and agent information sets differ | D3 plus the B3b-after-all-checks diagnostic (6.2) |
| Planner and oracle share tables | Separate seed streams and separate table builds |
| Hybrid gain is just "any prior" | Required B3b control trained on thousands of simulated cases; P3 |
| LLM prior noisy | k = 3 averaging; report dispersion |
| Likelihoods miscalibrated or double-counting | Dev calibration check; separate in-range / held-out reporting |
| Endpoint undefined for some arms | Trajectory truncation gives every arm a curve (8.2) |
| Cost reference overstates optimality | Called non-adaptive; negative regret allowed; optional DP |
| Costs arbitrary | Justification plus sensitivity analysis |
| Small test set / multiple comparisons | Paired intervals, effect sizes, permutation tests, Holm correction, "exploratory" wording |
| Simulation too slow | D4 timing in hour 1; fallback in 6.2 |
| API failure | Cached responses, offline replay |
| Scope creep | Cut order in Section 12 |
| Novelty overstated | Adjacent work stated; novelty treated as empirical |
| Retrieved literature is wrong or misread | Human check of every sourced constant against the paper; Bond is not used as ground truth; retrieval date logged |
| Bond setup or outage eats build time | Time-boxed to 30 min; Bond is outside the critical path and the benchmark does not depend on it |

---

## 16. Success criteria

- A documented, reproducible simulator with sourced constants and locked design decisions D1–D4.
- A benchmark with case-level, policy-independent labels, including unresolved cases.
- Executable checks with documented outputs and costs, each adding information beyond the screen.
- B0–B3 plus B3b implemented, and B4/B5 if feasible.
- A fair comparison with frozen thresholds and no test leakage.
- A replayable trace and a demo of a real decision.
- Honest limitations and no fabricated results.

A positive result is not required. A well-built benchmark showing that a simple policy matches the hybrid is still informative.

---

## 17. OSI submission

Per the official [agenda](https://llmhackathon.github.io/agenda/):

- A submission is a **social media post** containing a **video of up to 4 minutes** and a **write-up of up to 1 page**.
- Deadline: **Thu Oct 22, 7:00 PM CT** (5:00 PM PT; 00:00 UTC Oct 23; 2:00 AM CEST Oct 23).
- The [submission page](https://llmhackathon.github.io/submission/) showed 2025 dates when checked. Confirm current requirements in Slack. The 2026 judging criteria are not confirmed. Treat criteria from earlier editions as historical context, not verified 2026 rules.
- **India time conversion:** Thu Oct 22, 7:00 PM CDT (UTC−5) is **Fri Oct 23, 5:30 AM IST (UTC+5:30)**. Confirm the organizer's time zone and deadline in the event channel before relying on this.

### 17.1 Valency MCP Bond

[MCP Bond](https://docs.valency.io/) is Valency's MCP server. It connects an AI assistant to a daily-updated index of over 500 million research papers for search, retrieval, and analysis (per the Valency docs, which list 38 tools and setup guides for Claude, ChatGPT, VS Code and others). MaterialFalsify uses it **only on the builder side**:

| Use | Where | Rule |
|---|---|---|
| Source and cross-check constants (anatase/rutile lattice parameters, reference gaps, peak lists) | `sim_constants.yaml` (Section 4.1) | Record paper ID and retrieval date; verify by hand against the paper |
| Re-check prior art (sequential diagnosis, LLM-XRD, programmatic oracles) | Section 3, novelty statement | Add anything found; do not weaken the "not proof that none exists" caveat |
| Find literature for the H2 phenomenological perturbation ranges | Section 4.1 | Keep the "phenomenological, not derived physics" label |

**Not done by design:** the agent does not call Bond during diagnosis. Doing so would change every arm's information set and break the fair comparisons (P1–P4). A Bond-equipped agent arm would be a separate Tier C experiment and must not be mixed into the primary results.

**To confirm before submitting for this track**

- [ ] Prize track name, eligibility, and any extra requirements (stated in the hackathon Slack or the sponsor's page)
- [ ] Whether a separate submission or a particular disclosure is needed
- [ ] Whether the account sign-in is the same one used for the connector (per the Valency docs, use the account you claimed Valency with)

**Submission checklist**

- [ ] Public repo with setup instructions, tests, and a real `LICENSE`
- [ ] 1-page write-up: problem, approach, results, novelty, limitations
- [ ] Video ≤ 4:00 (target 3:30), public
- [ ] Social post linking the repo and write-up
- [ ] Statement of what was simulated vs measured and what is unvalidated
- [ ] Disclosure of reused code, data, pretrained models, and the number of few-shot examples (if any)
- [ ] Disclosure of how Valency MCP Bond was used (builder-side only) and which constants or citations came from it

---

## 18. References

1. Nori et al., *Sequential Diagnosis with Language Models* (SDBench, MAI-DxO), [arXiv:2506.22405](https://arxiv.org/abs/2506.22405).
2. *AutoXRD: Autonomous LLM Agents and Comprehensive Evaluation for Powder Diffraction Analysis* (XRDBench), [arXiv:2609.00070](https://arxiv.org/abs/2609.00070).
3. *Physics as the label for measuring and correcting materials reasoning in multimodal models* (MatPCR), [arXiv:2609.12181](https://arxiv.org/abs/2609.12181). Author list to be verified before formal citation.
4. A. Jain and O. Dartsi, talk on LLM-assisted interpretation of characterization data, MRS Fall Meeting 2025.
5. [Open Scientific Intelligence Hackathon](https://llmhackathon.github.io/).
6. [Valency MCP Bond documentation](https://docs.valency.io/).

Verify citations, versions, and licences before using them in any formal research claim.

## 19. Licence and attribution

Choose an OSI-approved licence (for example MIT or Apache-2.0), add the `LICENSE` file, and attribute reused code, data, and models. Do not describe the project as open source until the licence and redistribution rights are in place.