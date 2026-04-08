# CPT Test Pipeline Flow

## Overview

This document illustrates how the CPT test (separate Δm²₃₁ for neutrinos and antineutrinos) is implemented in the PyNu framework.

---

## Standard vs CPT Oscillation Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STANDARD OSCILLATION MODE                           │
│                         (Dm231 = Dm231_bar)                                 │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐
    │   XML Config │     │   PyNuFit    │     │  AtmosphericOsc      │
    │              │────▶│              │────▶│                      │
    │  Dm231=2.5e-3│     │ SetUpPhysics │     │  Single Propagation  │
    │              │     │    Tunes()   │     │  with Dm31           │
    └──────────────┘     └──────────────┘     └──────────┬───────────┘
                                                         │
                                                         ▼
                              ┌───────────────────────────────────────┐
                              │            nuSQuIDS                   │
                              │  ┌─────────────────────────────────┐  │
                              │  │  Propagate ν AND ν̄ together    │  │
                              │  │  with SAME Δm²₃₁ = 2.5×10⁻³    │  │
                              │  └─────────────────────────────────┘  │
                              └───────────────────────────────────────┘
                                                         │
                                                         ▼
                              ┌───────────────────────────────────────┐
                              │         Oscillation Weights           │
                              │    w[i] = P(νₐ → νᵦ) for all events  │
                              └───────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                            CPT TEST MODE                                    │
│                         (Dm231 ≠ Dm231_bar)                                 │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐
    │ run_cpt_real │     │   PyNuFit    │     │  AtmosphericOsc      │
    │    .py       │────▶│              │────▶│                      │
    │              │     │ physics_tunes│     │  *** CPT MODE ***    │
    │ Sets Dm231   │     │   [name]     │     │  Dual Propagation    │
    │ Sets Dm231_bar     │ .OscTunes    │     │                      │
    └──────────────┘     └──────────────┘     └──────────┬───────────┘
                                                         │
                         ┌───────────────────────────────┴───────────────────┐
                         │                                                   │
                         ▼                                                   ▼
          ┌─────────────────────────────┐             ┌─────────────────────────────┐
          │     nuSQuIDS (Run 1)        │             │     nuSQuIDS (Run 2)        │
          │  ┌───────────────────────┐  │             │  ┌───────────────────────┐  │
          │  │  Propagate with       │  │             │  │  Propagate with       │  │
          │  │  Δm²₃₁ = Dm231        │  │             │  │  Δm²₃₁ = Dm231_bar    │  │
          │  │  (for NEUTRINOS)      │  │             │  │  (for ANTINEUTRINOS)  │  │
          │  └───────────────────────┘  │             │  └───────────────────────┘  │
          └─────────────────────────────┘             └─────────────────────────────┘
                         │                                                   │
                         ▼                                                   ▼
                    w_nu[i]                                             w_nubar[i]
                         │                                                   │
                         └───────────────────┬───────────────────────────────┘
                                             │
                                             ▼
                              ┌───────────────────────────────────────┐
                              │         Combine by PDG Sign           │
                              │  ┌─────────────────────────────────┐  │
                              │  │  if PDG > 0:  w[i] = w_nu[i]    │  │
                              │  │  if PDG < 0:  w[i] = w_nubar[i] │  │
                              │  └─────────────────────────────────┘  │
                              └───────────────────────────────────────┘
                                             │
                                             ▼
                              ┌───────────────────────────────────────┐
                              │      Final Oscillation Weights        │
                              │   (ν with Dm231, ν̄ with Dm231_bar)   │
                              └───────────────────────────────────────┘
```

---

## Code Location Map

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        WHERE CPT PARAMETERS LIVE                            │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  run_cpt_real.py                                                            │
│  ════════════════                                                           │
│                                                                             │
│  for dm231 in dm231_grid:                                                   │
│      for dm231_bar in dm231_bar_grid:                                       │
│                                                                             │
│          # ◀══════ CPT PARAMETERS SET HERE ══════▶                         │
│          pt.OscillationTunes.Parameters["Dm231"] = dm231                    │
│          pt.OscillationTunes.Parameters["Dm231_bar"] = dm231_bar  ◀── NEW   │
│          pt.OscillationTunes.reset_cache()                                  │
│                                                                             │
│          pynufit.ApplyOscillations("Physics")  # Triggers GetOscillations() │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  pynu/PhysicsTunes/Oscillations/AtmOsc.py                                   │
│  ═════════════════════════════════════════                                  │
│                                                                             │
│  class AtmosphericOscillations(Oscillator):                                 │
│                                                                             │
│      def __init__(self, ...):                                               │
│          ...                                                                │
│          self.nuPDG = experiment.nuPDG  # ◀── Store PDG for CPT mode        │
│          self.Parameters["Dm231_bar"] = self.Parameters["Dm231"]  # ◀── NEW │
│                                                                             │
│      def GetOscillations(self):                                             │
│          dm31_nu = self.Parameters["Dm231"]                                 │
│          dm31_nubar = self.Parameters["Dm231_bar"]  # ◀── NEW               │
│                                                                             │
│          # ◀══════ CPT MODE CHECK ══════▶                                   │
│          cpt_mode = abs(dm31_nu - dm31_nubar) > 1e-10                       │
│                                                                             │
│          if not cpt_mode:                                                   │
│              # Standard: single propagation                                 │
│              return self._standard_propagation()                            │
│          else:                                                              │
│              # CPT: dual propagation  ◀── NEW LOGIC                         │
│              w_nu = self._single_propagation(dm31_nu)                       │
│              w_nubar = self._single_propagation(dm31_nubar)                 │
│              is_neutrino = self.nuPDG > 0                                   │
│              return np.where(is_neutrino, w_nu, w_nubar)                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  pynu/PhysicsTunes/Oscillations/Oscillations.py                             │
│  ═══════════════════════════════════════════════                            │
│                                                                             │
│  class Oscillator(Tune):                                                    │
│                                                                             │
│      self.Parameters = {                                                    │
│          "Sin2Theta12": 0,                                                  │
│          "Sin2Theta13": 0,                                                  │
│          "Sin2Theta23": 0,                                                  │
│          "Dm221": 0,                                                        │
│          "Dm231": 0,                                                        │
│          "Dm231_bar": 0,  # ◀══════ NEW PARAMETER ══════                    │
│          "dCP": 0,                                                          │
│          "Ordering": "normal",                                              │
│      }                                                                      │
│                                                                             │
│      def Dm231_bar(self, experiment, x):  # ◀══════ NEW METHOD ══════       │
│          self.Parameters["Dm231_bar"] = x                                   │
│          return self.GetOscillations()                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Flow: Single Grid Point Evaluation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  EVALUATING χ² AT POINT (Dm231=2.4e-3, Dm231_bar=2.6e-3)                   │
└─────────────────────────────────────────────────────────────────────────────┘

Step 1: Set Parameters
══════════════════════
    run_cpt_real.py
    │
    ├── pt.OscillationTunes.Parameters["Dm231"] = 2.4e-3
    ├── pt.OscillationTunes.Parameters["Dm231_bar"] = 2.6e-3
    └── pt.OscillationTunes.reset_cache()

Step 2: Trigger Oscillation Calculation
═══════════════════════════════════════
    pynufit.ApplyOscillations("Physics")
    │
    └── AtmosphericOscillations.GetOscillations()
        │
        ├── Check: |2.4e-3 - 2.6e-3| > 1e-10 ? YES → CPT MODE
        │
        ├── Run 1: _single_propagation(2.4e-3)
        │   │
        │   ├── Osc.Set_SquareMassDifference(2, 2.4e-3)
        │   ├── Osc.Set_initial_state(flux, flavor)
        │   ├── Osc.EvolveState()  ← nuSQuIDS Earth propagation
        │   └── w_nu = [EvalFlavor for each event]
        │
        ├── Run 2: _single_propagation(2.6e-3)
        │   │
        │   ├── Osc.Set_SquareMassDifference(2, 2.6e-3)
        │   ├── Osc.Set_initial_state(flux, flavor)
        │   ├── Osc.EvolveState()  ← nuSQuIDS Earth propagation
        │   └── w_nubar = [EvalFlavor for each event]
        │
        └── Combine:
            │
            │  Event PDG   │  Weight Used
            │──────────────│────────────────
            │  +14 (νμ)    │  w_nu[i]
            │  -14 (ν̄μ)   │  w_nubar[i]
            │  +12 (νe)    │  w_nu[i]
            │  -12 (ν̄e)   │  w_nubar[i]
            │
            └── weights = np.where(PDG > 0, w_nu, w_nubar)

Step 3: Bin Events & Compute Likelihood
═══════════════════════════════════════
    ├── Apply nuisance weights
    ├── Bin in (E_reco, cos_zenith, morphology)
    ├── Compute Barlow-Beeston χ²
    └── Minimize over 14 nuisance parameters (L-BFGS-B with jacobian)

Step 4: Store Result
════════════════════
    chi2_grid[i,j] = minimized_chi2
```

---

## File Modification Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FILES MODIFIED FOR CPT SUPPORT                           │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────────┐
                              │   NEW ENTRY POINT   │
                              │   run_cpt_real.py   │
                              │                     │
                              │ • 2D grid scan loop │
                              │ • Sets Dm231_bar    │
                              │ • Calls PyNuFit     │
                              └──────────┬──────────┘
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              │                          │                          │
              ▼                          ▼                          ▼
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│    UNCHANGED        │    │     MODIFIED        │    │     MODIFIED        │
│    PyNuFit.py       │    │    Oscillations.py  │    │     AtmOsc.py       │
│    (restored from   │    │                     │    │                     │
│     git e43d0ac)    │    │ + Dm231_bar in      │    │ + Dm231_bar init    │
│                     │    │   Parameters dict   │    │ + nuPDG storage     │
│ • Experiment setup  │    │ + Dm231_bar() method│    │ + CPT mode check    │
│ • Likelihood calc   │    │ + diff_Dm231_bar()  │    │ + Dual propagation  │
│ • Nuisance minim.   │    │                     │    │ + Weight combining  │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘

                    UNCHANGED: BarlowBeestonLikelihood.py
                               (receives weights, doesn't know about CPT)
```

---

## Physics Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CPT TEST PHYSICS                                    │
└─────────────────────────────────────────────────────────────────────────────┘

Standard Model (CPT Conserved):
───────────────────────────────
    Δm²₃₁(ν) = Δm²₃₁(ν̄)

    All events propagated with same mass splitting.


CPT Violation Test:
───────────────────
    Δm²₃₁(ν) ≠ Δm²₃₁(ν̄)

    ┌─────────────────┐         ┌─────────────────┐
    │    NEUTRINOS    │         │  ANTINEUTRINOS  │
    │                 │         │                 │
    │  P(νμ → νμ)    │         │  P(ν̄μ → ν̄μ)   │
    │  depends on     │         │  depends on     │
    │  Δm²₃₁ = Dm231  │         │  Δm²₃₁ = Dm231_bar │
    └─────────────────┘         └─────────────────┘

    Different oscillation patterns → Different event distributions
    → Distinguishable in (E, cos_zenith) binning
    → Constrains |Δm²₃₁(ν) - Δm²₃₁(ν̄)|


Scan Strategy:
──────────────
    • Generate Asimov data at TRUE point: Dm231 = Dm231_bar = 2.511e-3 (CPT symmetric)
    • Scan TEST points: (Dm231, Dm231_bar) grid
    • At each point: minimize χ² over 14 nuisance parameters
    • Result: Δχ² surface → Confidence contours → CPT limit
```
