"""Per-true-E flux-ratio q tables (the banded flux-ratio dials' abundance ratios).

  SK    site "sk",  center "geometric" on the SK response
  IC    site "spl", center "stored" on the IC response
  ORCA  site "spl", center "stored" on the ORCA response
(ORCA uses the South-Pole table because that is the flux its forward model uses,
Orca.py SetInitialFlux.)

q(E) = sum_z Phi_heavy(E, z) / sum_z Phi_light(E, z) over the response cosZ cell
centres at the true-E cell centres:
  q_flavor = (numu + numubar) / (nue + nuebar), q_nuebar = nuebar / nue,
  q_numubar = numubar / numu.
"""
import os

import numpy as np

SITE_MODEL = {"sk": "IPhonda2014_sk_solmin", "spl": "IPhonda2014_spl_solmin"}
HI_MODEL = "honda2006"
HI_ABOVE = 1e4      # GeV
CLAMP_LOW = 0.1     # GeV


def build_fluxq(response, out_path, site, center):
    import nuflux

    model = SITE_MODEL[site]

    z = np.load(response, allow_pickle=False)
    e_key = "e_edges" if "e_edges" in z.files else "e_true_edges"
    z_key = "z_edges" if "z_edges" in z.files else "cz_true_edges"
    e_edges, z_edges = z[e_key], z[z_key]
    if center == "stored":
        e_c = np.asarray(z["e_true_centers"], float)
    elif center == "geometric":
        e_c = np.sqrt(e_edges[:-1] * e_edges[1:])
    else:
        raise ValueError(f"center must be 'geometric' or 'stored', got {center!r}")
    z_c = 0.5 * (z_edges[:-1] + z_edges[1:])

    mid = nuflux.makeFlux(model)
    hi = nuflux.makeFlux(HI_MODEL)
    SP = {"nue": nuflux.NuE, "nuebar": nuflux.NuEBar,
          "numu": nuflux.NuMu, "numubar": nuflux.NuMuBar}

    def phi(sp, e, cz):
        src = mid if e <= HI_ABOVE else hi
        return max(src.getFlux(SP[sp], e, cz), 0.0)

    F = {s: np.zeros(e_c.size) for s in SP}
    e_eval = np.clip(e_c, CLAMP_LOW, None)     # clamp below the nuflux rail
    for i, e in enumerate(e_eval):
        for s in SP:
            F[s][i] = sum(phi(s, float(e), float(c)) for c in z_c)

    eps = 1e-300
    q_flavor = (F["numu"] + F["numubar"]) / np.maximum(F["nue"] + F["nuebar"], eps)
    q_nuebar = F["nuebar"] / np.maximum(F["nue"], eps)
    q_numubar = F["numubar"] / np.maximum(F["numu"], eps)
    prov = (f"{model} (<={HI_ABOVE:g} GeV) + {HI_MODEL} (>{HI_ABOVE:g}); "
            f"cosZ flux-sum over response z grid; E<{CLAMP_LOW:g} clamped")
    np.savez(out_path, q_flavor=q_flavor, q_nuebar=q_nuebar, q_numubar=q_numubar,
             e_c=e_c, provenance=np.array(prov))
    print(f"[fluxq] {response} -> {out_path} ({os.path.getsize(out_path)} bytes): "
          f"{prov}; E n={e_c.size} ({center} centres), cosZ n={z_c.size}")
    return out_path
