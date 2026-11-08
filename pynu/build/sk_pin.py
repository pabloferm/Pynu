"""The SK kernel ``Ninv``: the SK response pinned per species per reco bin to the SK
release NO tables at the anchor oscillation point.

Release tables: SK atmospheric 2023 data release v1 (Zenodo record 8401262,
``sk_atm_2023.tar.gz`` md5 58f26b39ba0bb36cc3570b7723298e2b), directory
``sk_atm_2023/bins/normal``.

Anchor tensor (``osc_tensor_pe_nominal.npz``): Dm231 2.475e-3, s23 0.45, dCP -1.89 rad
(stored mod 2pi = 4.39318531), s13 0.020, on the pinned 400 x 80 SK grid; built by
``pynu.build.osc.build_sk_single`` (``scripts/build.py tensors --experiment sk --single``).
"""
import json
import os
import time

import numpy as np

SPECIES = ["nue", "numu", "nutau", "nuebar", "numubar", "nunc"]
NOFILE = {"nue": "sk_2023_MCNueNO.txt", "numu": "sk_2023_MCNumuNO.txt",
          "nutau": "sk_2023_MCNutauNO.txt", "nuebar": "sk_2023_MCNueBarNO.txt",
          "numubar": "sk_2023_MCNumuBarNO.txt", "nunc": "sk_2023_MCNCNO.txt"}
_PDG_SPECIES = {12: "nue", 14: "numu", 16: "nutau", -12: "nuebar", -14: "numubar",
                -16: "nutau"}


def ld(p):
    return np.array([[float(x) for x in l.split()] for l in open(p)
                     if l.strip() and not l.strip().startswith("#")])


def make_anchor_counts(response, anchor, out_path):
    """Per-species per-bin counts of ``response`` contracted with the anchor tensor.
    CC weight = R_v x phi[type, flavour](e, z); NC weight = R_v."""
    Z = np.load(response, allow_pickle=True)
    A = np.load(anchor, allow_pickle=True)
    phi = A["phi"]
    phi = phi[0] if phi.ndim == 5 else phi
    assert np.allclose(A["e_edges"], Z["e_edges"])
    classes = Z["classes"]; pdg = classes[:, 0]; cc = classes[:, 1]
    fl = np.clip(np.abs(pdg) // 2 - 6, 0, 2); ty = (pdg < 0).astype(int)
    P = np.where(cc[:, None, None] == 1, phi[ty, fl].astype(float), 1.0)
    Rk, Re, Rz, Rb, Rv = (Z[k] for k in ("R_k", "R_e", "R_z", "R_b", "R_v"))
    n_bins = int(Z["n_bins"]); w = Rv * P[Rk, Re, Rz]
    sp_of_cls = np.empty(classes.shape[0], dtype=object)
    for k in range(classes.shape[0]):
        sp_of_cls[k] = "nunc" if cc[k] == 0 else _PDG_SPECIES[int(pdg[k])]
    counts = np.zeros((len(SPECIES), n_bins))
    for i, sp in enumerate(SPECIES):
        m = np.isin(Rk, np.where(sp_of_cls == sp)[0])
        counts[i] = np.bincount(Rb[m], weights=w[m], minlength=n_bins)
    tot = counts.sum()
    for i, sp in enumerate(SPECIES):
        print(f"[anchor-counts] {sp:8s} total {counts[i].sum():10.2f}")
    print(f"[anchor-counts] ALL total {tot:.2f}  (expected 62088.4; "
          f"nutau 307.6)")
    assert 50000 < tot < 80000, "total outside the sanity window"
    np.savez_compressed(out_path, counts=counts, species=np.array(SPECIES),
                        n_bins=np.int64(n_bins),
                        source=json.dumps(dict(response=os.path.abspath(response),
                                               anchor=os.path.abspath(anchor),
                                               built=time.strftime("%Y-%m-%d %H:%M"))))
    print(f"[anchor-counts] wrote {out_path}")
    return out_path


def pin_to_release(ours, our_counts, anchor, release_dir, out_path, min_our=0.5,
                   max_scale=5.0):
    """Every entry of species s in bin b is scaled by Counts^NO_{s,b} / ours_{s,b}(anchor)
    where ours >= min_our (clipped to [1/max_scale, max_scale]); S2 by the square. Checks
    that the re-contracted totals equal the release tables to 1e-9 on pinned, unclipped
    bins."""
    t0 = time.time()
    Z = np.load(ours, allow_pickle=True); C = np.load(our_counts, allow_pickle=True)
    assert [str(x) for x in C["species"]] == SPECIES
    n_bins = int(Z["n_bins"])

    def _load_anchor():
        A_ = np.load(anchor, allow_pickle=True); phi_ = A_["phi"]
        return phi_[0] if phi_.ndim == 5 else phi_

    phi0 = _load_anchor()
    _cl = Z["classes"]; _pdg = _cl[:, 0]; _cc = _cl[:, 1]
    _fl = np.clip(np.abs(_pdg) // 2 - 6, 0, 2); _ty = (_pdg < 0).astype(int)
    _P = np.where(_cc[:, None, None] == 1, phi0[_ty, _fl].astype(float), 1.0)
    _w = Z["R_v"] * _P[Z["R_k"], Z["R_e"], Z["R_z"]]
    _spk = np.array([SPECIES.index("nunc") if _cc[k] == 0
                     else SPECIES.index(_PDG_SPECIES[int(_pdg[k])])
                     for k in range(_cl.shape[0])])
    ours_c = np.zeros((len(SPECIES), n_bins))
    for i in range(len(SPECIES)):
        m = _spk[Z["R_k"]] == i
        ours_c[i] = np.bincount(Z["R_b"][m], weights=_w[m], minlength=n_bins)
    print(f"[Ninv] per-species counts at the anchor: total {ours_c.sum():.1f}")
    rel = np.array([ld(os.path.join(release_dir, NOFILE[sp]))[:, 0] for sp in SPECIES])
    pin = ours_c >= min_our
    scale = np.where(pin, np.clip(rel / np.maximum(ours_c, 1e-300), 1.0 / max_scale,
                                  max_scale), 1.0)   # (6, n_bins)
    classes = Z["classes"]; pdg = classes[:, 0]; cc = classes[:, 1]
    sp_idx = np.array([SPECIES.index("nunc") if cc[k] == 0
                       else SPECIES.index(_PDG_SPECIES[int(pdg[k])])
                       for k in range(classes.shape[0])])
    for i, sp in enumerate(SPECIES):
        m = pin[i]
        if not m.any():
            print(f"[Ninv] {sp:8s} no pinned bins"); continue
        clipped = int((np.abs(np.log(scale[i][m])) >= np.log(max_scale) - 1e-12).sum())
        print(f"[Ninv] {sp:8s} scale median {np.median(scale[i][m]):.4f} min "
              f"{scale[i][m].min():.4f} max {scale[i][m].max():.4f}; pinned "
              f"{int(m.sum())} (clipped {clipped})")
    f = scale[sp_idx[Z["R_k"]], Z["R_b"]]
    Rv = Z["R_v"] * f; S2v = Z["S2_v"] * scale[sp_idx[Z["S2_k"]], Z["S2_b"]] ** 2
    # check: re-contract with the same anchor tensor
    phi = _load_anchor()
    fl = np.clip(np.abs(pdg) // 2 - 6, 0, 2); ty = (pdg < 0).astype(int)
    P = np.where(cc[:, None, None] == 1, phi[ty, fl].astype(float), 1.0)
    w = Rv * P[Z["R_k"], Z["R_e"], Z["R_z"]]
    worst = 0.0
    for i in range(len(SPECIES)):
        m = sp_idx[Z["R_k"]] == i
        tot = np.bincount(Z["R_b"][m], weights=w[m], minlength=n_bins)
        ok = pin[i] & (np.abs(np.log(scale[i])) < np.log(max_scale) - 1e-12)
        if not ok.any():
            continue
        d = np.max(np.abs(tot[ok] - rel[i][ok]) / np.maximum(rel[i][ok], 1e-9))
        worst = max(worst, d)
    print(f"[Ninv] check: re-contracted per-species per-bin totals vs release NO: "
          f"max rel dev {worst:.3e}")
    assert worst < 1e-9
    meta = json.loads(str(Z["meta"])) if Z["meta"].ndim == 0 else {}
    meta = dict(meta)
    meta.update(dict(arm="Ninv_pinned", mirror="response pinned per species per bin "
                     "to the SK release NO tables at the anchor point", min_our=min_our,
                     max_scale=max_scale, source=os.path.abspath(ours),
                     our_counts=os.path.abspath(our_counts),
                     built=time.strftime("%Y-%m-%d %H:%M")))
    np.savez_compressed(
        out_path, sample_event_counts=Z["sample_event_counts"], classes=classes,
        xsec_tune_names=Z["xsec_tune_names"],
        R_k=Z["R_k"], R_e=Z["R_e"], R_z=Z["R_z"], R_b=Z["R_b"], R_era=Z["R_era"], R_v=Rv,
        S2_k=Z["S2_k"], S2_e=Z["S2_e"], S2_z=Z["S2_z"], S2_b=Z["S2_b"],
        S2_era=Z["S2_era"], S2_v=S2v,
        n_era=Z["n_era"], e_edges=Z["e_edges"], z_edges=Z["z_edges"], n_bins=Z["n_bins"],
        observed=Z["observed"], sample_table=Z["sample_table"], meta=json.dumps(meta))
    print(f"[Ninv] wrote {out_path} ({time.time() - t0:.0f}s) PASS")
    return out_path


def build_ninv(response, anchor, release_dir, out_path):
    """Anchor counts (written next to ``out_path``), then the pin."""
    counts_path = os.path.join(os.path.dirname(os.path.abspath(out_path)),
                               "our_counts_pe_nominal.npz")
    make_anchor_counts(response, anchor, counts_path)
    return pin_to_release(response, counts_path, anchor, release_dir, out_path)
