"""Run, check and assemble a lattice scan described by a configuration XML.

    python -m pynu.binned.scan run      --config configs/Combined.xml --ordering NO --task 17 --out cells/NO
    python -m pynu.binned.scan check    --config configs/Combined.xml
    python -m pynu.binned.scan assemble --config configs/Combined.xml --ordering NO --cells cells/NO --out combined_NO.npz

``run`` fits one LEG: one (Delta m^2_31, sin^2 theta_23) cell of the configured
region, at every delta_CP node in the configured order. Task t is the t-th leg of
the region (row-major over i_dm31, i_s23), so a scheduler array of
``len(region)`` tasks covers it; ``python -m pynu.binned.scan legs`` prints that
number (``--tensor-tasks``: the tensor files the region needs). Each node is
written as one JSON; finished nodes are skipped, so a killed task resumes. With
``warm_carry`` the converged parameters of one node seed the next.

``check`` refits the reference cells listed in the configuration (walking the
leg up to the reference node when warm-carry is on) and compares chi^2 with the
recorded value.
"""
import argparse
import json
import os
import socket
import sys
import time

import numpy as np

from . import config as C
from .tensors import TensorDir
from .arms import SKArm, ICArm, ORCAArm, DriverFit, SENTINEL_CHI2
from .combined import CombinedFit


# =============================================================================
# one fitter per configuration
# =============================================================================
class Fitter:
    def __init__(self, cfg, ordering):
        self.cfg, self.ordering = cfg, ordering
        la = cfg.lattice
        self.dm31, self.s23, self.dcp = la.dm31_axis(ordering), la.s23_axis(), la.dcp_nodes()
        self.dirs = {e: TensorDir(cfg.input(f"{e.lower()}_tensors", ordering), la, ordering)
                     for e in cfg.experiments}
        if cfg.combined:
            self.model = CombinedFit(cfg)
            self.names = self.model.flat_names
        elif cfg.experiments == ("SK",):
            self.model = SKArm(cfg)
            self.names = self.model.names
        else:
            arm = ICArm(cfg) if cfg.experiments == ("IC",) else ORCAArm(cfg)
            self.model = DriverFit(cfg, arm)
            self.names = self.model.names
        self.npolish = cfg.fit["npolish"]
        self.warm = cfg.fit["warm_carry"]

    def start(self):
        if self.cfg.combined:
            return self.model.start(self.ordering)
        if self.cfg.experiments == ("SK",):
            return np.array([self.cfg.dial(n).start_value(self.ordering)
                             for n in self.names], float)
        return None                                  # IC / ORCA: cold at nominal

    def leg(self, i, j):
        """Load the leg's tensors once; return fit(node, x0) -> (record, theta)."""
        dm = float(self.dm31[i])
        phis = {e: d.load(i, j) for e, d in self.dirs.items()}
        if self.cfg.combined:
            def fit(d, x0):
                self.model.bind(dm, phis["SK"][d].astype(float),
                                np.asarray(phis["IC"][d], float),
                                np.asarray(phis["ORCA"][d], float))
                return self.model.fit_cell(x0)
        elif self.cfg.experiments == ("SK",):
            def fit(d, x0):
                return self.model.fit_cell(phis["SK"], d, x0, self.npolish)
        else:
            e = self.cfg.experiments[0]

            def fit(d, x0):
                self.model.arm.bind(dm)
                return self.model.fit_cell(np.asarray(phis[e][d], float), x0)
        return fit


def run_leg(fitter, i, j, nodes, out_dir, stop_after=None, log=print):
    """Fit nodes of leg (i, j) in order with optional warm-carry; return records."""
    os.makedirs(out_dir, exist_ok=True)
    fit = fitter.leg(i, j)
    x0, src, recs = fitter.start(), "start", []
    for d in nodes:
        p = os.path.join(out_dir, f"cell_{i:03d}_{j:03d}_{d:02d}.json")
        if os.path.exists(p):
            rec = json.load(open(p))
            x_done = np.asarray(rec["theta"], float)
        else:
            t0 = time.time()
            rec, th = fit(d, x0)
            rec.update(cell=[i, j, d], ordering=fitter.ordering, dm31=float(fitter.dm31[i]),
                       s23=float(fitter.s23[j]), dcp=float(fitter.dcp[d]), seed=src,
                       wall_s=time.time() - t0, host=socket.gethostname(),
                       config=os.path.basename(fitter.cfg.path), names=fitter.names,
                       theta=[float(v) for v in th])
            with open(p + ".tmp", "w") as fh:
                json.dump(rec, fh)
            os.replace(p + ".tmp", p)
            x_done = np.asarray(th, float)
            log(f"cell ({i},{j},{d}) chi2={rec['chi2']:.6f} converged={rec['converged']} "
                f"wall={rec['wall_s']:.0f}s seed={src}")
        recs.append(rec)
        if fitter.warm and rec["chi2"] < SENTINEL_CHI2:
            x0, src = x_done, f"node{d:02d}"
        if stop_after is not None and d == stop_after:
            break
    return recs


# =============================================================================
# assemble
# =============================================================================
def _interval(x, prof, level=1.0):
    """Delta chi^2 = level crossings either side of the profile minimum, by linear
    interpolation between lattice nodes; None if the profile does not cross."""
    k = int(np.nanargmin(prof))
    lo = hi = None
    for a in range(k, 0, -1):
        if prof[a - 1] >= level:
            lo = float(np.interp(level, [prof[a], prof[a - 1]], [x[a], x[a - 1]]))
            break
    for a in range(k, len(prof) - 1):
        if prof[a + 1] >= level:
            hi = float(np.interp(level, [prof[a], prof[a + 1]], [x[a], x[a + 1]]))
            break
    return dict(best=float(x[k]), lo=lo, hi=hi,
                plus=None if hi is None else hi - float(x[k]),
                minus=None if lo is None else float(x[k]) - lo)


def _periodic_interval(deg, prof, level=1.0):
    """The same on the periodic delta_CP axis (evenly spaced nodes, in degrees)."""
    n, step, k = len(prof), 360.0 / len(prof), int(np.nanargmin(prof))
    out = {}
    for name, d in (("plus", +1), ("minus", -1)):
        j, s, out[name] = k, 0, None
        while s < n:
            j2 = (j + d) % n
            if prof[j2] >= level:
                out[name] = (s + (level - prof[j]) / (prof[j2] - prof[j])) * step
                break
            j, s = j2, s + 1
    return dict(best_deg=float(deg[k]), plus_deg=out["plus"], minus_deg=out["minus"],
                depth=float(np.nanmax(prof)))


def summarize(cfg, ordering, chi2):
    """Best fit and Delta chi^2 = 1 intervals of a (n_dm31, n_s23, n_dcp) chi^2 cube,
    over the configured region. Delta m^2_31 intervals are on |Delta m^2_31| in
    units of 1e-3 eV^2; delta_CP intervals in degrees."""
    la = cfg.lattice
    idm = cfg.regions[ordering].i_dm31 or list(range(la.n_dm31))
    is23 = cfg.regions[ordering].i_s23 or list(range(la.n_s23))
    idcp = sorted(cfg.dcp_order(ordering))
    sub = chi2[np.ix_(idm, is23, idcp)]
    floor = float(np.nanmin(sub))
    a, b, c = np.unravel_index(np.nanargmin(sub), sub.shape)
    d = sub - floor
    dm = np.abs(la.dm31_axis(ordering)[idm]) * 1e3
    s23 = la.s23_axis()[is23]
    deg = np.degrees(la.dcp_nodes()[idcp])
    return dict(config=os.path.basename(cfg.path), ordering=ordering, floor=floor,
                argmin=[int(idm[a]), int(is23[b]), int(idcp[c])],
                dm31_1sigma=_interval(dm, np.nanmin(d, axis=(1, 2))),
                s23_1sigma=_interval(s23, np.nanmin(d, axis=(0, 2))),
                dcp_1sigma=_periodic_interval(deg, np.nanmin(d, axis=(0, 1))),
                n_cells=int(np.isfinite(sub).sum()), n_expected=int(sub.size))


def assemble(cfg, ordering, cells_dir, out):
    la = cfg.lattice
    shape = (la.n_dm31, la.n_s23, la.n_dcp)
    chi2 = np.full(shape, np.nan)
    conv = np.zeros(shape, bool)
    for f in os.listdir(cells_dir):
        if f.startswith("cell_") and f.endswith(".json"):
            r = json.load(open(os.path.join(cells_dir, f)))
            i, j, d = r["cell"]
            chi2[i, j, d], conv[i, j, d] = r["chi2"], r["converged"]
    summary = summarize(cfg, ordering, chi2)
    summary["converged"] = int(conv.sum())
    dchi2 = chi2 - summary["floor"]
    np.savez_compressed(out, chi2=chi2, converged=conv, dm31=la.dm31_axis(ordering),
                        s23=la.s23_axis(), dcp=la.dcp_nodes(),
                        profile_dm31_s23=np.nanmin(dchi2, axis=2))
    with open(os.path.splitext(out)[0] + "_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    return summary


# =============================================================================
# CLI
# =============================================================================
def _verdict(diff, ref):
    """``tol`` covers the minimizer path on the published inputs. Inputs you built move
    chi^2 by more, up to ``tol_rebuilt``: rebuilt IC and ORCA tensors are a different
    random draw of the production height than the published ones, and a slightly
    different SK kernel changes the minimizer path."""
    if abs(diff) <= ref.tol:
        return "PASS"
    if abs(diff) <= ref.tol_rebuilt:
        return f"PASS for rebuilt inputs (limit {ref.tol_rebuilt:g})"
    return "FAIL"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m pynu.binned.scan", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="fit one leg (task) of the region")
    r.add_argument("--config", required=True)
    r.add_argument("--ordering", required=True, choices=["NO", "IO"])
    r.add_argument("--task", type=int, help="leg index into the region")
    r.add_argument("--leg", help="i_dm31,i_s23 instead of --task")
    r.add_argument("--out", required=True, help="directory for the cell JSONs")
    g = sub.add_parser("legs", help="print the number of legs of the region")
    g.add_argument("--config", required=True)
    g.add_argument("--ordering", required=True, choices=["NO", "IO"])
    g.add_argument("--tensor-tasks", action="store_true",
                   help="print the tensor-build task of every leg (i_dm31 * n_s23 + i_s23)")
    c = sub.add_parser("check", help="refit the reference cells and compare")
    c.add_argument("--config", required=True)
    c.add_argument("--ordering", choices=["NO", "IO"])
    c.add_argument("--out", default=None, help="scratch directory (default: a temp dir)")
    s = sub.add_parser("assemble", help="cell JSONs -> cube npz + summary")
    s.add_argument("--config", required=True)
    s.add_argument("--ordering", required=True, choices=["NO", "IO"])
    s.add_argument("--cells", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cfg = C.load(a.config)

    if a.cmd == "legs":
        legs = cfg.region_cells(a.ordering)
        if a.tensor_tasks:
            print(" ".join(str(i * cfg.lattice.n_s23 + j) for i, j in legs))
        else:
            print(len(legs))
        return 0
    if a.cmd == "run":
        legs = cfg.region_cells(a.ordering)
        if a.leg:
            i, j = (int(v) for v in a.leg.split(","))
        else:
            if a.task is None or not 0 <= a.task < len(legs):
                raise SystemExit(f"--task must be in [0, {len(legs)})")
            i, j = legs[a.task]
        fitter = Fitter(cfg, a.ordering)
        run_leg(fitter, i, j, cfg.dcp_order(a.ordering), a.out)
        return 0
    if a.cmd == "assemble":
        print(json.dumps(assemble(cfg, a.ordering, a.cells, a.out), indent=1))
        return 0
    if a.cmd == "check":
        import tempfile
        base = a.out or tempfile.mkdtemp(prefix="pynu_check_")
        ok = True
        for ref in cfg.references:
            if a.ordering and ref.ordering != a.ordering:
                continue
            i, j, d = ref.cell
            fitter = Fitter(cfg, ref.ordering)
            order = cfg.dcp_order(ref.ordering)
            nodes = order[:order.index(d) + 1] if fitter.warm else [d]
            recs = run_leg(fitter, i, j, nodes,
                           os.path.join(base, f"{ref.ordering}_{i:03d}_{j:03d}"))
            got = recs[-1]["chi2"]
            diff = got - ref.chi2
            verdict = _verdict(diff, ref)
            ok &= verdict != "FAIL"
            print(f"CHECK {cfg.name} {ref.ordering} cell {ref.cell}: chi2 {got:.10f} "
                  f"reference {ref.chi2:.10f} diff {diff:+.3e} tol {ref.tol:g} "
                  f"{verdict}  {ref.note}", flush=True)
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
