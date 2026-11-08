"""Oscillated-flux tensors phi[dcp, type, flavour, E, cosZ] per lattice cell.

phi is the nuSQuIDS-evolved atmospheric flux x oscillation, EvalFlavor at (flavour,
cosZ, E, type), evaluated on a true grid. type 0 = nu / 1 = nubar, flavour 0 = e /
1 = mu / 2 = tau. One npz per (i_dm, i_s23) cell, ``osc_tensor_{i:03d}_{j:03d}.npz``,
holding all 20 dCP nodes; task = i_dm*41 + i_s23.

SK   The 400 x 80 grid of the SK response, s13 0.022, 4pi averaging. The nuSQuIDS energy
     nodes are geomspace(emin, emax, 200) of the pinned true-E range of the response, and
     no MC is read: a stand-in for the experiment provides that range and the SK flux.
     Averaged evaluation is deterministic, so these tensors repeat bit for bit.
IC   The true cell centres of the IC response, s13 0.022, averaging off.
ORCA The quantized true centres of the ORCA response, s13 0.022, averaging off; each cell
     is the mean over 5 log-uniform sub-bin energies e_c * 10**(0.1*((j+0.5)/5 - 0.5)).

IC and ORCA evaluate with averaging off, i.e. EvalFlavor(..., randomize_height=True):
every call is a fresh production-height draw, so their tensors are one realization.
nuSQuIDS' generator restarts in each process, and every call advances it.
"""
import os

import numpy as np

N_DM = 41
N_S23 = 41
N_DCP = 20
DM21 = 7.41e-5                      # Dm221, the dm3l IO offset
S13 = 0.022
S23_BOX = (0.30, 0.70)
SK_DM_BOX = {"NO": (1.6e-3, 3.0e-3), "IO": (-2.9259e-3, -1.5259e-3)}
NO_DM_BOX = (1.6e-3, 3.0e-3)

# SK oscillation object
SK_S12 = 0.303
SK_DM221 = 7.41e-5
SK_ETRUE_RANGE = (0.1000172462187625, 466048.7104907757)   # = the SK response's pin
SK_N_ETRUE, SK_N_CZTRUE = 400, 80
SK_AVG_SCALE = 4.0 * np.pi
# Anchor point of the Ninv pin
PE_NOMINAL = dict(dm231=2.475e-3, s23=0.45, dcp=-1.89, s13=0.020)


# ----------------------------------------------------------------------- lattices
def sk_lattice(ordering):
    """(dm231[41], s23[41], dcp[20]) of the SK tensors."""
    lo, hi = SK_DM_BOX[ordering]
    return (np.linspace(lo, hi, N_DM), np.linspace(*S23_BOX, N_S23),
            np.linspace(0.0, 2.0 * np.pi, N_DCP, endpoint=False))


def dm3l_lattice(ordering):
    """(dm231[41], s23[41], dcp[20]) of the IC and ORCA tensors
    (IO: Dm231 = -(Dm231_NO - Dm221))."""
    lo, hi = NO_DM_BOX
    if ordering == "NO":
        dm = np.linspace(lo, hi, N_DM)
    elif ordering == "IO":
        dm = np.linspace(-(hi - DM21), -(lo - DM21), N_DM)
    else:
        raise ValueError(f"ordering {ordering!r} is neither 'NO' nor 'IO'")
    return dm, np.linspace(*S23_BOX, N_S23), np.arange(N_DCP) * 2.0 * np.pi / N_DCP


def cell_of_task(task):
    if not 0 <= task < N_DM * N_S23:
        raise ValueError(f"task {task} outside [0, {N_DM * N_S23 - 1}]")
    return task // N_S23, task % N_S23


def cell_path(out_dir, i, j):
    return os.path.join(out_dir, f"osc_tensor_{i:03d}_{j:03d}.npz")


# ------------------------------------------------------------------------ helpers
def grid_centers(e_edges, z_edges):
    return np.sqrt(e_edges[:-1] * e_edges[1:]), 0.5 * (z_edges[:-1] + z_edges[1:])


def set_grid_coords(osc, e_c, z_c):
    """Override the osc object's per-event arrays with the tiled (type, flavor,
    E, cz) grid so GetOscillations evaluates at the grid points."""
    nE, nZ = e_c.size, z_c.size
    E2, Z2 = np.meshgrid(e_c, z_c, indexing="ij")     # (nE, nZ)
    cells_E = E2.ravel()
    cells_Z = Z2.ravel()
    n_cells = cells_E.size
    types, flavors = [], []
    Es, Zs = [], []
    for t in (0, 1):           # 0 = nu, 1 = nubar  (NSQNeutrinoType convention)
        for f in (0, 1, 2):    # e, mu, tau         (NSQNeutrinoFlavor convention)
            types.append(np.full(n_cells, t, dtype=np.uint32))
            flavors.append(np.full(n_cells, f, dtype=np.uint32))
            Es.append(cells_E)
            Zs.append(cells_Z)
    osc.ETrue = np.concatenate(Es)
    osc.CosZTrue = np.concatenate(Zs)
    osc.NSQneutype = np.concatenate(types).tolist()
    osc.NSQneuflavor = np.concatenate(flavors).tolist()
    return nE, nZ


def eval_point(osc, dm231, s23, dcp, nE, nZ, s13):
    P = osc.Parameters
    P["Dm231"] = dm231
    P["Sin2Theta23"] = s23
    P["dCP"] = dcp
    P["Sin2Theta13"] = s13
    w = np.asarray(osc.GetOscillations(), float)
    return w.reshape(2, 3, nE, nZ).astype(np.float32)


def check_phi(phi, label):
    if not np.all(np.isfinite(phi)):
        raise RuntimeError(f"{label}: non-finite phi")
    if phi.min() < 0:
        raise RuntimeError(f"{label}: negative phi (min {phi.min():.3e})")
    print(f"[tensors] {label}: phi {phi.shape} finite, min {phi.min():.3e} "
          f"max {phi.max():.3e}")


# --------------------------------------------------------------------------- SK
class _SKGridSource:
    """The things AtmosphericOscillations reads from the SK experiment, without the MC:
    true-E range (node placement), Z_edges, the per-event arrays (replaced by
    set_grid_coords) and the experiment's initial-flux function."""

    def __init__(self, etrue_min, etrue_max):
        self.Etrue_min, self.Etrue_max = float(etrue_min), float(etrue_max)
        self.Z_edges = [-1, 1]
        self.NumberOfEvents = 0
        self.ETrue = np.zeros(0)
        self.CosZTrue = np.zeros(0)
        self.nuPDG = np.zeros(0, dtype=np.int64)

    def SetInitialFlux(self, energy_nodes, cth_nodes, neutrino_flavors):
        from pynu.Experiments.SuperK_Atm_Pheno import SuperK

        return SuperK.SetInitialFlux(self, energy_nodes, cth_nodes, neutrino_flavors)


def sk_osc():
    from pynu.PhysicsTunes.Oscillations.AtmOsc import AtmosphericOscillations

    osc = AtmosphericOscillations(3, _SKGridSource(*SK_ETRUE_RANGE), avg_scale=SK_AVG_SCALE)
    osc.Parameters["Sin2Theta12"] = SK_S12
    osc.Parameters["Dm221"] = SK_DM221
    return osc


def _sk_grid():
    from .sk_response import make_true_grid

    e_edges, z_edges = make_true_grid(SK_ETRUE_RANGE[0], SK_ETRUE_RANGE[1], SK_N_ETRUE,
                                      SK_N_CZTRUE)
    return e_edges, z_edges


def build_sk_cell(task, ordering, out_dir):
    i, j = cell_of_task(task)
    dm_ax, s23_ax, dcp_nodes = sk_lattice(ordering)
    dm, s23 = dm_ax[i], s23_ax[j]
    e_edges, z_edges = _sk_grid()
    osc = sk_osc()
    nE, nZ = set_grid_coords(osc, *grid_centers(e_edges, z_edges))
    phi = np.stack([eval_point(osc, dm, s23, dcp, nE, nZ, S13) for dcp in dcp_nodes])
    check_phi(phi, f"sk/{ordering} ({i},{j}) dm231={dm:.4e} s23={s23:.4f}")
    out = cell_path(out_dir, i, j)
    np.savez_compressed(out, phi=phi, dm231=dm, s23=s23, dcp=dcp_nodes,
                        e_edges=e_edges, z_edges=z_edges)
    return out


def build_sk_single(out_path):
    """The anchor tensor of the Ninv pin: one off-lattice point, one dCP slice,
    phi (1, 2, 3, nE, nZ) at PE_NOMINAL."""
    dm231, s23, dcp, s13 = (PE_NOMINAL[k] for k in ("dm231", "s23", "dcp", "s13"))
    e_edges, z_edges = _sk_grid()
    osc = sk_osc()
    nE, nZ = set_grid_coords(osc, *grid_centers(e_edges, z_edges))
    dcp_list = np.array([dcp % (2 * np.pi)])
    phi = np.stack([eval_point(osc, dm231, s23, d, nE, nZ, s13) for d in dcp_list])
    check_phi(phi, f"sk single dm231={dm231} s23={s23} dcp={dcp_list[0]} s13={s13}")
    np.savez_compressed(out_path, phi=phi, dm231=dm231, s23=s23, dcp=dcp_list,
                        s13=s13, e_edges=e_edges, z_edges=z_edges)
    return out_path


# ----------------------------------------------------------------------- IC/ORCA
def _event_osc(event_xml, values, response):
    from .events import load_pynufit

    _, _, _, pt = load_pynufit(event_xml, values)
    osc = pt.OscillationTunes
    resp = np.load(response, allow_pickle=True)
    e_c = np.asarray(resp["e_true_centers"], float)
    z_c = np.asarray(resp["cz_true_centers"], float)
    if not (np.all(np.isfinite(e_c)) and np.all(np.isfinite(z_c))):
        raise RuntimeError(f"{response}: NaN true-cell centres")
    return osc, e_c, z_c


def build_ic_cell(task, ordering, out_dir, event_xml, ic_mc, ic_response):
    i, j = cell_of_task(task)
    dm_ax, s23_ax, dcp_nodes = dm3l_lattice(ordering)
    dm, s23 = float(dm_ax[i]), float(s23_ax[j])
    osc, e_c, z_c = _event_osc(event_xml, {"IC_MC": ic_mc}, ic_response)
    nE, nZ = set_grid_coords(osc, e_c, z_c)
    phi = np.stack([eval_point(osc, dm, s23, float(dcp), nE, nZ, S13)
                    for dcp in dcp_nodes]).astype(np.float32)
    check_phi(phi, f"ic/{ordering} ({i},{j}) dm231={dm:.4e} s23={s23:.4f}")
    out = cell_path(out_dir, i, j)
    np.savez_compressed(out, phi=phi, dm231=dm, s23=s23, dcp=dcp_nodes, s13=S13,
                        ordering=np.array(ordering), e_true_centers=e_c,
                        cz_true_centers=z_c, osc_averaging=np.array("off"))
    return out


ORCA_DLOG = 0.1      # log10 width of every ORCA true-energy bin
ORCA_NSUB = 5        # sub-bin energies per cell


def orca_sub_grids(e_c, nsub):
    return [e_c * 10.0 ** (ORCA_DLOG * ((j + 0.5) / nsub - 0.5)) for j in range(nsub)]


def build_orca_cell(task, ordering, out_dir, event_xml, orca_mc, orca_response):
    i, j = cell_of_task(task)
    dm_ax, s23_ax, dcp_nodes = dm3l_lattice(ordering)
    dm, s23 = float(dm_ax[i]), float(s23_ax[j])
    osc, e_c, z_c = _event_osc(event_xml, {"ORCA_MC": orca_mc}, orca_response)
    nsub = ORCA_NSUB
    acc = None
    for e_j in orca_sub_grids(e_c, nsub):
        nE, nZ = set_grid_coords(osc, e_j, z_c)
        blk = np.stack([np.asarray(eval_point(osc, dm, s23, float(dcp), nE, nZ, S13), float)
                        for dcp in dcp_nodes])
        acc = blk if acc is None else acc + blk
    phi = (acc / nsub).astype(np.float32)
    check_phi(phi, f"orca/{ordering} ({i},{j}) dm231={dm:.4e} s23={s23:.4f} nsub={nsub}")
    out = cell_path(out_dir, i, j)
    np.savez_compressed(out, phi=phi, dm231=dm, s23=s23, dcp=dcp_nodes, s13=S13,
                        ordering=np.array(ordering), nsub=np.int64(nsub),
                        e_true_centers=e_c, cz_true_centers=z_c,
                        osc_averaging=np.array("off"))
    return out
