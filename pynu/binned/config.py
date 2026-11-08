"""Read a scan configuration XML (``configs/*.xml``) into plain Python objects.

One XML file describes one scan completely: the input files, the oscillation
lattice and the region of it to scan, the minimizer protocol, each experiment's
parameter order, and the full nuisance-parameter table (name, prior centre and
width, fit bounds, start value, pinned or free). Nothing is read from environment
variables except ``${VAR}`` references inside paths.

The parameter table is the single source of truth for the fit. The code derives
bounds by fixed rules (see ``arms.py``) and refuses to run if they differ from
the ``lo``/``hi`` written in the XML, so the file always states what was fitted.
"""
import hashlib
import os
import re
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

EXPERIMENTS = ("SK", "IC", "ORCA")
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


def _bool(s, default=False):
    if s is None:
        return default
    v = str(s).strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    raise ValueError(f"not a boolean: {s!r}")


def _ranges(spec):
    """'15-34' -> [15..34]; '0,2,5' -> [0, 2, 5]; 'all' -> None."""
    if spec is None or spec.strip() == "all":
        return None
    out = []
    for tok in spec.replace(" ", "").split(","):
        if "-" in tok:
            a, b = tok.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif tok:
            out.append(int(tok))
    return out


def expand_path(p):
    """Expand ${VAR} and ~; fail if any ${VAR} is left unresolved."""
    if p is None:
        return None
    q = os.path.expanduser(os.path.expandvars(p))
    left = re.findall(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?", q)
    if left:
        raise KeyError(f"path {p!r}: environment variable(s) {left} not set")
    return q


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Dial:
    name: str
    nominal: float
    sigma: float
    lo: float
    hi: float
    pinned: bool = False
    block: str = ""                 # combined fit: "shared" (one slot) or "sk" (SK-only)
    experiments: tuple = ()
    start: dict = field(default_factory=dict)   # {"": value} or {"NO": .., "IO": ..}

    def start_value(self, ordering):
        if ordering in self.start:
            return self.start[ordering]
        if "" in self.start:
            return self.start[""]
        return self.nominal


@dataclass
class Arm:
    name: str                       # SK / IC / ORCA
    dials: list                     # the engine's parameter order


@dataclass
class Lattice:
    dm31_min: float
    dm31_max: float
    n_dm31: int
    s23_min: float
    s23_max: float
    n_s23: int
    n_dcp: int
    dm21: float
    sin2theta13: float

    def dm31_axis(self, ordering):
        """Signed Delta m^2_31. IO uses Delta m^2_31(IO) = -(Delta m^2_31(NO) - Delta m^2_21)."""
        if ordering == "NO":
            return np.linspace(self.dm31_min, self.dm31_max, self.n_dm31)
        if ordering == "IO":
            return np.linspace(-(self.dm31_max - self.dm21),
                               -(self.dm31_min - self.dm21), self.n_dm31)
        raise ValueError(ordering)

    def s23_axis(self):
        return np.linspace(self.s23_min, self.s23_max, self.n_s23)

    def dcp_nodes(self):
        return np.arange(self.n_dcp) * 2.0 * np.pi / self.n_dcp


@dataclass
class Region:
    ordering: str
    i_dm31: list
    i_s23: list
    dcp_nodes: list


@dataclass
class Reference:
    ordering: str
    cell: tuple
    chi2: float
    tol: float
    note: str = ""
    tol_rebuilt: float = 0.0


@dataclass
class Config:
    path: str
    name: str
    experiments: tuple
    inputs: dict
    input_md5: dict
    lattice: Lattice
    regions: dict
    fit: dict
    arms: dict
    dials: list
    references: list

    # ------------------------------------------------------------------ helpers
    @property
    def combined(self):
        return len(self.experiments) > 1

    def dial(self, name, prefer=("", "shared", "sk")):
        """The dial called ``name``. In the combined fit a name can have two slots
        (a pinned shared one and SK's own); ``prefer`` orders the blocks to look in."""
        for block in prefer:
            for d in self.dials:
                if d.name == name and d.block == block:
                    return d
        raise KeyError(name)

    def dial_table(self, names, prefer=("", "shared", "sk")):
        """(nominal, sigma, lo, hi, pinned) arrays for the given names, in that order."""
        ds = [self.dial(n, prefer) for n in names]
        return (np.array([d.nominal for d in ds], float),
                np.array([d.sigma for d in ds], float),
                np.array([d.lo for d in ds], float),
                np.array([d.hi for d in ds], float),
                np.array([d.pinned for d in ds], bool))

    def input(self, key, ordering=None):
        p = expand_path(self.inputs[key])
        if ordering is not None:
            p = p.replace("{ordering}", ordering)
        if key in self.input_md5 and os.path.isfile(p):
            got = md5_file(p)
            if got != self.input_md5[key]:
                warnings.warn(f"{key}: {p} has md5 {got}, the published file has "
                              f"{self.input_md5[key]}. Expected if you built this input "
                              f"yourself; tests/test_build.py compares it with the "
                              f"published one array by array.", stacklevel=2)
        return p

    def region_cells(self, ordering):
        """(i_dm31, i_s23) legs of the region, row-major; task index = position here."""
        r = self.regions[ordering]
        idm = r.i_dm31 if r.i_dm31 is not None else list(range(self.lattice.n_dm31))
        is23 = r.i_s23 if r.i_s23 is not None else list(range(self.lattice.n_s23))
        return [(i, j) for i in idm for j in is23]

    def dcp_order(self, ordering):
        r = self.regions[ordering]
        return r.dcp_nodes if r.dcp_nodes is not None else list(range(self.lattice.n_dcp))


def load(path):
    """Parse a scan configuration XML."""
    root = ET.parse(path).getroot()
    if root.tag != "binned_scan":
        raise ValueError(f"{path}: root element is <{root.tag}>, expected <binned_scan>")
    exps = tuple(root.get("experiments").split())
    bad = [e for e in exps if e not in EXPERIMENTS]
    if bad:
        raise ValueError(f"{path}: unknown experiment(s) {bad}")

    inputs, md5 = {}, {}
    for el in root.find("inputs"):
        key = el.get("key")
        inputs[key] = el.get("path")          # ${VAR} expanded when the file is used
        if el.get("md5"):
            md5[key] = el.get("md5")

    la = root.find("lattice").attrib
    lattice = Lattice(float(la["dm31_min"]), float(la["dm31_max"]), int(la["n_dm31"]),
                      float(la["s23_min"]), float(la["s23_max"]), int(la["n_s23"]),
                      int(la["n_dcp"]), float(la["dm21"]), float(la["sin2theta13"]))

    regions = {}
    for el in root.findall("region"):
        o = el.get("ordering")
        regions[o] = Region(o, _ranges(el.get("i_dm31")), _ranges(el.get("i_s23")),
                            _ranges(el.get("dcp_nodes")))

    fe = root.find("fit")
    fit = dict(fe.attrib)
    for k in ("ftol", "gtol"):
        if k in fit:
            fit[k] = float(fit[k])
    for k in ("maxiter", "npolish"):
        if k in fit:
            fit[k] = int(fit[k])
    if "polish_min_gain" in fit:
        fit["polish_min_gain"] = float(fit["polish_min_gain"])
    fit["warm_carry"] = _bool(fit.get("warm_carry"))

    dials = []
    for el in root.find("dials"):
        a = el.attrib
        start = {}
        for k, v in a.items():
            if k == "start":
                start[""] = float(v)
            elif k.startswith("start_"):
                start[k[6:]] = float(v)
        dials.append(Dial(name=a["name"], nominal=float(a["nominal"]),
                          sigma=float(a["sigma"]), lo=float(a["lo"]), hi=float(a["hi"]),
                          pinned=_bool(a.get("pinned")), block=a.get("block", ""),
                          experiments=tuple(a.get("experiments", "").split()),
                          start=start))
    names = [d.name for d in dials]
    arms = {}
    for el in root.findall("arm"):
        # the engine's parameter order: the explicit list, or the table order of the
        # dials that feed this experiment
        a = el.get("name")
        order = el.get("dials").split() if el.get("dials") else \
            [d.name for d in dials if not d.experiments or a in d.experiments]
        arms[a] = Arm(a, order)
    keys = [(d.block, d.name) for d in dials]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{path}: a dial name appears twice in the same block")
    for arm in arms.values():
        missing = [n for n in arm.dials if n not in names]
        if missing:
            raise ValueError(f"{path}: arm {arm.name} lists dials with no <dial> entry: "
                             f"{missing}")

    refs = []
    for el in root.findall("reference"):
        refs.append(Reference(el.get("ordering"),
                              tuple(int(x) for x in el.get("cell").split(",")),
                              float(el.get("chi2")), float(el.get("tol")),
                              el.get("note", ""),
                              float(el.get("tol_rebuilt", el.get("tol")))))

    return Config(path=os.path.abspath(path), name=root.get("name"), experiments=exps,
                  inputs=inputs, input_md5=md5, lattice=lattice, regions=regions,
                  fit=fit, arms=arms, dials=dials, references=refs)
