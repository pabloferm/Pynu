"""Gate C-1 — dev-tree <-> upstream module comparator for the IC/ORCA binned port.

Precedent: the SK shipment's E7 comparator (800/800 exact). The port contract
(UPSTREAM_PORT_PLAN_icorca_binned_2026-08-17.md sec 3, ruling 3 "faithful port")
is that every ported module is byte-identical to its certified dev source EXCEPT
for one contiguous import-header block, whose only edits are dropping
`sys.path` bootstrapping and rewriting cross-module imports to package-relative
form. Any other difference invalidates gates G-ORCA-1/2 and G-IC-3/4.

Two assertions per module:

  BODY-IDENTITY (the strong one) — the source lines OUTSIDE the declared header
      region and the upstream lines OUTSIDE the declared header region are equal
      line-for-line, byte for byte. A single changed body line fails the gate
      and is named with its file and line number.

  HEADER-WHITELIST (the self-check on the declaration) — every line INSIDE both
      declared regions matches the import-header pattern. This is what stops a
      too-wide region from hiding a body edit inside the whitelist: a real body
      line placed in the region trips this check.

`orca_binned_support.py` is not a moved file but a verbatim EXTRACTION from the
dev scan driver `orca_exact_scan.py`, so it is compared region by region: each
extracted constant block / function body must equal its source line range
exactly, and every upstream line outside those regions must be new-prologue
(module docstring + imports) or blank.

Run from the repo root:  python3 test/binned_icorca/compare_ported_modules.py
Override the dev tree with --dev-root (default: the AtmNuDataFit checkout that
this branch was ported from).
"""
import argparse
import ast
import difflib
import os
import sys

DEV_ROOT_DEFAULT = os.path.expanduser("~/Desktop/Harvard/AtmNuDataFit")
BINNED_ARMS = "claude/2-atmospheric-oscillation/combined-fit/binned_arms"
MULTI_EXP = "claude/2-atmospheric-oscillation/multi-experiment-systematics"
COMBINED = "claude/2-atmospheric-oscillation/combined-fit"

PKG = "pynu/Experiments"

def header_offenders(lines, region):
    """Statements in `region` that are not import-header material.

    Checked by PARSING the region rather than regex-matching lines, so a
    parenthesized `from x import (a, b,\\n c)` counts as the single import it is
    instead of tripping on its continuation lines. Permitted at the top level:
    `import` / `from ... import`, the try/except-ImportError wrapper the dev tree
    uses for dual script-or-package running (whose branches must themselves be
    imports only), `sys.path.insert(...)` bootstraps, and a `_`-prefixed path
    anchor assignment such as `_HERE = os.path.dirname(...)`. A `def`, a `class`,
    or any real assignment is an offender. Comments and blanks leave no node.
    """
    lo, hi = region
    src = "\n".join(lines[lo - 1:hi])
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return [f"region {lo}-{hi} does not parse as a standalone block: {exc}"]

    def is_import(node):
        return isinstance(node, (ast.Import, ast.ImportFrom))

    def is_syspath(node):
        return (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func).startswith("sys.path."))

    def is_path_anchor(node):
        return (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id.startswith("_")
                and "os.path" in ast.unparse(node.value))

    bad = []
    for node in tree.body:
        if is_import(node) or is_syspath(node) or is_path_anchor(node):
            continue
        if isinstance(node, ast.Try) and all(
                is_import(n) for n in node.body) and all(
                all(is_import(n) for n in h.body) for h in node.handlers):
            continue
        bad.append(f"line {lo + node.lineno - 1} is not import-header material "
                   f"(region declared too wide): {ast.unparse(node)!r}")
    return bad

# (upstream file, dev source file, upstream header lines, dev header lines)
# Line ranges are 1-indexed and INCLUSIVE; None means the module has no edited
# header at all and must be byte-identical end to end.
PAIRS = [
    ("binned_contract.py",     f"{BINNED_ARMS}/binned_contract.py",     None,       None),
    ("binned_dial_fields.py",  f"{BINNED_ARMS}/binned_dial_fields.py",  None,       None),
    ("ic_dial_fields.py",      f"{BINNED_ARMS}/ic_dial_fields.py",      None,       None),
    ("ic_binned_builder.py",   f"{MULTI_EXP}/ic_binned_builder.py",     None,       None),
    ("ic_binned_cells.py",     f"{BINNED_ARMS}/ic_cells.py",            (36, 42),   (36, 45)),
    ("ic_binned_engine.py",    f"{BINNED_ARMS}/ic_binned_engine.py",    (104, 108), (104, 113)),
    ("orca_cell_phi.py",       f"{BINNED_ARMS}/orca_cell_phi.py",       (28, 30),   (28, 34)),
    ("orca_binned_engine.py",  f"{BINNED_ARMS}/orca_binned_engine.py",  (49, 58),   (49, 63)),
    ("orca_binned_builder.py", f"{MULTI_EXP}/orca_binned_builder.py",   (21, 33),   (21, 35)),
]

# The two EXTRACTED support modules. Each is compared region by region against
# its dev-tree scan driver: (label, upstream range, source range), both 1-indexed
# inclusive. `prologue_end` is the last line of the new docstring + imports; every
# upstream line that is neither an extracted region, nor prologue, nor blank is a
# failure, so nothing can be smuggled in between the regions.
EXTRACTIONS = [
    dict(
        upstream="orca_binned_support.py",
        source=f"{COMBINED}/orca_exact_scan.py",
        prologue_end=32,
        # `binned_expectation` is the production reference model of G-ORCA-1 and
        # G-ORCA-2. Plan sec 3.1 O2 listed it as non-shipping; amended 2026-08-17
        # because without it neither ORCA gate can run upstream at all.
        regions=[
            ("grid constants",      (34, 35),   (48, 49)),
            ("_flat900",            (38, 39),   (70, 71)),
            ("observed_900",        (42, 47),   (74, 79)),
            ("muon_900",            (50, 64),   (82, 96)),
            ("nu_cell_index",       (67, 90),   (99, 122)),
            ("poisson_chi2",        (93, 98),   (125, 130)),
            ("binned_expectation",  (101, 120), (185, 204)),
        ],
        # names the extraction must NOT carry, with the reason it was omitted
        omitted={"add_pynu_root": "path helper; no extracted function calls it",
                 "event_expectation": "no gate imports it"},
    ),
    dict(
        upstream="ic_binned_support.py",
        source=f"{MULTI_EXP}/ic_divergence_scan.py",
        prologue_end=30,
        # The IC mirror of the above; the plan had no slot for this module.
        # G-IC-4 compares the engine against `_corrected_expectation`.
        regions=[
            ("grid constants",         (32, 33),   (93, 94)),
            ("POINTS",                 (35, 37),   (96, 98)),
            ("_load_reco_edges",       (40, 43),   (108, 111)),
            ("_digitize_clamp",        (46, 48),   (114, 116)),
            ("_flat200",               (51, 52),   (119, 120)),
            ("observed_200",           (55, 63),   (123, 131)),
            ("muon_200",               (66, 75),   (134, 143)),
            ("nu_index",               (78, 108),  (146, 176)),
            ("poisson_chi2",           (111, 116), (179, 184)),
            ("_hs_params_from_theta",  (124, 125), (235, 236)),
            ("_hs_correction_factor",  (128, 141), (239, 252)),
            ("_corrected_expectation", (144, 154), (255, 265)),
        ],
        omitted={"add_pynu_root": "path helper; no extracted function calls it"},
        # the one hand-written section divider inside the body
        allow_comment_lines=(119, 123),
    ),
]


def read_lines(path):
    with open(path) as fh:
        return fh.read().split("\n")


def outside(lines, region):
    """Lines outside a 1-indexed inclusive region, as (lineno, text) pairs."""
    if region is None:
        return [(i + 1, t) for i, t in enumerate(lines)]
    lo, hi = region
    return [(i + 1, t) for i, t in enumerate(lines) if not (lo <= i + 1 <= hi)]


def inside(lines, region):
    if region is None:
        return []
    lo, hi = region
    return [(i + 1, t) for i, t in enumerate(lines) if lo <= i + 1 <= hi]


def check_pair(up_path, dev_path, up_region, dev_region, label):
    fails = []

    up = read_lines(up_path)
    dev = read_lines(dev_path)

    # --- HEADER-WHITELIST: the declared regions may hold imports only ---------
    if up_region is not None:
        for tag, lines, region in (("upstream", up, up_region), ("dev", dev, dev_region)):
            for msg in header_offenders(lines, region):
                fails.append(f"{label}: {tag} {msg}")

    # --- BODY-IDENTITY: everything outside the regions must match exactly -----
    up_body = outside(up, up_region)
    dev_body = outside(dev, dev_region)
    sm = difflib.SequenceMatcher(None, [t for _, t in dev_body],
                                 [t for _, t in up_body], autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        for k in range(i1, i2):
            fails.append(f"{label}: BODY {op} at dev {dev_path}:{dev_body[k][0]}: "
                         f"{dev_body[k][1]!r}")
        for k in range(j1, j2):
            fails.append(f"{label}: BODY {op} at upstream {up_path}:{up_body[k][0]}: "
                         f"{up_body[k][1]!r}")

    n_body = len(up_body)
    n_hdr_up = len(inside(up, up_region))
    n_hdr_dev = len(inside(dev, dev_region))
    return fails, n_body, n_hdr_up, n_hdr_dev


def check_extraction(up_path, src_path, spec):
    label = spec["upstream"]
    fails = []
    up = read_lines(up_path)
    src = read_lines(src_path)
    covered = set()

    for name, (u0, u1), (s0, s1) in spec["regions"]:
        u_seg = up[u0 - 1:u1]
        s_seg = src[s0 - 1:s1]
        covered.update(range(u0, u1 + 1))
        if len(u_seg) != len(s_seg):
            fails.append(f"{label} [{name}]: region length "
                         f"{len(u_seg)} != source length {len(s_seg)}")
            continue
        for off, (a, b) in enumerate(zip(u_seg, s_seg)):
            if a != b:
                fails.append(
                    f"{label} [{name}]: line {u0 + off} differs from "
                    f"{src_path}:{s0 + off}\n      upstream {a!r}\n      source   {b!r}")

    # every non-extracted upstream line must be prologue, blank, or one of the
    # explicitly declared hand-written comment lines
    c0, c1 = spec.get("allow_comment_lines", (0, -1))
    for i, txt in enumerate(up):
        ln = i + 1
        if ln in covered or ln <= spec["prologue_end"] or txt.strip() == "":
            continue
        if c0 <= ln <= c1 and txt.lstrip().startswith("#"):
            continue
        fails.append(f"{label}: line {ln} is neither an extracted region, "
                     f"prologue, nor blank: {txt!r}")

    # the omitted names must genuinely be unreferenced by the extracted CODE.
    # Checked on the AST, so a prose mention in the prologue does not count.
    tree = ast.parse("\n".join(up))
    refs = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    refs |= {n.name for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for name, why in spec["omitted"].items():
        if name in refs:
            fails.append(f"{label}: references {name}, omitted as "
                         f"'{why}' — the omission is unsound")
    n_lines = sum(u1 - u0 + 1 for _, (u0, u1), _ in spec["regions"])
    return fails, n_lines


def main():
    ap = argparse.ArgumentParser(description="Gate C-1: ported-module comparator.")
    ap.add_argument("--dev-root", default=DEV_ROOT_DEFAULT,
                    help="AtmNuDataFit checkout this branch was ported from")
    ap.add_argument("--repo-root", default=os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    args = ap.parse_args()

    print("=== GATE C-1 — dev<->upstream ported-module comparator ===")
    print(f"  upstream repo : {args.repo_root}")
    print(f"  dev tree      : {args.dev_root}")
    print()

    all_fails = []
    for up_name, dev_rel, up_region, dev_region in PAIRS:
        up_path = os.path.join(args.repo_root, PKG, up_name)
        dev_path = os.path.join(args.dev_root, dev_rel)
        if not os.path.exists(up_path):
            all_fails.append(f"{up_name}: MISSING upstream file {up_path}")
            print(f"  {up_name:<26} FAIL  missing upstream file")
            continue
        if not os.path.exists(dev_path):
            all_fails.append(f"{up_name}: MISSING dev source {dev_path}")
            print(f"  {up_name:<26} FAIL  missing dev source")
            continue
        fails, n_body, n_hu, n_hd = check_pair(up_path, dev_path, up_region,
                                               dev_region, up_name)
        all_fails += fails
        hdr = "verbatim (no header edit)" if up_region is None else \
            f"header {up_region[0]}-{up_region[1]} <- dev {dev_region[0]}-{dev_region[1]}" \
            f" ({n_hd}->{n_hu} lines)"
        print(f"  {up_name:<26} {'PASS' if not fails else 'FAIL'}  "
              f"{n_body} body lines identical; {hdr}")
        for f in fails:
            print(f"      {f}")

    for spec in EXTRACTIONS:
        name = spec["upstream"]
        up_sup = os.path.join(args.repo_root, PKG, name)
        src_sup = os.path.join(args.dev_root, spec["source"])
        if not os.path.exists(up_sup) or not os.path.exists(src_sup):
            all_fails.append(f"{name}: missing upstream file or source")
            print(f"  {name:<26} FAIL  missing file")
            continue
        fails, n_lines = check_extraction(up_sup, src_sup, spec)
        all_fails += fails
        print(f"  {name:<26} {'PASS' if not fails else 'FAIL'}  "
              f"{n_lines} extracted lines byte-identical to "
              f"{os.path.basename(spec['source'])} across "
              f"{len(spec['regions'])} regions")
        for f in fails:
            print(f"      {f}")

    print()
    print(f"C-1 COMPARATOR: {'ALL PASS' if not all_fails else f'FAIL ({len(all_fails)})'} "
          f"— {len(PAIRS) + len(EXTRACTIONS)} modules")
    return 0 if not all_fails else 1


if __name__ == "__main__":
    sys.exit(main())
