"""BinnedConfig + parse_binned_config — the independent second XML parse for the
optional ``<BinnedEngine>`` opt-in block.

Home: ``pynu.analysis_reader`` (S.F1 re-homing — this is a reader concern, a
sibling of ``ParseXML``; NOT absorbed into ``ParseXML`` itself, which is open
call O-2, RULED at Track T / T2: sibling module, reader-wired routing —
``ParseXML`` attaches ``BinnedConfigs`` from its own tree).

Mirrors ``PyNuFit._parse_marginalization_config``: a self-contained
``xml.etree`` pass that reads a tag the main ``ParseXML`` reader never touches
(``ParseXML.reader`` iterates only target/source/nuisance/fixed/physics
children), so every existing analysis XML parses to "toggle absent" -> ``{}``
and no forward-model code runs. This module imports only the standard library,
so it can be loaded without the heavy pynu import chain (nuSQuIDS, event MC).
"""
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass


@dataclass
class BinnedConfig:
    """Resolved ``<BinnedEngine>`` block for one experiment.

    response / tensors: filesystem paths (``${VAR}`` expanded at parse time).
    likelihood: 'poisson' (default, project convention since 2026-06-25) | 'bb'.
    migration:  'weighted' (default) | 'rawcount'.
    nuisance_spec: 'self' (default -> this XML's active dials) | a named engine
                   spec ('barr'/'R2'/'phased'/...) | an explicit .xml path.
    interp: 'nodes' (default, exact grid-node lookup) | 'cubic'.
    osc_averaging: provenance DECLARATION of the fast-oscillation averaging baked
                   into the tensor set ('off' default | '4pi' | a float). Load
                   path only records/validates it (tensors already carry it); the
                   phase-2 builder will consume it. See DESIGN §8.
    """
    response: str
    tensors: str
    likelihood: str = "poisson"
    migration: str = "weighted"
    nuisance_spec: object = "self"
    interp: str = "nodes"
    osc_averaging: str = "off"    # provenance declaration: off | 4pi | <float>


def _txt(elem, tag, default=None):
    """Stripped, ``${VAR}``-expanded text of a child tag, or ``default``."""
    child = elem.find(tag)
    if child is None or child.text is None:
        return default
    return os.path.expandvars(child.text.strip())


def parse_binned_config(xml_path):
    """Return ``{experiment_name: BinnedConfig}`` for every
    ``<NeutrinoExperiment>`` that carries an enabled ``<BinnedEngine>`` block.

    ``{}`` when no experiment opts in — the toggle-OFF default. A malformed or
    unreadable XML also yields ``{}`` (parse errors are the main parser's job to
    report); a *present* block missing ``<response>``/``<tensors>`` raises, since
    that is an explicit-but-broken opt-in.

    Track T / T2 (O-2 ruling): the analysis reader now attaches these configs
    itself (``ParseXML.BinnedConfigs``, from its already-parsed tree via
    ``parse_binned_config_root``) and PyNuFit consumes them from the reader —
    this path-based entry stays for standalone/tool use.
    """
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return {}
    return parse_binned_config_root(root)


def parse_binned_config_root(root):
    """``parse_binned_config`` on an already-parsed ElementTree root — the form
    the analysis reader uses (it owns the tree; no second file parse)."""
    configs = {}
    for exp in root.iter("NeutrinoExperiment"):
        block = exp.find("BinnedEngine")
        if block is None:
            continue
        status = block.find("status")
        if status is None or status.text is None or not int(status.text):
            continue
        name = exp.attrib.get("name")
        response = _txt(block, "response")
        tensors = _txt(block, "tensors")
        if not response or not tensors:
            raise ValueError(
                f"<BinnedEngine> for experiment {name!r} needs both "
                "<response> and <tensors>")
        configs[name] = BinnedConfig(
            response=response,
            tensors=tensors,
            likelihood=_txt(block, "likelihood", "poisson"),
            migration=_txt(block, "migration", "weighted"),
            nuisance_spec=_txt(block, "nuisance_spec", "self"),
            interp=_txt(block, "interp", "nodes"),
            osc_averaging=_txt(block, "osc_averaging", "off"),
        )
    return configs
