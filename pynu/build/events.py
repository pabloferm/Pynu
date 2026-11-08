"""Event-XML templates -> a constructed PyNuFit.

The templates in ``configs/events/`` carry ``${NAME}`` placeholders for their input
files (``string.Template`` syntax). ``render_event_xml`` substitutes them and fails on
any placeholder left unfilled.
"""
import os
import string
import tempfile


def render_event_xml(template_path, values, out_dir):
    """Write the rendered template to ``out_dir`` and return the rendered path. Paths in
    ``values`` are made absolute."""
    with open(template_path) as fh:
        text = fh.read()
    mapping = {k: os.path.abspath(str(v)) for k, v in values.items()}
    rendered = string.Template(text).substitute(mapping)   # KeyError if unfilled
    out = os.path.join(out_dir, os.path.basename(template_path))
    with open(out, "w") as fh:
        fh.write(rendered)
    return out


def load_pynufit(template_path, values):
    """Construct PyNuFit from a rendered template. Returns
    (pynufit, experiment_key, experiment, physics_tunes) for the single
    experiment an event XML defines."""
    from pynu.PyNuFit import PyNuFit

    with tempfile.TemporaryDirectory(prefix="pynu_events_") as tmp:
        xml = render_event_xml(template_path, values, out_dir=tmp)
        pynufit = PyNuFit(xml)
    keys = list(pynufit.Experiments.keys())
    if len(keys) != 1:
        raise RuntimeError(f"{template_path}: expected one experiment, got {keys}")
    k = keys[0]
    return pynufit, k, pynufit.Experiments[k], pynufit.physics_tunes[k]


def md5sum(path):
    import hashlib

    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()
