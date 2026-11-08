"""Build stage: produce every input the binned fit (``pynu.binned``) reads.

One module per product; each exposes plain functions (paths and numbers as
arguments, no environment variables) that ``scripts/build.py`` wraps.

  events       render a ``configs/events/*.xml`` template and construct PyNuFit
  sk_response  SK COO response from the event-by-event MC        (sk-response)
  sk_pin       anchor counts + per-species pin -> Ninv            (sk-pin)
  ic_response  IC DeepCore COO response, 160 x 320, |mode| axis   (ic-response)
  orca_response ORCA COO response, flat900 layout                 (orca-response)
  fluxq        per-true-E flux-ratio q tables                      (fluxq)
  aux          IC / ORCA observed, muon, NORM, few, hypersurfaces  (aux)
  osc          oscillated-flux tensor helpers + per-cell builders  (tensors)

Nothing is imported here: the event framework (nuSQuIDS, nuflux) is loaded only
by the modules that need it.
"""
