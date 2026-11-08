"""Binned likelihood fits of Super-Kamiokande, IceCube-DeepCore and KM3NeT/ORCA6.

    config     read a scan configuration XML
    tensors    read one lattice cell's oscillation tensors
    sk_engine  Super-Kamiokande engine (response matrix x oscillation x systematics)
    ic_engine  IceCube-DeepCore engine (+ ic_cells, ic_dials)
    orca_engine  KM3NeT/ORCA6 engine
    dial_fields, contract  flux / cross-section parameter fields and contractions
               shared by the IC and ORCA engines
    arms       the three experiments as fit arms; single-experiment fits
    combined   the combined SK + IC + ORCA fit
    scan       command line: run / check / assemble a lattice scan

Only numpy and scipy are needed here; nothing in this package imports the PyNu
event framework.
"""
