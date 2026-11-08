"""Pynu - Python Neutrino Fitting Framework.

The event framework (PyNuFit, Experiments, PhysicsTunes, analysis_reader) is used only to
build the binned fit inputs (see ``pynu.build``). The fit stage (``pynu.binned``) needs
numpy and scipy only, so nothing is imported at package load.
"""
