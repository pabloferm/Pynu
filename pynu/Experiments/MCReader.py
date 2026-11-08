# Manages experiments

import sys
import pathlib
import pandas as pd
import h5py
import numpy as np


SUPPORTED = ("SuperK_pheno_2023", "IceCube-DeepCore", "ORCA")


def Manager(detector, source, dict_of_details):
    """Experiment factory for the three experiments of this analysis; any other
    detector or source raises."""
    if source == "Atmospheric":
        if detector == "SuperK_pheno_2023":
            from .SuperK_Atm_Pheno import SuperK_2023

            return SuperK_2023(dict_of_details)

        elif detector == "ORCA":
            from .Orca import ORCA_Atm

            return ORCA_Atm(dict_of_details)

        elif detector == "IceCube-DeepCore":
            from .ICDeepCore import ICDeepCore_Atm

            return ICDeepCore_Atm(dict_of_details)

    raise ValueError(
        f"Experiment {detector!r} with source {source!r} is not part of this release "
        f"(supported, Atmospheric source only: {', '.join(SUPPORTED)})."
    )


def reader(filename):
    extension = pathlib.Path(filename).suffix
    if extension in {".HDF5", ".HDF", ".hdf", ".hdf5", ".h5"}:
        fdata = {}
        with h5py.File(filename, "r") as hf:
            for var in hf.keys():
                fdata[var] = np.array(hf[var])
    elif extension == ".csv":
        data = pd.read_csv(filename)
        fdata = {var: data[var].to_numpy() for var in data}
    else:
        sys.exit(f"Not supported data format, {extension}")
    return fdata
