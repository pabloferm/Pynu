# Manages experiments

import sys
import os
import pathlib
import pandas as pd
import h5py
import numpy as np


# , dict_of_fixed, dict_of_true, dict_of_nominal):
def Manager(detector, source, dict_of_details, scenario):
    if 'SuperK' in detector:
        if source == 'Atmospheric':
            if 'Pheno' in detector:
                if 'Htag' in detector:
                    from .SuperK_Atm_Pheno import SuperK_Htag
                    return SuperK_Htag(dict_of_details, scenario)
                elif 'Gdtag' in detector:
                    from .SuperK_Atm_Pheno import SuperK_Gdtag
                    return SuperK_Gdtag(dict_of_details, scenario)
                else:
                    from .SuperK_Atm_Pheno import SuperK
                    return SuperK(dict_of_details, scenario)
            else:
                if 'SuperK_I' == detector:
                    return SuperK_I(dict_of_details, scenario)
                elif 'SuperK_II' == detector:
                    return SuperK_II(dict_of_details, scenario)
                elif 'SuperK_III' == detector:
                    return SuperK_III(dict_of_details, scenario)
                elif 'SuperK_IV' == detector:
                    from .SuperK_Atm_Official import SuperK_IV
                    return SuperK_IV(dict_of_details, scenario)
                elif 'SuperK_V' == detector:
                    return SuperK_V(dict_of_details, scenario)
                elif 'SuperK_VI' == detector:
                    return SuperK_VI(dict_of_details, scenario)
                else:
                    return SuperK_IV_noNtag(dict_of_details, scenario)
        elif source == 'Solar':
            pass
        elif source == 'Reactors':
            pass
        elif source in ['Accelerator', 'LBL', 'T2K']:
            pass
        else:
            sys.exit('Source not found for ' + detector)

    elif detector in ['HK', 'HyperK', 'Hyper-Kamiokande']:
        if source == 'Atmospheric':
            from .HyperK_Atm import HyperK_Atm
            return HyperK_Atm(dict_of_details, scenario)
        elif source == 'Solar':
            pass
        elif source == 'Reactors':
            pass
        elif source in ['Accelerator', 'LBL', 'T2K']:
            # from .HyperK.HyperK import HyperK_LBL
            # return HyperK_LBL(dict_of_details, scenario)
            pass
        else:
            sys.exit('Source not found for ' + detector + '!!')

    elif detector in ['ICUp', 'IceCube-Upgrade']:
        if source == 'Atmospheric':
            from .ICUp_Atm import ICUp_Atm
            return ICUp_Atm(dict_of_details, scenario)
        else:
            sys.exit('No valid source for ' + detector)
    else:
        sys.exit(
            f'Experiment not found!! \nPlease, include it at {os.path.dirname(os.path.abspath(__file__))}/MCReader.py .')


def reader(filename):
    extension = pathlib.Path(filename).suffix
    if extension == '.root':
        sys.exit(
            'Not there yet. Please go to utils/ and convert it to HD5F.\nSupported file types are HDF5 and csv')
    elif extension == '.HDF5' or extension == '.HDF' or extension == '.hdf' or extension == '.hdf5':
        fdata = {}
        with h5py.File(filename, 'r') as hf:
            for var in hf.keys():
                fdata[var] = np.array(hf[var])
    elif extension == '.csv':
        data = pd.read_csv(filename)
        fdata = {}
        for var in data:
            if int(pd.__version__[0]) > 0:
                fdata[var] = data[var].to_numpy()
            else:
                fdata[var] = np.array(data[var])
    else:
        sys.exit('Not supported data format, ' + extension)
    return fdata
