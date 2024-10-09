# Manages experiments

import sys
import os
import pathlib
import pandas as pd
import h5py
import numpy as np


# , dict_of_fixed, dict_of_true, dict_of_nominal):
def Manager(detector, source, dict_of_details, scenario):
    if "SuperK" in detector:
        if source == "Atmospheric":
            if "pheno" in detector.lower():
                if "htag" in detector.lower():
                    from .SuperK_Atm_Pheno import SuperK_Htag

                    return SuperK_Htag(dict_of_details, scenario)
                elif "gdtag" in detector.lower():
                    from .SuperK_Atm_Pheno import SuperK_Gdtag

                    return SuperK_Gdtag(dict_of_details, scenario)
                elif "2023" in detector:
                    from .SuperK_Atm_Pheno import SuperK_2023

                    return SuperK_2023(dict_of_details, scenario)
                else:
                    from .SuperK_Atm_Pheno import SuperK

                    return SuperK(dict_of_details, scenario)
            elif detector == "SuperK_I":
                return SuperK_I(dict_of_details, scenario)
            elif detector == "SuperK_II":
                return SuperK_II(dict_of_details, scenario)
            elif detector == "SuperK_III":
                return SuperK_III(dict_of_details, scenario)
            elif detector == "SuperK_IV":
                from .SuperK_Atm_Official import SuperK_IV

                return SuperK_IV(dict_of_details, scenario)
            elif detector == "SuperK_V":
                return SuperK_V(dict_of_details, scenario)
            elif detector == "SuperK_VI":
                return SuperK_VI(dict_of_details, scenario)
            else:
                return SuperK_IV_noNtag(dict_of_details, scenario)
        elif source == "Solar":
            pass
        elif source == "Reactors":
            pass
        elif source not in ["Accelerator", "LBL", "T2K"]:
            sys.exit("SOURCE not found for " + detector)

    elif "HyperK" in detector:
        if source == "Atmospheric":
            if "Pheno" in detector:
                from .HyperK_from_SuperK_Atm_Pheno import HyperK

                return HyperK(dict_of_details, scenario)
            else:
                if "NoNeutron" in detector:
                    from .HyperK_from_SuperK_Atm_Official import HyperK_NoNeutron

                    return HyperK_NoNeutron(dict_of_details, scenario)
                else:
                    from .HyperK_from_SuperK_Atm_Official import HyperK

                    return HyperK(dict_of_details, scenario)
        elif source == "Solar":
            pass
        elif source == "Reactors":
            pass
        elif source not in ["Accelerator", "LBL", "T2K"]:
            sys.exit("Source not found for " + detector + "!!")

    elif detector in ["ICUp", "IceCube-Upgrade"]:
        if source == "Atmospheric":
            from .IceCube import ICUp_Atm

            return ICUp_Atm(dict_of_details, scenario)
        else:
            sys.exit("No valid source for " + detector)

    elif detector == "DeepCore":
        if source == "Atmospheric":
            from .IceCube import DeepCore

            return DeepCore(dict_of_details, scenario)
        else:
            sys.exit("No valid source for " + detector)

    elif 'IceCube-2017' in detector:
        if source == 'Atmospheric':
            from .IC2017 import IC2017
            return IC2017(dict_of_details, scenario)
        else:
            sys.exit('No valid source for ' + detector)

    elif detector == "ORCA":
        if source == "Atmospheric":
            from .Orca import Orca

            return Orca(dict_of_details, scenario)
        else:
            sys.exit("No valid source for " + detector)

    else:
        sys.exit(
            f"Experiment not found!! \nPlease, include it at {os.path.dirname(os.path.abspath(__file__))}/MCReader.py ."
        )


# Project to improve Manager
'''
def import_and_return(module_path, class_name, dict_of_details, scenario):
    """Dynamically imports the module and returns an instance of the class."""
    module = __import__(module_path, fromlist=[class_name])
    class_ref = getattr(module, class_name)
    return class_ref(dict_of_details, scenario)

def check_valid_source(source, detector):
    """Check if the source is valid for the given detector."""
    valid_sources = ["Atmospheric", "Solar", "Reactors", "Accelerator", "LBL", "T2K"]
    if source not in valid_sources:
        sys.exit(f"Source not found for {detector}!")

def Manager(detector, source, dict_of_details, scenario):
    # Map detectors to their respective handling modules and class names
    detector_config = {
        "SuperK": {
            "Atmospheric": {
                "pheno": {
                    "htag": ("SuperK_Atm_Pheno", "SuperK_Htag"),
                    "gdtag": ("SuperK_Atm_Pheno", "SuperK_Gdtag"),
                    "2023": ("SuperK_Atm_Pheno", "SuperK_2023"),
                    "default": ("SuperK_Atm_Pheno", "SuperK"),
                },
                "official": {
                    "SuperK_IV": ("SuperK_Atm_Official", "SuperK_IV"),
                },
                "stages": {
                    "SuperK_I": "SuperK_I",
                    "SuperK_II": "SuperK_II",
                    "SuperK_III": "SuperK_III",
                    "SuperK_V": "SuperK_V",
                    "SuperK_VI": "SuperK_VI",
                    "default": ("SuperK_Atm_Official", "SuperK_IV_noNtag"),
                },
            },
        },
        "HyperK": {
            "Atmospheric": {
                "pheno": ("HyperK_from_SuperK_Atm_Pheno", "HyperK"),
                "official": {
                    "NoNeutron": ("HyperK_from_SuperK_Atm_Official", "HyperK_NoNeutron"),
                    "default": ("HyperK_from_SuperK_Atm_Official", "HyperK"),
                },
            },
        },
        "ICUp": {
            "Atmospheric": ("IceCube", "ICUp_Atm"),
        },
        "IceCube-Upgrade": {
            "Atmospheric": ("IceCube", "ICUp_Atm"),
        },
        "DeepCore": {
            "Atmospheric": ("IceCube", "DeepCore"),
        },
        "ORCA": {
            "Atmospheric": ("Orca", "Orca"),
        },
    }

    # Check for SuperK detectors and map based on source and subtypes
    if "SuperK" in detector:
        if source == "Atmospheric":
            detector_lower = detector.lower()
            if "pheno" in detector_lower:
                subconfig = detector_config["SuperK"]["Atmospheric"]["pheno"]
                key = next((k for k in subconfig if k in detector_lower), "default")
                module, class_name = subconfig[key]
                return import_and_return(module, class_name, dict_of_details, scenario)
            else:
                stages = detector_config["SuperK"]["Atmospheric"]["stages"]
                key = detector if detector in stages else "default"
                class_name = stages[key]
                if isinstance(class_name, tuple):
                    module, class_name = class_name
                    return import_and_return(module, class_name, dict_of_details, scenario)
                else:
                    return globals()[class_name](dict_of_details, scenario)
        check_valid_source(source, detector)
    
    # HyperK handling
    elif "HyperK" in detector:
        if source == "Atmospheric":
            detector_lower = detector.lower()
            if "pheno" in detector:
                module, class_name = detector_config["HyperK"]["Atmospheric"]["pheno"]
                return import_and_return(module, class_name, dict_of_details, scenario)
            else:
                subconfig = detector_config["HyperK"]["Atmospheric"]["official"]
                key = "NoNeutron" if "noneutron" in detector_lower else "default"
                module, class_name = subconfig[key]
                return import_and_return(module, class_name, dict_of_details, scenario)
        check_valid_source(source, detector)

    # ICUp/IceCube-Upgrade handling
    elif detector in ["ICUp", "IceCube-Upgrade"]:
        if source == "Atmospheric":
            module, class_name = detector_config[detector]["Atmospheric"]
            return import_and_return(module, class_name, dict_of_details, scenario)
        sys.exit(f"No valid source for {detector}")

    # DeepCore handling
    elif detector == "DeepCore":
        if source == "Atmospheric":
            module, class_name = detector_config["DeepCore"]["Atmospheric"]
            return import_and_return(module, class_name, dict_of_details, scenario)
        sys.exit(f"No valid source for {detector}")

    # ORCA handling
    elif detector == "ORCA":
        if source == "Atmospheric":
            module, class_name = detector_config["ORCA"]["Atmospheric"]
            return import_and_return(module, class_name, dict_of_details, scenario)
        sys.exit(f"No valid source for {detector}")

    # Catch-all for missing experiments
    else:
        sys.exit(f"Experiment not found!!\nPlease, include it in {os.path.dirname(os.path.abspath(__file__))}/MCReader.py .")

'''




def reader(filename):
    extension = pathlib.Path(filename).suffix
    if extension == ".root":
        sys.exit(
            "Not there yet. Please go to utils/ and convert it to HD5F.\nSupported file types are HDF5 and csv"
        )
    elif extension in {".HDF5", ".HDF", ".hdf", ".hdf5", ".h5"}:
        fdata = {}
        with h5py.File(filename, "r") as hf:
            for var in hf.keys():
                fdata[var] = np.array(hf[var])
    elif extension == ".csv":
        data = pd.read_csv(filename)
        fdata = {
            var: (
                data[var].to_numpy()
                if int(pd.__version__[0]) > 0
                else np.array(data[var])
            )
            for var in data
        }
    else:
        sys.exit(f"Not supported data format, {extension}")
    return fdata
