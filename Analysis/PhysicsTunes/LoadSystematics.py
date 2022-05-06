import pandas as pd
import numpy as np
from scipy.interpolate import griddata

def ICUp(Ebin, Zbin, syst, cut=[]):

    hp_nue_cc = pd.read_csv("Systematics/hyperplanes_nue_cc.csv")
    hp_numu_cc = pd.read_csv("Systematics/hyperplanes_numu_cc.csv")
    hp_nutau_cc = pd.read_csv("Systematics/hyperplanes_nutau_cc.csv")
    hp_nu_nc = pd.read_csv("Systematics/hyperplanes_all_nc.csv")

    method = 'nearest'

    grid_cz = hp_nue_cc['reco_coszen'].to_numpy()
    grid_Er = hp_nue_cc['reco_energy'].to_numpy()
    pid_nue = hp_nue_cc['pid'].to_numpy()
    ICSyst = [ "offset", "ice_absorption", "ice_scattering", "opt_eff_headon", "opt_eff_lateral", "opt_eff_overall", "coin_fraction"]
    if syst not in ICSyst:
        print('Systematic source not known for IC.')
    
    # for syst in ICSyst:
    values_nueCC = hp_nue_cc[syst].to_numpy()
    values_numuCC = hp_numu_cc[syst].to_numpy()
    values_nutauCC = hp_nutau_cc[syst].to_numpy()
    values_NC = hp_nu_nc[syst].to_numpy()

    dic = {}
    nuecc = np.array([])
    numucc = np.array([])
    nutaucc = np.array([])
    nc = np.array([])

    for sample in Ebin:
        c = pid_nue==sample
        points = np.array([grid_cz[c],grid_Er[c]]).T

        nue = values_nueCC[c]
        numu = values_numuCC[c]
        nutau = values_nutauCC[c]
        nuNC = values_NC[c]

        cz = Zbin[sample]
        e = Ebin[sample]
        zbin = cz[:-1] + np.diff(cz)/2
        ebin = e[:-1] + np.diff(e)/2

        newCz, newEr = np.meshgrid(zbin,ebin)

        nuecc = np.append(nuecc,griddata(points, nue, (newCz, newEr), method=method))
        numucc = np.append(numucc,griddata(points, numu, (newCz, newEr), method=method))
        nutaucc = np.append(nutaucc,griddata(points, nutau, (newCz, newEr), method=method))
        nc = np.append(nc,griddata(points, nuNC, (newCz, newEr), method=method))

    if len(cut)>0:
        dic['nueCC'] = nuecc.reshape(-1)[cut]
        dic['numuCC'] = numucc.reshape(-1)[cut]
        dic['nutauCC'] = nutaucc.reshape(-1)[cut]
        dic['NC'] = nc.reshape(-1)[cut]
    else:
        dic['nueCC'] = nuecc.reshape(-1)
        dic['numuCC'] = numucc.reshape(-1)
        dic['nutauCC'] = nutaucc.reshape(-1)
        dic['NC'] = nc.reshape(-1)

    return dic


# Erec_min = 1
# Erec_max = 1e4
# NErec = 40
# erec = np.logspace(np.log10(Erec_min), np.log10(Erec_max), NErec+1, endpoint = True)
# z10bins = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
# EnergyBins = {0:erec, 1:erec}
# CzBins = {0:z10bins, 1:z10bins}

# ICSyst = [ "offset", "ice_absorption", "ice_scattering", "opt_eff_headon", "opt_eff_lateral", "opt_eff_overall", "coin_fraction"]

# print(ICUp(EnergyBins, CzBins, ICSyst[3]))

