import h5py
import argparse
import sys
import os
import subprocess
import numpy as np
import math

parser = argparse.ArgumentParser()
parser.add_argument('in_dir', type=str, nargs='?', default='NULL')
parser.add_argument('outfile', type=str, nargs='?', default='NULL')
args = parser.parse_args()
path = args.in_dir
outfilename = args.outfile

if path == 'NULL':
    sys.exit('Please introduce the path to your HDF5 files.')

path = os.path.abspath(path)

lst = os.listdir(path)
ipnu= np.array([], dtype=np.double)
pnu = np.array([], dtype=np.double)
dirnuX = np.array([], dtype=np.double)
dirnuY = np.array([], dtype=np.double)
dirnuZ = np.array([], dtype=np.double)
cz  = np.array([], dtype=np.double)
azi = np.array([], dtype=np.double)
fluxho_numu = np.array([], dtype=np.double)
fluxho_nue  = np.array([], dtype=np.double)
fluxho_numub= np.array([], dtype=np.double)
fluxho_nueb = np.array([], dtype=np.double)
weightOsc_SKpaper   = np.array([], dtype=np.double)
weightOsc_SKbest   = np.array([], dtype=np.double)
weightSim   = np.array([], dtype=np.double)
weightReco   = np.array([], dtype=np.double)
plep= np.array([], dtype=np.double)
dirlepX= np.array([], dtype=np.double)
dirlepY= np.array([], dtype=np.double)
dirlepZ= np.array([], dtype=np.double)
## Reco variables
pmax   = np.array([], dtype=np.double)
evis= np.array([], dtype=np.double)
recodirX  = np.array([], dtype=np.double)
recodirY  = np.array([], dtype=np.double)
recodirZ  = np.array([], dtype=np.double)
ip  = np.array([], dtype=np.double)
nring   = np.array([], dtype=np.double)
muedk   = np.array([], dtype=np.double)
neutron = np.array([], dtype=np.double)
itype   = np.array([], dtype=np.double)
otype   = np.array([], dtype=np.double)
imass   = np.array([], dtype=np.double)
mode= np.array([], dtype=np.double)
variables = {'ipnu':ipnu, 'pnu':pnu, 'dirnuX':dirnuX, 'dirnuY':dirnuY, 'dirnuZ':dirnuZ, 'azi':azi, 
'cz':cz, 'fluxho_nue':fluxho_nue, 'fluxho_nueb':fluxho_nueb, 'fluxho_numu':fluxho_numu, 'fluxho_numub':fluxho_numub,
'plep':plep, 'dirlepX':dirlepX, 'dirlepY':dirlepY, 'dirlepZ':dirlepZ, 'pmax':pmax, 'evis':evis, 'itype':itype, 'otype':otype,
'recodirX':recodirX, 'recodirY':recodirY, 'recodirZ':recodirZ, 'ip':ip, 'nring':nring, 'muedk':muedk,
'neutron':neutron, 'imass':imass, 'mode':mode, 'weightSim':weightSim, 'weightOsc_SKpaper':weightOsc_SKpaper,
'weightOsc_SKbest':weightOsc_SKbest, 'weightReco':weightReco}

for i, file in enumerate(lst):
    if file.endswith(".hdf5"):
        full = os.path.join(path,file)
        with h5py.File(full, 'r') as hf:
            for key in hf.keys():
                j = str(key)
                ds = hf[key]
                data = np.array(ds[()])
                variables[j] = np.append(variables[j],data)
                if j == 'weightOsc_SKbest':
                    for x in data:
                        if math.isnan(x):
                            print('hey')

with h5py.File(os.path.join(path,'combined.hdf5'), 'w') as ff:
    for var in variables:
        ff.create_dataset(var, data=variables[var], compression='gzip')
print('Written output to ',os.path.join(path,'combined.hdf5'))

