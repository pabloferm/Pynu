import uproot as upt
import h5py
import numpy as np
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('fname', type=str, nargs='?', default='NULL')
parser.add_argument('treename', type=str, nargs='?', default='gst')
args = parser.parse_args()
infile = args.fname
tree = args.treename

if infile == 'NULL':
    sys.exit('Hey! You forgot to introduce your GENIE file')
print('Converting ', tree, ' ROOT tree to HDF5 file.')

f = upt.open(infile)
keep_columns = f[tree].keys()
print('Tree variables to be stored in ',infile,'.hdf5')
print(keep_columns)

dt = h5py.special_dtype(vlen=np.float32)

with h5py.File(infile+'.hdf5', 'w') as hf:
    for i,br in enumerate(keep_columns):
        dummy = f.get('gst')[br].array(interpretation=None, entry_start=None, entry_stop=None, decompression_executor=None, interpretation_executor=None, array_cache='inherit', library='np')
        dummy = np.array(dummy)
        if dummy.dtype==np.dtype('object'):
            # Special treatment for variable length datasets
            dt = h5py.special_dtype(vlen=np.float64)
            hf.create_dataset(br, data=dummy, dtype=dt, compression='gzip')
        else:
            hf.create_dataset(br, data=dummy, compression='gzip')


