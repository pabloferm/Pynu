#!/usr/bin/env python
"""Merge individual grid point HDF5 files into a single results file."""
import h5py
import numpy as np
import os
import glob
from datetime import datetime

def merge_grid_results(input_dir="results/grid_points", output_file=None):
    if output_file is None:
        output_file = f"results/ORCA_NTOA_grid_{datetime.now().strftime('%m%d')}.hdf5"
    
    # Find all point files
    point_files = sorted(glob.glob(os.path.join(input_dir, "point_*.hdf5")),
                         key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    
    if not point_files:
        print(f"No point files found in {input_dir}")
        return
    
    n_points = len(point_files)
    print(f"Found {n_points} point files")
    
    # Create output file by copying structure from first file
    with h5py.File(output_file, 'w') as out_hf:
        with h5py.File(point_files[0], 'r') as hf0:
            # Copy entire structure
            for key in hf0.keys():
                hf0.copy(key, out_hf)
        
        # Now collect chi2 values from all files
        chi2_systs = np.zeros(n_points)
        chi2_stats = np.zeros(n_points)
        
        for i, pf in enumerate(point_files):
            point_idx = int(os.path.basename(pf).split('_')[1].split('.')[0])
            with h5py.File(pf, 'r') as hf:
                if 'Analysis/Chi2 Systs.' in hf:
                    chi2_systs[point_idx] = hf['Analysis/Chi2 Systs.'][point_idx]
                if 'Analysis/Chi2 Stats. Only' in hf:
                    chi2_stats[point_idx] = hf['Analysis/Chi2 Stats. Only'][point_idx]
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{n_points} files")
        
        # Write collected chi2 values
        del out_hf['Analysis/Chi2 Systs.']
        out_hf.create_dataset('Analysis/Chi2 Systs.', data=chi2_systs, compression='gzip')
        
        if 'Analysis/Chi2 Stats. Only' in out_hf:
            del out_hf['Analysis/Chi2 Stats. Only']
            out_hf.create_dataset('Analysis/Chi2 Stats. Only', data=chi2_stats, compression='gzip')
    
    print(f"Merged results saved to {output_file}")
    print(f"Chi2 range: [{chi2_systs.min():.4f}, {chi2_systs.max():.4f}]")
    print(f"Nonzero points: {np.count_nonzero(chi2_systs)}")
    return output_file

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='results/grid_points')
    parser.add_argument('-o', '--output', default=None)
    args = parser.parse_args()
    merge_grid_results(args.input, args.output)
