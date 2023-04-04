import AnalysisReader as AR  # contains parse class to read and setup the analysis
import h5py
import numpy as np
import os
import sys

import matplotlib.pyplot as plt
plt.style.use(os.environ['PYNU'] + '/plot.mplstyle')


class Plot:

    def __init__(
            self,
            analysis_output_file,
            analysis_input_file=False,
            directory=None):
        ''' Set up basic analysis variables and structure to build full analysis '''
        if analysis_input_file:
            self.AnalysisInput = AR.parse(analysis_file, check=self.verbosity)

        with h5py.File(analysis_output_file, 'r') as hf:

            self.X2_stats = np.array(hf['Analysis/Chi2 Stats. Only'])
            self.X2 = np.array(hf['Analysis/Chi2 Systs.'])

            self.Physics = {}
            self.PhysicsFlat = {}
            for source in hf['Physics Parameters']:
                self.Physics[source] = {}
                for item, dset in hf['Physics Parameters/' + source].items():
                    self.Physics[source][item] = np.array(dset)
                    self.PhysicsFlat[source + '+' + item] = np.array(dset)
            self.NumberOfPhysicsPars = len(self.PhysicsFlat)

            self.Nuisance = {}
            for source in hf['Nuisance Parameters']:
                self.Nuisance[source] = {}
                for item, dset in hf['Nuisance Parameters/' + source].items():
                    self.Nuisance[source][item] = np.array(dset)

            self.Fixed = {}
            for source in hf['Fixed Parameters']:
                self.Fixed[source] = {}
                for item, dset in hf['Fixed Parameters/' + source].items():
                    self.Fixed[source][item] = np.array(dset)

        self.Plot1D(also_stats_only=True)

    def Plot1D(self, all_plots=True, also_stats_only=False):
        if all_plots:
            if self.NumberOfPhysicsPars < 4:
                fig, ax = plt.subplots(
                    nrows=1, ncols=self.NumberOfPhysicsPars, figsize=(
                        12, 36))
                axis = [ax]
            else:
                ncols = 3
                nrows = self.NumberOfPhysicsPars // ncols + 1
                fig, ax = plt.subplots(
                    nrows=nrows, ncols=ncols, figsize=(
                        6 * ncols, 6 * nrows))
                axis = ax.flat

            fig.tight_layout(h_pad=5)

            for i, (item, values) in enumerate(self.PhysicsFlat.items()):
                x = np.unique(values)
                y = np.zeros_like(x)
                print(item)
                for j, t in enumerate(x):
                    y[j] = np.amin(self.X2[values == t])
                axis[i].plot(x, y, label='w/ systemtics')
                if also_stats_only:
                    for j, t in enumerate(x):
                        y[j] = np.amin(self.X2_stats[values == t])
                    axis[i].plot(x, y, linewidth=0.5, label='Stats. only')
                # axis[i].set_ylim(0,25)
                axis[i].set_xlabel(self.Format(item))
                axis[i].set_ylabel(r'$\chi^2$')
                axis[i].legend()

            plt.tight_layout()
            plt.show()

    def Plot2D(self, all_plots=True, also_stats_only=False):
        if self.NumberOfPhysicsPars < 2:
            sys.exit('Cannot make 2D contour plots with 1 physics parameter')

        elif self.NumberOfPhysicsPars == 2:
            fig, ax = plt.subplots(nrows=1, ncols=1)
            axis = [ax]

        else:
            ncols = nrows = self.NumberOfPhysicsPars - 1
            fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
            axis = ax.flat

    def Format(self, string):
        new_string = None
        if 'Sin' in string:
            if 'Sin2' in string:
                new_string = r'$\sin^2$'
            else:
                new_string = r'$\sin$'

        if 'Theta' in string:
            new_string += r'$\theta_{' + string[-2] + string[-1] + '}$'

        if 'Eps' in string:
            new_string += r'$\epsilon{' + string[-2] + string[-1] + '}$'

        # print(new_string)

        if new_string is None:
            return string
        return new_string
