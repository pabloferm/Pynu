import AnalysisReader as AR  # contains parse class to read and setup the analysis
import h5py
import numpy as np
import os
import sys
from scipy.interpolate import interp1d, interp2d
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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

        self.levels_2d = (4.61, 9.21)
        self.levels_txt_2d = {4.61:r'$90\%$',9.21:r'$99\%$'}
        # self.colors = ('darkblue','orange','limegreen','mediumvioletred','crimson')
        self.colors_2d = ('orange', 'orange')
        self.lines_2d = ('dashed','solid')
        self.linewidths_2d = (1,1)

        self.levels_1d = [1,6.63,9, 16]
        self.levels_txt_1d = {1:r'$1\sigma$',6.63:r'$99\%$',9:r'$3\sigma$', 16:r'$4\sigma$'}

        # self.levels_1d = [1,2.71,4,6.63,9]
        # self.levels_txt_1d = {1:r'$1\sigma$',2.71:r'$90\%$',4:r'$2\sigma$',6.63:r'$99\%$',9:r'$3\sigma$'}

        # self.Plot1D(also_stats_only=True)
        # self.Plot2D()
        self.ResultPlotsMatrix()

    def ResultPlotsMatrix(self, interpolate=True):
        fig, ax = plt.subplots(
                    nrows=self.NumberOfPhysicsPars, 
                    ncols=self.NumberOfPhysicsPars, 
                    figsize=(15, 15))

        for i, (item_row, values_row) in enumerate(self.PhysicsFlat.items()):
            for j, (item_col, values_col) in enumerate(self.PhysicsFlat.items()):
                if i == j: # Diagonal 1D plots
                    x = np.unique(values_col)
                    y = np.zeros_like(x)
                    for k, t in enumerate(x):
                        y[k] = np.amin(self.X2[values_col == t])
                    if interpolate:
                        spl = interp1d(x, y, kind='quadratic')
                        x_dense = np.linspace(np.amin(x), np.amax(x), 10*x.size)
                        y_dense = spl(x_dense)
                        y_dense -= np.amin(y_dense)
                        ax[i,j].plot(x_dense, y_dense, label='w/ systemtics')
                    else:
                        y -= np.amin(y)
                        ax[i,j].plot(x, y, label='w/ systemtics')
                    
                    ax[i,j].set_ylabel(r'$\chi^2$', fontsize=12)
                    ax[i,j].set_xlabel(self.Format(item_col), fontsize=12)
                    ax[i,j].set_ylim(0,25)
                    ax[i,j].tick_params(axis='both', labelsize=12)
                    axmin,axmax = ax[i,j].get_xlim()
                    for lv in self.levels_1d:
                    	ax[i,j].axhline(y=lv, color='grey', linestyle='--', alpha=0.4, linewidth=0.7)
                    	# ax[i,j].text(0.15*axmin+0.85*axmax, lv, levels_txt[lv])
                    	ax[i,j].text(0.95*axmin+0.05*axmax, lv, self.levels_txt_1d[lv], fontsize=10)


                elif i>j: # Off-diagonal 2D plots
                    x = np.unique(values_col)
                    y = np.unique(values_row)
                    w = np.array([])
                    for kk,dx in enumerate(x):
                        for ll,dy in enumerate(y):
                            cut = np.logical_and(values_col==dx,values_row==dy)
                            w = np.append(w,np.amin(self.X2[cut]))
                    X, Y = np.meshgrid(x, y)
                    f = interp2d(x, y, w, kind='cubic')
                    Chi2 = np.reshape(w, (x.size,y.size)).T
                    if interpolate:
                    	x_dense = np.linspace(np.amin(x),np.amax(x),50)
                    	y_dense = np.linspace(np.amin(y),np.amax(y),50)
                    	newX, newY = np.meshgrid(x_dense, y_dense)
                    	newChi2 = f(x_dense, y_dense)
                    	CS = ax[i,j].contour(newX,newY,newChi2, colors=self.colors_2d, levels=self.levels_2d, linestyles=self.lines_2d, linewidths=self.linewidths_2d)

                    ax[i,j].set_xlabel(self.Format(item_col), fontsize=12)
                    ax[i,j].set_ylabel(self.Format(item_row), fontsize=12)
                    ax[i,j].tick_params(axis='both', labelsize=12)

        # fig.subplots_adjust(hspace=0.5, wspace=0.5)
        fig.tight_layout()
        plt.tight_layout()
        plt.show()



    def Plot1D(self, all_plots=True, also_stats_only=False, interpolate=True):
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
                if interpolate:
                    spl = interp1d(x, y, kind='quadratic')
                    x_dense = np.linspace(np.amin(x), np.amax(x), 10*x.size)
                    y_dense = spl(x_dense)
                    axis[i].plot(x_dense, y_dense, label='w/ systemtics')
                else:
                    axis[i].plot(x, y, label='w/ systemtics')
                if also_stats_only:
                    for j, t in enumerate(x):
                        y[j] = np.amin(self.X2_stats[values == t])
                    if interpolate:
                        spl = interp1d(x, y, kind='quadratic')
                        x_dense = np.linspace(np.amin(x), np.amax(x), 10*x.size)
                        y_dense = spl(x_dense)
                        axis[i].plot(x_dense, y_dense, linewidth=0.5, label='Stats. only')
                    else:
                        axis[i].plot(x, y, linewidth=0.5, label='Stats. only')
                axis[i].set_ylim(0,25)
                axis[i].set_xlabel(self.Format(item))
                axis[i].set_ylabel(r'$\chi^2$')
                axis[i].legend()

            plt.tight_layout()
            plt.show()

    def Plot2D(self, all_plots=True, also_stats_only=False, interpolate=True):
        if self.NumberOfPhysicsPars < 2:
            sys.exit('Cannot make 2D contour plots with 1 physics parameter')

        else:
            ncols = nrows = self.NumberOfPhysicsPars - 1
            fig = plt.figure()
            gs = GridSpec(nrows-1, ncols-1)
            ax = [[0]*(nrows-1)]*(ncols-1)
            ax[nrows-2][0] = plt.subplot(gs[nrows-2,0])
            for i in range(0, nrows-2):
                ax[i][0] = plt.subplot(gs[i,0], sharey = ax[nrows-2][0])
            for j in range(1, ncols-1):
            	ax[nrows-2][j] = plt.subplot(gs[nrows-2,j], sharex = ax[nrows-2][0])


                # for j in range(ncols):
                #     if i > j:
                #         if i==nrows-1 and j==0:
                #             ax[i][j] = plt.subplot(gs[i,j])
                #         elif i < nrows-1 and j==0:
                #             ax[i][j] = plt.subplot(gs[i,j], sharex = ax[nrows-1][j])
                #         elif i == nrows-1 and j>0:
                #             ax[i][j] = plt.subplot(gs[i,j], sharey = ax[i][0])
                        # ax[i][j] = fig.add_subplot(gs[i,j])

            # for i in range(ncols):
            #     for j in range(nrows):
            #         if i > j:
            #             ax[i][j] = fig.add_subplot(gs[i,j])

            # ax_scatter = fig.add_subplot(gs[1:4, 0:3])
            # ax_hist_y = fig.add_subplot(gs[0,0:3])
            # ax_hist_x = fig.add_subplot(gs[1:4, 3])

            fig.set_constrained_layout_pads(hspace=0.0, h_pad=0.0)
            plt.subplots_adjust(hspace=0.00)
            fig.tight_layout()
            plt.show()


    def Format(self, string):
        new_string = ''
        if 'Sin' in string:
            if 'Sin2' in string:
                new_string = r'$\sin^2$'
            else:
                new_string = r'$\sin$'

        if 'Theta' in string:
            new_string += r'$\theta_{' + string[-2] + string[-1] + '}$'

        if 'Eps' in string:
            new_string += r'$\epsilon{' + string[-2] + string[-1] + '}$'

        if 'Dm' in string:
            new_string += r'$\Delta m^2_{' + string[-2] + string[-1] + '}$'

        # print(new_string)

        if new_string == '':
            return string
        return new_string
