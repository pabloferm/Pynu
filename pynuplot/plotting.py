import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib.gridspec import GridSpec
import math
from scipy.interpolate import make_interp_spline
from scipy import interpolate
from plots.OfficialContours.Results import results


def cornerPlot(x,y,X2,title=''):
	# Data handling
	x0 = np.amin(x)
	xf = np.amax(x)
	y0 = np.amin(y)
	yf = np.amax(y)

	minChi2 = np.amin(X2)
	print(f'Minimum X2 {minChi2}')
	X2 = X2 - minChi2

	check23 = False
	check31 = False

	Chi2 = np.reshape(X2, (x.size,y.size)).T

	X2_x = np.array([])
	X2_y = np.array([])
	for i,t in enumerate(y):
		marg = np.amin(Chi2[i,0:x.size])
		X2_y = np.append(X2_y,marg)


	for i,t in enumerate(x):
		marg = np.amin(Chi2[0:y.size,i])
		X2_x = np.append(X2_x,marg)

	# Parameters for plots
	f = plt.figure()
	gs = GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3], wspace=0.06, hspace=0.1)

	axUp = plt.subplot(gs[0])
	ax = plt.subplot(gs[2])
	axRi = plt.subplot(gs[3])

	nullfmt = NullFormatter()
	axUp.xaxis.set_major_formatter(nullfmt)
	axRi.yaxis.set_major_formatter(nullfmt)
	axUp.tick_params(bottom=False,left=False,right=True)
	axRi.tick_params(left=False,bottom=False,top=True)
	axUp.yaxis.set_label_position("right")
	axUp.yaxis.tick_right()
	axUp.set_ylabel(r"$\chi^2$")
	axRi.xaxis.set_label_position("top")
	axRi.xaxis.tick_top()
	axRi.set_xlabel(r"$\chi^2$")
	if abs(x0+xf-1)<0.1:
		ax.set_xlabel(r'$\sin^2\theta_{23}$')
		check23=True
	elif abs(x0+xf)<0.01:
		ax.set_xlabel(r'$\Delta m^2_{31}$')
	elif abs(x0+xf-2*math.pi)<0.1:
		ax.set_xlabel(r'$\delta_{CP}$')
	else:
		ax.set_xlabel(r'$\sin^2\theta_{13}$')
	if abs(y0+yf-1)<0.1:
		ax.set_ylabel(r'$\sin^2\theta_{23}$')
	elif abs(y0+yf)<0.01:
		ax.set_ylabel(r'$\Delta m^2_{31}$')
		check31=True
	elif abs(y0+yf-2*math.pi)<0.1:
		ax.set_ylabel(r'$\delta_{CP}$')
	else:
		ax.set_ylabel(r'$\sin^2\theta_{13}$')

	colors = ['c','g','r','m']

	levels1d = np.array([1,4,9.0, 16.0, 25.0]) # 1, 2, 3, 4, 5 sigma
	levels2d = np.array([2.3,4.61,6.18,11.53]) # 1, 90%, 2, 3 sigma

	# Put data into plots
	axUp.plot(x,X2_x, color='k')
	for i in range(2):
		axUp.plot([x[0],x[-1]],[levels1d[i],levels1d[i]], color=colors[i], linewidth=0.5)
		axRi.plot([levels1d[i],levels1d[i]], [y[0],y[-1]],color=colors[i], linewidth=0.5)
	axUp.set_ylim(0.,20.)
	axUp.set_xlim(x0,xf)
	axRi.plot(X2_y, y, color='k')
	axRi.set_xlim(0.,100.)
	axRi.set_ylim(y0,yf)

	X, Y = np.meshgrid(x, y)
	levels = np.array([4.605,5.991,9.21])
	ax.contour(X,Y,Chi2, levels=levels2d, colors=colors)
	ax.set_xlim(0.9*x0,1.1*xf)
	ax.set_ylim(0.9*y0,1.1*yf)

	axUp.set_title(title)

	if check23 and check31:
		cont = results()
		cont.SK2017_theta23_dm31()
		cont.SK2020_theta23_dm31()
		cont.plot(ax)

	plt.show()


def cornerPlotBothO(x,y,X2_N,X2_I):
	# Data handling
	x0 = np.amin(x)
	xf = np.amax(x)
	y0 = np.amin(y)
	yf = np.amax(y)

	Chi2N = np.reshape(X2_N, (x.size,y.size)).T
	Chi2I = np.reshape(X2_I, (x.size,y.size)).T

	X2N_x = np.array([])
	X2N_y = np.array([])
	X2I_x = np.array([])
	X2I_y = np.array([])

	for i,t in enumerate(y):
		marg = np.amin(Chi2N[i,0:x.size])
		X2N_y = np.append(X2N_y,marg)
		marg = np.amin(Chi2I[i,0:x.size])
		X2I_y = np.append(X2I_y,marg)

	for i,t in enumerate(x):
		marg = np.amin(Chi2N[0:y.size,i])
		X2N_x = np.append(X2N_x,marg)
		marg = np.amin(Chi2I[0:y.size,i])
		X2I_x = np.append(X2I_x,marg)

	f = plt.figure()
	gs = GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3], wspace=0.06, hspace=0.1)


	axUp = plt.subplot(gs[0])
	ax = plt.subplot(gs[2])
	axRi = plt.subplot(gs[3])

	nullfmt = NullFormatter()
	axUp.xaxis.set_major_formatter(nullfmt)
	axRi.yaxis.set_major_formatter(nullfmt)
	axUp.tick_params(bottom=False,left=False,right=True)
	axRi.tick_params(left=False,bottom=False,top=True)
	axUp.yaxis.set_label_position("right")
	axUp.yaxis.tick_right()
	axUp.set_ylabel(r"$\chi^2$")
	axRi.xaxis.set_label_position("top")
	axRi.xaxis.tick_top()
	axRi.set_xlabel(r"$\chi^2$")
	if abs(x0+xf-1)<0.11:
		ax.set_xlabel(r'$\sin^2\theta_{23}$')
	elif abs(x0+xf)<0.01:
		ax.set_xlabel(r'$\Delta m^2_{31}$')
	elif abs(x0+xf-2*math.pi)<0.1:
		ax.set_xlabel(r'$\delta_{CP}$')
	else:
		ax.set_xlabel(r'$\sin^2\theta_{13}$')
	if abs(y0+yf-1)<0.11:
		ax.set_ylabel(r'$\sin^2\theta_{23}$')
	elif abs(y0+yf)<0.01:
		ax.set_ylabel(r'$\Delta m^2_{31}$')
	elif abs(y0+yf-2*math.pi)<0.1:
		ax.set_ylabel(r'$\delta_{CP}$')
	else:
		ax.set_ylabel(r'$\sin^2\theta_{13}$')

	colors = ['c','g','r','m']

	levels1d = np.array([1,4,9.0, 16.0, 25.0]) # 1, 2, 3, 4, 5 sigma
	levels2d = np.array([2.3,4.61,6.18,11.53]) # 1, 90%, 2, 3 sigma


	axUp.plot(x,X2N_x, color='k')
	axUp.plot(x,X2I_x, color='k', linestyle='dotted')

	for i in range(4):
		axUp.plot([x0,xf],[levels1d[i],levels1d[i]], color=colors[i], linewidth=0.5)
		axRi.plot([levels1d[i],levels1d[i]], [y0,yf],color=colors[i], linewidth=0.5)
	axUp.set_ylim(0.,20.)
	axUp.set_xlim(x0,xf)

	axRi.plot(X2N_y, y, color='k')
	axRi.plot(X2I_y, y, color='k', linestyle='dotted')

	axRi.set_xlim(0.,100.)
	axRi.set_ylim(y0,yf)

	X, Y = np.meshgrid(x, y)
	levels = np.array([4.605,5.991,9.21])
	ax.contour(X,Y,Chi2N, levels=levels2d, colors=colors)
	ax.contour(X,Y,Chi2I, levels=levels2d, colors=colors, linestyles='dotted')
	ax.set_xlim(x0,xf)
	ax.set_ylim(y0,yf)

	plt.show()
