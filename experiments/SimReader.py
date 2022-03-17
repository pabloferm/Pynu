import numpy as np
import pandas as pd
import h5py
import nuflux
import math
import nuSQuIDS as nsq
import nuSQUIDSTools
from itertools import repeat
from math import asin, sqrt
from Systematics import LoadSystematics

class Reader:
	def __init__(self, source, experiment, exposure, filename):

		self.Experiment = experiment
		self.Source = source
		self.Exposure = exposure
		self.FewEntries = None

		if self.Experiment == 'Super-Kamiokande' or self.Experiment == 'SK':
			self.Detector = 'Water'
			print(f'Processing simulation of {self.Experiment} experiment with a exposure of {self.Exposure} years.')
			with h5py.File(filename,'r') as hf:
				d_evis = np.array(hf['evis'][()])
				d_recocz = np.array(hf['recodirZ'][()])
				d_truecz = np.array(hf['dirnuZ'][()])
				d_azi = np.array(hf['azi'][()])
				d_mode = np.array(hf['mode'][()], dtype=int)
				d_ipnu = np.array(hf['ipnu'][()], dtype=int)
				d_pnu = np.array(hf['pnu'][()])
				d_weightReco = np.array(hf['weightReco'][()])
				d_weightSKTable = np.array(hf['weightOsc_SKpaper'][()])
				d_weightSim = np.array(hf['weightSim'][()])
				d_itype = np.array(hf['itype'][()], dtype=int)
				d_dcye = np.array(hf['muedk'][()], dtype=int)
			condition = (d_itype<16) * (d_itype>-1)
			self.EReco = d_evis[condition]
			self.CosZReco = d_recocz[condition]
			self.CosZTrue = d_truecz[condition]
			self.AziTrue = d_azi[condition]
			self.CC = np.abs(d_mode[condition]) < 30
			self.nuPDG = d_ipnu[condition]
			self.ETrue = d_pnu[condition]
			self.Weight = d_weightReco[condition] * d_weightSim[condition]
			self.Sample = d_itype[condition]
			self.NumberOfSamples = 16
			self.Erec_min = 0.1
			self.DecayE = d_dcye[condition]
			self.Mode = d_mode[condition]
			self.NumberOfEvents = self.nuPDG.size
			# SK: 39891.6 events in 5326 days
			SKrate = 36437.9607 / 5326 # SK events per day from 2017 atm. paper
			SKrateyr = SKrate * 365.25 # SK events per year from 2017 atm. paper
			norma = np.sum(d_weightReco[condition] * d_weightSKTable[condition])
			MCyears = norma / SKrateyr
			print(f'Your simulation file has {MCyears} years')
			self.Norm = self.Exposure / MCyears

		elif self.Experiment == 'SuperK-Gd' or self.Experiment == 'SKIV' or self.Experiment == 'SuperK_Htag' or self.Experiment == 'SuperK_Gdtag':
			self.Detector = 'Water'
			print(f'Processing simulation of {self.Experiment} experiment with a exposure of {self.Exposure} years.')
			with h5py.File(filename,'r') as hf:
				d_evis = np.array(hf['evis'][()])
				d_recocz = np.array(hf['recodirZ'][()])
				d_truecz = np.array(hf['dirnuZ'][()])
				d_azi = np.array(hf['azi'][()])
				d_mode = np.array(hf['mode'][()], dtype=int)
				d_ipnu = np.array(hf['ipnu'][()], dtype=int)
				d_pnu = np.array(hf['pnu'][()])
				d_weightReco = np.array(hf['weightReco'][()])
				d_weightSKTable = np.array(hf['weightOsc_SKpaper'][()])
				d_weightSim = np.array(hf['weightSim'][()])
				d_itype = np.array(hf['itype'][()], dtype=int)
				d_dcye = np.array(hf['muedk'][()], dtype=int)
				d_nn = np.array(hf['neutron'][()], dtype=int)
			# To be removed at some point
			condition = (d_itype<18) * (d_itype>-1) 
			self.EReco = d_evis[condition]
			self.CosZReco = d_recocz[condition]
			self.CosZTrue = d_truecz[condition]
			self.AziTrue = d_azi[condition]
			self.CC = np.abs(d_mode[condition]) < 30
			self.nuPDG = d_ipnu[condition]
			self.ETrue = d_pnu[condition]
			self.Weight = d_weightReco[condition] * d_weightSim[condition]
			self.Sample = d_itype[condition]
			self.NumberOfSamples = 18
			self.Erec_min = 0.1
			self.DecayE = d_dcye[condition]
			self.Neutron = d_nn[condition]
			self.Mode = d_mode[condition]
			self.NumberOfEvents = self.nuPDG.size
			# SK: 36437.9607 events in 5326 days
			SKrate = 36437.9607 / 5326 # SK events per day from 2017 atm. paper
			SKrateyr = SKrate * 365.25 # SK events per year from 2017 atm. paper
			norma = np.sum(d_weightReco[condition] * d_weightSKTable[condition])
			MCyears = norma / SKrateyr
			print(f'Your simulation file has {MCyears} years')
			self.Norm = self.Exposure / MCyears

		elif self.Experiment == 'IceCube-Upgrade' or self.Experiment == 'IC' or self.Experiment == 'DeepCore':
			self.Detector = 'Water'
			print(f'Processing simulation of {self.Experiment} experiment with a exposure of {self.Exposure} years.')
			input_data = pd.read_csv(filename)
			Time = self.Exposure * 365*24*60*60
			meter_to_cm_sq = 1e4
			self.Erec_min = 1
			if int(pd.__version__[0]) == 1:
				d_EReco = input_data['reco_energy'].to_numpy()
				d_CosZReco = np.cos(input_data['reco_zenith'].to_numpy())
				d_CosZTrue = np.cos(input_data['true_zenith'].to_numpy())
				d_AziTrue = input_data['true_azimuth'].to_numpy()
				d_CC = input_data['current_type'].to_numpy()
				d_nuPDG = np.int_(input_data['pdg'].to_numpy())
				d_ETrue = input_data['true_energy'].to_numpy()
				d_Weight = input_data['weight'].to_numpy()
				d_Sample = np.int_(input_data['pid'].to_numpy())
			elif int(pd.__version__[0]) == 0:
				d_nuPDG = np.int_(input_data["pdg"])
				d_EReco = np.array(input_data['reco_energy'])
				d_CosZReco = np.cos(input_data['reco_zenith'])
				d_CosZTrue = np.cos(input_data['true_zenith'])
				d_AziTrue = np.array(input_data['true_azimuth'])
				d_CC = np.array(input_data['current_type'])
				d_ETrue = np.array(input_data['true_energy'])
				d_Weight = np.array(input_data['weight'])
				d_Sample = np.int_(input_data['pid'])
			condition = (d_ETrue > 1) * (d_ETrue < 1e3) * (d_EReco > 1)
			self.EReco = d_EReco[condition]
			self.CosZReco = d_CosZReco[condition]
			self.CosZTrue = d_CosZTrue[condition]
			self.AziTrue = d_AziTrue[condition]
			self.CC = d_CC[condition]
			self.nuPDG = d_nuPDG[condition]
			self.ETrue = d_ETrue[condition]
			self.Weight = d_Weight[condition]
			self.Sample = d_Sample[condition]
			self.Norm = Time * meter_to_cm_sq
			self.NumberOfSamples = 2
			self.NumberOfEvents = self.nuPDG.size


		elif self.Experiment == 'ORCA':
			self.Detector = 'Water'
			print(f'Processing simulation of {self.Experiment} experiment with a exposure of {self.Exposure} years.')
			input_data = pd.read_csv(filename)
			Time = self.Exposure * 365*24*60*60
			meter_to_cm_sq = 1e4
			self.Erec_min = 1
			if int(pd.__version__[0]) == 1:
				d_EReco = input_data['reco_energy'].to_numpy()
				d_CosZReco = np.cos(input_data['reco_zenith'].to_numpy())
				d_CosZTrue = np.cos(input_data['true_zenith'].to_numpy())
				# d_AziTrue = input_data['true_azimuth'].to_numpy()
				# d_CC = input_data['current_type'].to_numpy()
				d_nuPDG = np.int_(input_data['pdg'].to_numpy())
				d_ETrue = input_data['true_energy'].to_numpy()
				d_Weight = input_data['weight'].to_numpy()
				d_Sample = np.int_(input_data['pid'].to_numpy())
			elif int(pd.__version__[0]) == 0:
				d_nuPDG = np.int_(input_data["pdg"])
				d_EReco = np.array(input_data['reco_energy'])
				d_CosZReco = np.cos(input_data['reco_zenith'])
				d_CosZTrue = np.cos(input_data['true_zenith'])
				# d_AziTrue = np.array(input_data['true_azimuth'])
				# d_CC = np.array(input_data['current_type'])
				d_ETrue = np.array(input_data['true_energy'])
				d_Weight = np.array(input_data['weight'])
				d_Sample = np.int_(input_data['pid'])
			condition = (d_ETrue > 1) * (d_ETrue < 1e3) * (d_EReco > 1)
			self.EReco = d_EReco[condition]
			self.CosZReco = d_CosZReco[condition]
			self.CosZTrue = d_CosZTrue[condition]
			# self.AziTrue = d_AziTrue[condition]
			# self.CC = d_CC[condition]
			self.nuPDG = d_nuPDG[condition]
			self.ETrue = d_ETrue[condition]
			self.Weight = d_Weight[condition]
			self.Sample = d_Sample[condition]
			self.Norm = Time * meter_to_cm_sq
			self.NumberOfSamples = 2
			self.NumberOfEvents = self.nuPDG.size

		self.FewEntries = []

	def Binning(self):
		if self.Experiment == 'Super-Kamiokande' or self.Experiment == 'SK':
			sge_ebins = np.array([0.1, 0.25, 0.4, 0.63, 1.0, 1.33])
			sgm_ebins = np.array([0.1, 0.25, 0.4, 0.63, 1.0, 1.33])
			sgsrpi0ebins = np.array([0.1, 0.25, 0.4, 0.63, 1.33])
			sgmrpi0ebins = np.array([0.1, 0.15, 0.25, 0.4, 0.63, 1.33])
			mge_ebins = np.array([1.3, 2.5, 5., 10., 500.])
			mgm_ebins = np.array([1.3, 3.0, 500.])
			mre_ebins = np.array([1.3, 2.5, 5.0, 500.])
			mrm_ebins = np.array([0.6, 1.3, 2.5, 5., 500.])
			mro_ebins = np.array([1.3, 2.5, 5.0, 10., 500.])
			pcs_ebins = np.array([0.1, 10., 1.0e5])
			pct_ebins = np.array([0.1, 10.0, 50., 1.0e5])
			z10bins = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
			z1bins = np.array([-1, 1.0])
			self.EnergyBins = {0:sge_ebins, 1:sge_ebins, 2:sgsrpi0ebins, 3:sgm_ebins, 4:sgm_ebins, 5:sgm_ebins, 6:sgmrpi0ebins,
			7:mge_ebins, 8:mge_ebins, 9:mgm_ebins, 10:mre_ebins, 11:mre_ebins, 12:mrm_ebins, 13:mro_ebins, 14:pcs_ebins, 15:pct_ebins}
			self.CzBins = {0:z10bins, 1:z1bins, 2:z1bins, 3:z10bins, 4:z10bins, 5:z1bins, 6:z1bins,
			7:z10bins, 8:z10bins, 9:z10bins, 10:z10bins, 11:z10bins, 12:z10bins, 13:z10bins, 14:z10bins, 15:z10bins}
			self.MaxNumberOfEnergyBins = 5
			self.MaxNumberOfCzBins = 10
		elif self.Experiment == 'SuperK-Gd' or self.Experiment == 'SKIV' or self.Experiment == 'SuperK_Htag' or self.Experiment == 'SuperK_Gdtag':
			sge_ebins = np.array([0.1, 0.25, 0.4, 0.63, 1.0, 1.33])
			sgm_ebins = np.array([0.1, 0.25, 0.4, 0.63, 1.0, 1.33])
			sgsrpi0ebins = np.array([0.1, 0.25, 0.4, 0.63, 1.33])
			sgmrpi0ebins = np.array([0.1, 0.15, 0.25, 0.4, 0.63, 1.33])
			mge_ebins = np.array([1.3, 2.5, 5., 10., 500.])
			mgm_ebins = np.array([1.3, 3.0, 500.])
			mre_ebins = np.array([1.3, 2.5, 5.0, 500.])
			mrm_ebins = np.array([0.6, 1.3, 2.5, 5., 500.])
			mro_ebins = np.array([1.3, 2.5, 5.0, 10., 500.])
			pcs_ebins = np.array([0.1, 10., 1.0e5])
			pct_ebins = np.array([0.1, 10.0, 50., 1.0e5])
			z10bins = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
			z1bins = np.array([-1, 1.0])
			self.EnergyBins = {0:sge_ebins, 1:sge_ebins, 2:sge_ebins, 3:sgsrpi0ebins, 4:sgm_ebins, 5:sgm_ebins, 6:sgmrpi0ebins,
			7:mge_ebins, 8:mge_ebins, 9:mge_ebins, 10:mgm_ebins, 11:mgm_ebins, 12:mre_ebins, 13:mre_ebins, 14:mrm_ebins, 15:mro_ebins, 16:pcs_ebins, 17:pct_ebins}
			self.CzBins = {0:z1bins, 1:z10bins, 2:z10bins, 3:z1bins, 4:z10bins, 5:z10bins, 6:z1bins,
			7:z10bins, 8:z10bins, 9:z10bins, 10:z10bins, 11:z10bins, 12:z10bins, 13:z10bins, 14:z10bins, 15:z10bins, 16:z10bins, 17:z10bins}
			self.MaxNumberOfEnergyBins = 5
			self.MaxNumberOfCzBins = 10
		elif self.Experiment == 'IceCube-Upgrade' or self.Experiment == 'IC' or self.Experiment == 'DeepCore' or self.Experiment == 'ORCA':
			Erec_max = 1e4
			NErec = 40
			erec = np.logspace(np.log10(self.Erec_min), np.log10(Erec_max), NErec+1, endpoint = True)
			z10bins = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
			self.EnergyBins = {0:erec, 1:erec}
			self.CzBins = {0:z10bins, 1:z10bins}
			self.MaxNumberOfEnergyBins = 40
			self.MaxNumberOfCzBins = 10

	def ICSystematicTables(self):
		ev = np.zeros(self.NumberOfEvents)
		c = (np.abs(self.nuPDG)==12) * (self.CC)
		ev[c] = 1
		self.ExpFracNuECC = self.Exp_wBinIt(ev) / self.weightOscBF_binned
		ev = np.zeros(self.NumberOfEvents)
		c = (np.abs(self.nuPDG)==14) * (self.CC)
		ev[c] = 1
		self.ExpFracNuMuCC = self.Exp_wBinIt(ev) / self.weightOscBF_binned
		ev = np.zeros(self.NumberOfEvents)
		c = (np.abs(self.nuPDG)==16) * (self.CC)
		ev[c] = 1
		self.ExpFracNuTauCC = self.Exp_wBinIt(ev) / self.weightOscBF_binned
		ev = np.zeros(self.NumberOfEvents)
		c = np.logical_not(self.CC)
		ev[c] = 1
		self.ExpFracNC = self.Exp_wBinIt(ev) / self.weightOscBF_binned
		# Load systematics tables
		self.ice_absorption = LoadSystematics.ICUp(self.EnergyBins, self.CzBins, 'ice_absorption', cut=self.FewEntries)
		self.ice_scattering = LoadSystematics.ICUp(self.EnergyBins, self.CzBins, 'ice_scattering', cut=self.FewEntries)
		# self.offset = LoadSystematics.ICUp(self.EnergyBins, self.CzBins, 'offset', cut=self.FewEntries)
		self.opt_eff_headon = LoadSystematics.ICUp(self.EnergyBins, self.CzBins, 'opt_eff_headon', cut=self.FewEntries)
		self.opt_eff_lateral = LoadSystematics.ICUp(self.EnergyBins, self.CzBins, 'opt_eff_lateral', cut=self.FewEntries)
		self.opt_eff_overall = LoadSystematics.ICUp(self.EnergyBins, self.CzBins, 'opt_eff_overall', cut=self.FewEntries)
		self.coin_fraction = LoadSystematics.ICUp(self.EnergyBins, self.CzBins, 'coin_fraction', cut=self.FewEntries)

	def BFOscillator(self,neutrino_flavors, Sin2Theta12=0, Sin2Theta13=0, Sin2Theta23=0, Dm221=0, Dm231=0, dCP=0, Ordering='normal'):
		self.weightOscBF = self.Oscillator(neutrino_flavors, Sin2Theta12, Sin2Theta13, Sin2Theta23, Dm221, Dm231, dCP, Ordering)
		self.weightOscBF_binned = self.Exp_wBinIt(1)
		self.FewEntries = self.weightOscBF_binned>4
		self.weightOscBF_binned = self.weightOscBF_binned[self.FewEntries]
		self.NumberOfBins = self.weightOscBF_binned.size
		
		if self.Experiment == 'IceCube-Upgrade' or self.Experiment == 'IC' or self.Experiment == 'DeepCore' or self.Experiment == 'ORCA':
			self.ICSystematicTables()
			

	def BinOscillator(self, neutrino_flavors, t12, t13, t23, dm21, dm31, dcp, Ordering='normal'):
		w = self.Oscillator(neutrino_flavors, t12, t13, t23, dm21, dm31, dcp, Ordering)
		wo = self.wBinIt(w)
		return wo

	def BinIt(self,array,shift_E=1):
		v = np.array([])
		E = self.EReco * shift_E
		for s in range(self.NumberOfSamples):
			cond = self.Sample==s
			dummy_w = array[cond]
			Obs, __, __ = np.histogram2d(E[cond], self.CosZReco[cond], bins=(self.EnergyBins[s], self.CzBins[s]), weights=dummy_w)
			v = np.append(v,Obs)
		v = v.reshape(-1)
		if len(self.FewEntries) > 0:
			v = v[self.FewEntries]
		return v

	def wBinIt(self,array,shift_E=1):
		return self.BinIt(array*self.Weight*self.Norm,shift_E=shift_E)

	def Exp_wBinIt(self,array,shift_E=1):
		return self.wBinIt(array*self.weightOscBF,shift_E=shift_E)	

	def InitialFlux(self):
		if self.Experiment == 'Super-Kamiokande' or self.Experiment == 'SK' or self.Experiment == 'SuperK-Gd' or self.Experiment == 'SKIV' or self.Experiment == 'SuperK_Htag' or self.Experiment == 'SuperK_Gdtag':
			flux = nuflux.makeFlux('IPhonda2014_sk_solmin')
			E_min = 0.1
			E_max = 4.0e2
		else:
			flux = nuflux.makeFlux('IPhonda2014_spl_solmin')
			E_min = 1.0
			E_max = 1.0e3
			
		E_nodes = 100
		energy_range = nsq.logspace(E_min,E_max,E_nodes)
		energy_nodes = nsq.logspace(E_min,E_max,E_nodes)
		cth_min = -1.0
		cth_max = 1.0
		cth_nodes = 40
		cth_nodes = nsq.linspace(cth_min,cth_max,cth_nodes)
		neutrino_flavors = 3

		#Initialize the flux
		AtmInitialFlux = np.zeros((len(cth_nodes),len(energy_nodes),2,neutrino_flavors))

		for ic,nu_cos_zenith in enumerate(cth_nodes):
			for ie,nu_energy in enumerate(energy_range):
				AtmInitialFlux[ic][ie][0][0] = flux.getFlux(nuflux.NuE,nu_energy,nu_cos_zenith) # nue
				AtmInitialFlux[ic][ie][1][0] = flux.getFlux(nuflux.NuEBar,nu_energy,nu_cos_zenith) # nue bar
				AtmInitialFlux[ic][ie][0][1] = flux.getFlux(nuflux.NuMu,nu_energy,nu_cos_zenith) # numu
				AtmInitialFlux[ic][ie][1][1] = flux.getFlux(nuflux.NuMuBar,nu_energy,nu_cos_zenith) # numu bar
				AtmInitialFlux[ic][ie][0][2] = 0.  # nutau
				AtmInitialFlux[ic][ie][1][2] = 0.  # nutau bar
		self.energy_nodes = energy_nodes
		self.cth_nodes = cth_nodes
		self.AtmInitialFlux = AtmInitialFlux

	def Oscillator(self, neutrino_flavors, t12, t13, t23, dm21, dm31, dcp, Ordering='normal'):
		units = nsq.Const()
		interactions = False
		AtmOsc = nsq.nuSQUIDSAtm(self.cth_nodes,self.energy_nodes*units.GeV,neutrino_flavors,nsq.NeutrinoType.both,interactions)
		AtmOsc.Set_rel_error(1.0e-4);
		AtmOsc.Set_abs_error(1.0e-4);
		AtmOsc.Set_MixingAngle(0,1, asin(sqrt(t12)))
		AtmOsc.Set_MixingAngle(0,2, asin(sqrt(t13)))
		AtmOsc.Set_MixingAngle(1,2, asin(sqrt(t23)))
		AtmOsc.Set_SquareMassDifference(1,dm21)
		AtmOsc.Set_SquareMassDifference(2,dm31)
		if Ordering!='normal':
			AtmOsc.Set_SquareMassDifference(2,dm21-dm31)
		AtmOsc.Set_CPPhase(0,2,dcp)
		AtmOsc.Set_initial_state(self.AtmInitialFlux,nsq.Basis.flavor)
		AtmOsc.EvolveState()

		neutype = np.zeros(self.NumberOfEvents)
		neutype[self.nuPDG<0] = 1
		neuflavor = 0.5*np.abs(self.nuPDG)-6
		neutype = neutype.astype(np.uint32).tolist()
		neuflavor = neuflavor.astype(np.uint32).tolist()

		w = list(map(AtmOsc.EvalFlavor, neuflavor, self.CosZTrue, self.ETrue*units.GeV, neutype, repeat(True)))

		return np.array(w)

