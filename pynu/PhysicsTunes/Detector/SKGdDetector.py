from PhysicsTunes import Tune
from .SKDetector import SuperK

import sys
sys.path.append('../')

####################
###### SK-Gd #######
####################


class SuperK_Gd(Tune):
    r"""Class containing general implementation of a Super-Kamiokande with Gadolinium detector."""

    def energy_scale(self, experiment, x):
        """ See `pynu.PhysicsTunes.Detector.SKDetector.SuperK.energy_scale`. """
        return SuperK.energy_scale(experiment, x)

    def diff_energy_scale(self, experiment, x):
        """ See `pynu.PhysicsTunes.Detector.SKDetector.SuperK.diff_energy_scale`. """
        return SuperK.diff_energy_scale(experiment, x)

    def FCPC_separation(self, experiment, x):
        """ See `pynu.PhysicsTunes.Detector.SKDetector.SuperK.FCPC_separation`. """
        return SuperK.FCPC_separation(experiment, x)

    def diff_FCPC_separation(self, experiment, x):
        """ See `pynu.PhysicsTunes.Detector.SKDetector.SuperK.diff_FCPC_separation`. """
        return SuperK.diff_FCPC_separation(experiment, x)

    def fiducial_volume(self, experiment, x):
        """ See `pynu.PhysicsTunes.Detector.SKDetector.SuperK.fiducial_volume_separation`. """
        return SuperK.fiducial_volume(experiment, x)

    def diff_fiducial_volume(self, experiment, x):
        """ See `pynu.PhysicsTunes.Detector.SKDetector.SuperK.diff_fiducial_volume_separation`. """
        return SuperK.diff_fiducial_volume(experiment, x)

    def FC_reduction(self, experiment, x):
        """ See `pynu.PhysicsTunes.Detector.SKDetector.SuperK.FC_reduction`. """
        return SuperK.FC_reduction(experiment, x)

    def diff_FC_reduction(self, experiment, x):
        """ See `pynu.PhysicsTunes.Detector.SKDetector.SuperK.diff_FC_reduction`. """
        return SuperK.diff_FC_reduction(experiment, x)

    def PC_reduction(self, experiment, x):
        """ See `pynu.PhysicsTunes.Detector.SKDetector.SuperK.PC_reduction`. """
        return SuperK.PC_reduction(experiment, x)

    def diff_PC_reduction(self, experiment, x):
        """ See `pynu.PhysicsTunes.Detector.SKDetector.SuperK.diff_PC_reduction`. """
        return SuperK.diff_PC_reduction(experiment, x)

    def subgev_2ring_pi0(self, experiment, x):
        """ See `pynu.PhysicsTunes.Detector.SKDetector.SuperK.subgev_2ring_pi0`. """
        return SuperK.subgev_2ring_pi0(experiment, x)

    def diff_subgev_2ring_pi0(self, experiment, x):
        """ See `pynu.PhysicsTunes.Detector.SKDetector.SuperK.diff_subgev_2ring_pi0`. """
        return SuperK.diff_subgev_2ring_pi0(experiment, x)

    def subgev_1ring_pi0(self, experiment, x):
        """ See `pynu.PhysicsTunes.Detector.SKDetector.SuperK.subgev_1ring_pi0`. """
        return SuperK.subgev_1ring_pi0(experiment, x)

    def diff_subgev_1ring_pi0(self, experiment, x):
        """ See `pynu.PhysicsTunes.Detector.SKDetector.SuperK.diff_subgev_1ring_pi0`. """
        return SuperK.diff_subgev_1ring_pi0(experiment, x)

    def multiring_nunubar_separation(self, experiment, x):
        """ See `pynu.PhysicsTunes.Detector.SKDetector.SuperK.multiring_nunubar_separation`. """
        return SuperK.multiring_nunubar_separation(experiment, x)

    def diff_multiring_nunubar_separation(self, experiment, x):
        """ See `pynu.PhysicsTunes.Detector.SKDetector.SuperK.diff_multiring_nunubar_separation`. """
        return SuperK.diff_multiring_nunubar_separation(experiment, x)

    def multiring_emu_separation(self, experiment, x):
        """ See `pynu.PhysicsTunes.Detector.SKDetector.SuperK.multiring_emu_separation`. """
        return SuperK.multiring_emu_separation(experiment, x)

    def diff_multiring_emu_separation(self, experiment, x):
        """ See `pynu.PhysicsTunes.Detector.SKDetector.SuperK.diff_multiring_emu_separation`. """
        return SuperK.diff_multiring_emu_separation(experiment, x)

    def multiring_eother_separation(self, experiment, x):
        """ See `pynu.PhysicsTunes.Detector.SKDetector.SuperK.multiring_eother_separation`. """
        return SuperK.multiring_eother_separation(experiment, x)

    def diff_multiring_eother_separation(self, experiment, x):
        """ See `pynu.PhysicsTunes.Detector.SKDetector.SuperK.diff_multiring_eother_separation`. """
        return SuperK.diff_multiring_eother_separation(experiment, x)

    def pc_stopthru_separation(self, experiment, x):
        """ See `pynu.PhysicsTunes.Detector.SKDetector.SuperK.pc_stopthru_separation`. """
        return SuperK.pc_stopthru_separation(experiment, x)

    def diff_pc_stopthru_separation(self, experiment, x):
        """ See `pynu.PhysicsTunes.Detector.SKDetector.SuperK.diff_pc_stopthru_separation`. """
        return SuperK.diff_pc_stopthru_separation(experiment, x)

    def pi0_ring_separation(self, experiment, x):
        """ See `pynu.PhysicsTunes.Detector.SKDetector.SuperK.pi0_ring_separation`. """
        return SuperK.pi0_ring_separation(experiment, x)

    def diff_pi0_ring_separation(self, experiment, x):
        """ See `pynu.PhysicsTunes.Detector.SKDetector.SuperK.diff_pi0_ring_separation`. """
        return SuperK.diff_pi0_ring_separation(experiment, x)

    def e_ring_separation(self, experiment, x):
        """ See `pynu.PhysicsTunes.Detector.SKDetector.SuperK.e_ring_separation`. """
        return SuperK.e_ring_separation(experiment, x)

    def diff_e_ring_separation(self, experiment, x):
        """ See `pynu.PhysicsTunes.Detector.SKDetector.SuperK.diff_e_ring_separation`. """
        return SuperK.diff_e_ring_separation(experiment, x)

    def mu_ring_separation(self, experiment, x):
        """ See `pynu.PhysicsTunes.Detector.SKDetector.SuperK.mu_ring_separation`. """
        return SuperK.mu_ring_separation(experiment, x)

    def diff_mu_ring_separation(self, experiment, x):
        """ See `pynu.PhysicsTunes.Detector.SKDetector.SuperK.diff_mu_ring_separation`. """
        return SuperK.diff_mu_ring_separation(experiment, x)

    def singlering_pid(self, experiment, x):
        """ See `pynu.PhysicsTunes.Detector.SKDetector.SuperK.singlering_pid`. """
        return SuperK.singlering_pid(experiment, x)

    def diff_singlering_pid(self, experiment, x):
        """ See `pynu.PhysicsTunes.Detector.SKDetector.SuperK.diff_singlering_pid`. """
        return SuperK.diff_singlering_pid(experiment, x)

    def multiring_pid(self, experiment, x):
        """ See `pynu.PhysicsTunes.Detector.SKDetector.SuperK.multiring_pid`. """
        return SuperK.multiring_pid(experiment, x)

    def diff_multiring_pid(self, experiment, x):
        """ See `pynu.PhysicsTunes.Detector.SKDetector.SuperK.diff_multiring_pid`. """
        return SuperK.diff_multiring_pid(experiment, x)

    def neutron_tagging(self, experiment, x):
        """ See `pynu.PhysicsTunes.Detector.SKDetector.SuperK.neutron_tagging`. """
        return SuperK.neutron_tagging(experiment, x)

    def diff_neutron_tagging(self, experiment, x):
        """ See `pynu.PhysicsTunes.Detector.SKDetector.SuperK.diffneutron_tagging`. """
        return SuperK.diff_neutron_tagging(experiment, x)

    def decay_e_tagging(self, experiment, x):
        """ See `pynu.PhysicsTunes.Detector.SKDetector.SuperK.decay_e_tagging`. """
        return SuperK.decay_e_tagging(experiment, x)

    def diff_decay_e_tagging(self, experiment, x):
        """ See `pynu.PhysicsTunes.Detector.SKDetector.SuperK.diff_decay_e_tagging`. """
        return SuperK.diff_decay_e_tagging(experiment, x)
