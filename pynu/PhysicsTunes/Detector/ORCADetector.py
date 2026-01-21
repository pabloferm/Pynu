"""
ORCA detector systematics for Pynu.

Implements the detector-specific systematic uncertainties for the KM3NeT/ORCA
neutrino telescope. These systematics are applied as weight modifiers to
the binned event rates.

Each systematic has two methods:
- SystName(experiment, x): Returns the weight modifier (1 + delta)
- diff_SystName(experiment, x): Returns the derivative d(weight)/dx
"""

import numpy as np
from PhysicsTunes import Tune

import sys
sys.path.append("../")


class ORCADetector(Tune):
    """
    ORCA detector systematics.
    
    Systematics implemented:
    - f_all: Overall normalization
    - f_HPT: High-purity track sample normalization  
    - f_Shower: Shower sample normalization
    - f_tauCC: Tau CC normalization
    - f_NC: NC normalization
    - f_HE: High-energy events normalization
    - E_shift: Energy scale shift
    """
    
    def f_all(self, experiment, x):
        """
        Overall normalization systematic.
        Scales all neutrino events by factor x.
        
        Args:
            experiment: ORCA experiment object
            x: Normalization factor (nominal = 1.0)
        
        Returns:
            Weight modifier array (x for all events)
        """
        return x * np.ones(experiment.NumberOfEvents)
    
    def diff_f_all(self, experiment, x):
        """Derivative of f_all w.r.t. x"""
        return np.ones(experiment.NumberOfEvents)
    
    def f_HPT(self, experiment, x):
        """
        High-purity track (HPT) sample normalization.
        Scales events in morphology 1 (HPT) by factor x.
        
        Args:
            experiment: ORCA experiment object
            x: HPT normalization factor (nominal = 1.0)
        
        Returns:
            Weight modifier array
        """
        weights = np.ones(experiment.NumberOfEvents)
        weights[experiment.Sample == 1] = x
        return weights
    
    def diff_f_HPT(self, experiment, x):
        """Derivative of f_HPT w.r.t. x"""
        deriv = np.zeros(experiment.NumberOfEvents)
        deriv[experiment.Sample == 1] = 1.0
        return deriv
    
    def f_Shower(self, experiment, x):
        """
        Shower sample normalization.
        Scales events in morphology 0 (shower) by factor x.
        
        Args:
            experiment: ORCA experiment object
            x: Shower normalization factor (nominal = 1.0)
        
        Returns:
            Weight modifier array
        """
        weights = np.ones(experiment.NumberOfEvents)
        weights[experiment.Sample == 0] = x
        return weights
    
    def diff_f_Shower(self, experiment, x):
        """Derivative of f_Shower w.r.t. x"""
        deriv = np.zeros(experiment.NumberOfEvents)
        deriv[experiment.Sample == 0] = 1.0
        return deriv
    
    def f_tauCC(self, experiment, x):
        """
        Tau CC normalization.
        Scales tau neutrino CC events by factor x.
        
        Args:
            experiment: ORCA experiment object
            x: Tau CC normalization factor (nominal = 1.0)
        
        Returns:
            Weight modifier array
        """
        weights = np.ones(experiment.NumberOfEvents)
        # Select tau CC events: |PDG| = 16 and CC (current_type = 1)
        tau_cc_mask = (np.abs(experiment.nuPDG) == 16) & (experiment.CC == 1)
        weights[tau_cc_mask] = x
        return weights
    
    def diff_f_tauCC(self, experiment, x):
        """Derivative of f_tauCC w.r.t. x"""
        deriv = np.zeros(experiment.NumberOfEvents)
        tau_cc_mask = (np.abs(experiment.nuPDG) == 16) & (experiment.CC == 1)
        deriv[tau_cc_mask] = 1.0
        return deriv
    
    def f_NC(self, experiment, x):
        """
        NC (neutral current) normalization.
        Scales NC events by factor x.
        
        Args:
            experiment: ORCA experiment object
            x: NC normalization factor (nominal = 1.0)
        
        Returns:
            Weight modifier array
        """
        weights = np.ones(experiment.NumberOfEvents)
        # NC events have current_type = 0
        weights[experiment.CC == 0] = x
        return weights
    
    def diff_f_NC(self, experiment, x):
        """Derivative of f_NC w.r.t. x"""
        deriv = np.zeros(experiment.NumberOfEvents)
        deriv[experiment.CC == 0] = 1.0
        return deriv
    
    def f_HE(self, experiment, x):
        """
        High-energy events normalization.
        Scales high-energy events by factor x.
        HE is defined as:
        - CC events with E_true > 500 GeV
        - NC events with E_true > 100 GeV
        
        Args:
            experiment: ORCA experiment object
            x: HE normalization factor (nominal = 1.0)
        
        Returns:
            Weight modifier array
        """
        weights = np.ones(experiment.NumberOfEvents)
        he_mask = ((experiment.ETrue > 500) & (experiment.CC == 1)) |                   ((experiment.ETrue > 100) & (experiment.CC == 0))
        weights[he_mask] = x
        return weights
    
    def diff_f_HE(self, experiment, x):
        """Derivative of f_HE w.r.t. x"""
        deriv = np.zeros(experiment.NumberOfEvents)
        he_mask = ((experiment.ETrue > 500) & (experiment.CC == 1)) |                   ((experiment.ETrue > 100) & (experiment.CC == 0))
        deriv[he_mask] = 1.0
        return deriv
    
    def E_shift(self, experiment, x):
        """
        Energy scale systematic.
        This is handled differently - it shifts the reconstructed energy
        used for binning rather than modifying weights directly.
        
        For the weight-based approach, we approximate the effect:
        A shift in E_reco changes which bin events fall into.
        
        Args:
            experiment: ORCA experiment object
            x: Energy scale factor (nominal = 1.0, E_reco_shifted = x * E_reco)
        
        Returns:
            Weight modifier array (ones - actual effect is in binning)
        """
        # The energy shift effect is implemented in the experiment's binning
        # This method returns 1 as a placeholder
        # The actual shift should be applied when calling BinMC with E_scale parameter
        experiment.set_energy_scale(x)
        return np.ones(experiment.NumberOfEvents)
    
    def diff_E_shift(self, experiment, x):
        """
        Derivative of E_shift w.r.t. x.
        Computed numerically since the effect depends on binning.
        """
        # Return zeros - the derivative effect is handled in the fitter
        # when computing binned derivatives
        return np.zeros(experiment.NumberOfEvents)


# Alias for consistency
ORCA = ORCADetector
