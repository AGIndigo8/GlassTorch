import torch
from .spin_glass_attributes import SpinGlassAttributes
from ..cfe.coupling_field_encoding import CouplingFieldEncoding
from ..utilities.energy_calc import EnergyCalculator

class SpinGlass:
    def __init__(self, attributes: SpinGlassAttributes, gamma=None):
        self.attributes = attributes
        self._initialize_gamma(gamma)

        self.cfe = CouplingFieldEncoding(self.attributes)
        self._quenched_harmony() # Just a safety check to make sure the quenched state of the spin glass and coupling field encoding match.

        if self.attributes.is_quenched:
            self.energy_calc = EnergyCalculator(self)

    '''
    Safety check to make sure the quenched state of the spin glass and coupling field encoding match. This is important because, instead of using inheritance, we have two flavors of spin glasses. The unquenched is an abstract template and the quenched will have definite couplings. The coupling encodings of unqueched flavor are indefinite.
    '''
    def _quenched_harmony(self):
        if self.attributes.is_quenched != self.cfe.is_quenched:
            raise ValueError("Quenched state of the spin glass and coupling field encoding must match.")

    def _initialize_gamma(self, gamma):
        if gamma is not None:
            if gamma.shape[0] != self.attributes.N:
                raise ValueError(f"Gamma must have shape ({self.attributes.N},), but got {gamma.shape}")
            self.attributes.gamma = gamma

        # Default is all 1s, which will have equal weight for all interactions.
        elif self.attributes.gamma is None:
            self.attributes.gamma = torch.ones(self.attributes.N)

    
    def parent(self):
        if self.attributes.is_quenched:
            return self.attributes.parent
        else:
            raise ValueError("This spin glass is not quenched, so it does not have a parent.")
        
    '''
    Returns a new spin glass from the current configuation by quenching it. Gamma is inherited, but can be changed by the user. The new spin glass is a child of the current spin glass, and the current spin glass becomes the parent of the new spin glass.
    '''
    def quench(self):
        quenched_rng = self.attributes.get_next_quenched_rng()
        quenched_attributes = SpinGlassAttributes(
            p=self.attributes.p,
            N=self.attributes.N,
            interaction_level=self.attributes.interaction_level,
            mixture_type=self.attributes.mixture_type,
            gamma=self.attributes.gamma,
            temperature=self.attributes.temperature,
            master_seed=self.attributes.master_seed,
            quenched_seed=quenched_rng.initial_seed(),
            master_rng=self.attributes.master_rng,
            quenched_rng=quenched_rng,
            is_quenched=True,
            parent=self
        )
        return SpinGlass(quenched_attributes)
    
    def energy(self, sigma: torch.Tensor) -> torch.Tensor:
        if not self.attributes.is_quenched:
            raise ValueError("Energy is only defined for quenched spin glasses, as the coupling coefficients must be drawn and fixed for the energy calculation to be well-defined.")
        
        if sigma.shape[0] != self.attributes.N or sigma.dim() != 1:
            raise ValueError(f"Sigma must be of shape ({self.attributes.N},), but got {sigma.shape}.")
        
        if self.attributes.mixture_type == "pure":
            return self.energy_calc.p_aspect_contraction(sigma)
        else:
            return self.energy_calc.mixed_hamiltonian_contraction(sigma)

    