from __future__ import annotations
from pydantic import BaseModel, ConfigDict
from typing import Literal

import torch
import math

class SpinGlassAttributes(BaseModel):
    """
    Attributes for a spin glass model.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    p : int = 3
    N : int = 1000
    interaction_level : Literal["simple", "self_interacting"] = "simple"
    mixture_type : Literal["pure", "mixed"] = "pure"
    gamma : torch.Tensor = None
    temperature : float = 1.0
    master_seed : int = 42
    quenched_seed : int = None
    master_rng : torch.Generator | None = None
    quenched_rng : torch.Generator | None = None
    is_quenched : bool = False
    parent : SpinGlass | None = None
    normalization : Literal["thermodynamic", "none"] = "thermodynamic"
    
    p_aspect_batch_size : int = 1000 # This is the batch size for the p-aspect of the energy calculation. For high N and p, the p-aspect can be very large, so this must be tempered to the GPU memory limitations of the user.

    def __post_init__(self):
        if self.master_rng is None:
            self.master_rng = torch.Generator().manual_seed(self.master_seed)

    '''
    This grabs the next random seed from the master_rng for use in a new quenched_rng.
    The master_rng is used to ensure reproducibility across different quenched configurations of the same spin glass model, as the master_rng will generate the same sequence of random seeds for the quenched_rngs each time, given the same master_seed.
    '''
    def get_next_master_random_seed(self):
        return torch.randint(0, 2**32, (1,), generator=self.master_rng).item()

    def get_next_quenched_rng(self):
        return torch.Generator().manual_seed(self.get_next_master_random_seed())
    
    def get_coupling_coefficient_variance(self, p, N):
        if self.normalization == "thermodynamic":
            return math.factorial(p) / (N ** (p - 1))
        elif self.normalization == "none":
            return 1.0
        else:
            raise ValueError(f"Invalid normalization type: {self.normalization}")
        
    def get_coupling_coefficient(self):
        if not self.is_quenched:
            raise ValueError("Coupling coefficients are only defined for quenched spin glasses. (quenched random number generator is necessary to draw the coupling coefficients from the appropriate distribution in a reproducible manner.)")
        
        p = self.p
        N = self.N

        variance = self.get_coupling_coefficient_variance(p, N)
        std = math.sqrt(variance)
        return torch.normal(0.0, std, size=(), generator=self.quenched_rng)

    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"
