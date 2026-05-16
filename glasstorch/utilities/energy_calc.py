import torch
import itertools

from .dynamic_index import dynamic_index, dynamic_aspect, list_to_target_dict, dynamic_set

class EnergyCalculator:
    def __init__(self, spin_glass_model):
        self.model = spin_glass_model
        
        if not self.model.attributes.is_quenched:
            raise ValueError("EnergyCalculator requires a quenched spin glass model.")
        
        self.model._quenched_harmony()

        self.N = self.model.attributes.N
        self.p = self.model.attributes.p
        self.interaction_level = self.model.attributes.interaction_level
        self.mixture_type = self.model.attributes.mixture_type
        self.gamma = self.model.attributes.gamma
        
    '''
    For a given configuration, sigma, sigma morph transforms the 1D tensor of shape (N,) into a ND-tensor of shape (2, 2, ..., 2) with N dimensions.
    sigma[i] goes to index morph[0, 0, ..., 0, 1, 0, ..., 0], where the 1 is at the i-th position.
    Everything else stays 1.
    Effectively, this is a Hamming-weight-1 shell embedding of the configuration, sigma, into the space of the cubic tensor, T.
    This will be Haddamard multiplied with the coupling field encoding during the energy calculation.
    See the whitepaper for an explanation of why this is the case.
    '''
    @staticmethod
    def sigma_morph(sigma: torch.Tensor) -> torch.Tensor:
        N = sigma.shape[0]
        morph = torch.ones(*([2]*N), device=sigma.device) # shape (2, 2, ..., 2) with N dimensions
        for i in range(N):
            dynamic_set(morph, {i: 1}, sigma[i])
        return morph

    '''
    This creates the (batched) p-aspect expansion of tensor, T.
    In our case T will be CFE + log(sigma_morph(sigma)), but this function is general for any tensor T.
    '''
    @staticmethod
    def p_aspect_expasion(T: torch.Tensor, p: int, batch_size: int):
        N = T.dim()
        aspects = []
        for index_tuple in itertools.combinations(range(N), p):
            aspect = dynamic_aspect(T, list(index_tuple))
            aspects.append(aspect)

            if len(aspects) == batch_size:
                yield torch.stack(aspects)
                aspects = []

        if aspects:
            yield torch.stack(aspects)

    '''
    The p-aspect contraction is the main operation in the energy calculation.
    For simple (non-self-interacting) spin glasses, this is equivilent to the Hamiltonian evalutated over the p-degree interaction terms on the input configuration, sigma.
    For self-interacting spin glasses, a more complex operation is required (and not currently implemented).

    For the entire mixed-spin Hamiltonian, see the mixed_hamiltonian_contraction function below.

    For pure spin glasses, p must be set to None or exactly equal the p attribute of the spin glass model, as there is only one p-aspect to contract over in the pure case, which is the p specified by the attributes of the spin glass model, and in this case the p-aspect contraction is exactly the Hamiltonian.
    '''
    def p_aspect_contraction(self, sigma: torch.Tensor, p: int= None, batch_size: int = None):

        if sigma.shape[0] != self.N or sigma.dim() != 1:
            raise ValueError(f"Sigma must be of shape ({self.N},), but got {sigma.shape}.")

        if batch_size is None:
            batch_size = self.model.attributes.p_aspect_batch_size

        if p is None:
            p = self.p

        if self.mixture_type == "pure" and p != self.p:
            raise ValueError(f"For pure spin glasses,p-aspect contraction cannot be performed meaningfully on p other that the configured p in the SpinGlassAttributes. p must be None or equal to {self.p}, but got {p}.")

        if p < 2 or p > self.p:
            raise ValueError(f"p must be between 2 and {self.p}, but got {p}.")

        sigma_morph = self.sigma_morph(sigma)
        T = self.model.cfe.cfe + torch.log(torch.abs(sigma_morph))
        signs = self.model.cfe.cfe_sign * torch.sign(sigma_morph).detach() # We detach the signs from the computational graph, as they are not differentiable and we will handle them separately in the energy calculation.
        
        energy = 0.0

        for aspect_batch, sign_batch in zip(self.p_aspect_expasion(T, p, batch_size), self.p_aspect_expasion(signs, p, batch_size)):
            energy += torch.sum(torch.exp(aspect_batch) * sign_batch.detach())

        return energy

    '''
    The mixed Hamiltonian contraction is the main operation for calculating the energy of a mixed-spin system.
    It sums the contributions from different p-aspects, weighted by their respective gamma coefficients.

    By default, it will sum over all p-aspects from 2 to the `p` specified in the attributes of the spin glass model.
    However, the user may choose to sum over a subset of the p-aspects by providing a list of integers, Ps, which specify the p-aspects to sum over. For example, if Ps=[2, 4], then only the 2-aspect and 4-aspect will be calculated and summed in the final energy calculation.
    '''
    def mixed_hamiltonian_contraction(self, sigma: torch.Tensor, Ps: list[int]= None, batch_size: int = None):
        energy = 0.0

        if sigma.shape[0] != self.N or sigma.dim() != 1:
            raise ValueError(f"Sigma must be of shape ({self.N},), but got {sigma.shape}.")
        
        if self.mixture_type == "pure":
            return self._pure_contraction_redirect(sigma, Ps, batch_size)

        if Ps is None:
            Ps = list(range(2, self.p + 1))
        elif not all(p in range(2, self.p + 1) for p in Ps):
            raise ValueError(f"Ps must be a list of integers between 2 and {self.p}, but got {Ps}.")

        for p in Ps:
            energy += self.gamma[p-1] * self.p_aspect_contraction(sigma, p, batch_size)
        return energy
    
    '''
    In the event of a pure spin glass, when `mixed_hamiltonian_contraction` is called, this defaults to a single p-aspect contraction with p equal to the p specified in the attributes of the spin glass model, as there is only one p-aspect to contract over in the pure case, which is the p specified by the attributes of the spin glass model.
    This function simply performs the redirection and does some safety checks to communicate to the user why a failing case is not a valid input in the scheme of a pure spin glass.
    '''
    def _pure_contraction_redirect(self, sigma, Ps, batch_size):
        if Ps is not None and (len(Ps) != 1 or Ps[0] != self.p):
            raise ValueError(f"For pure spin glasses, Ps must be None or [p], where p is the configured p in the SpinGlassAttributes, but got {Ps}. Alternatively, you can use the p_aspect_contraction function directly for pure spin glasses, as there is only one p-aspect to contract over in the pure case, which is the p specified by the attributes of the spin glass model. If you want to perform mixed contractions, you must use a mixed spin glass model.")
        
        return self.p_aspect_contraction(sigma, p=self.p, batch_size=batch_size)