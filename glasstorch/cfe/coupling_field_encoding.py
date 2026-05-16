import torch
import itertools
from ..utilities.tupling_iterator import TuplingIterator
from ..utilities.dynamic_index import dynamic_index, dynamic_set, list_to_target_dict

class CouplingFieldEncoding:
    def __init__(self, attributes):
        self.attributes = attributes
        self.is_quenched = attributes.is_quenched

        self.cfe = self._initialize_cfe() # Proper Coupling Field Encoding
        self.cfe_sign = self.cfe.clone() # Keeps track of the signs of cfe for the log space transformation.
        self.raw_cfe = self._initialize_cfe(raw=True) # This contains the raw coupling coefficients, as expected in the hamiltonian. By contrast, the cfe has recursive corrections built into the embedding, so it is not interpretable in the same way as the raw_cfe.

        if self.is_quenched:
            if self.attributes.mixture_type == "pure":
                self._initialize_cfe_for_pure_quenched_spin_glass() # The pure case is so much simpler that we will just treat it separately.
            else:
                self._initialize_cfe_for_quenched_spin_glass()

    '''
    Returns a tensor of shape (2, 2, ..., 2) with N dimensions of all 1s. This is the defult template for the coupling field encoding and will be adjusted for the quenched spin glass per the attributes of the spin glass.
    '''
    def _initialize_cfe(self, raw=False):
        if raw:
            cfe = torch.ones(*([2]* self.attributes.N), device=self.attributes.device())
        else:
            cfe = torch.zeros(*([2]* self.attributes.N), device=self.attributes.device())
        return cfe
    
    '''
    We will be leaving the cfe mostly unchanged for the pure quenched spin glass (all 1s).
    The cfe and raw_cfe will be identical, as no corrections must be iteratively made as in the mixed case.
    Only index combinations of cardinality p will be assigned their coupling coefficients, J, which are drawn from a normal distribution with mean 0 and variance p!/N^(p-1) for normalization="thermodynamic"; However the varience can be adjusted by the user via the attributes of the spin glass *In later versions (this version will only implement the thermodynamic normalization)*.
    '''
    def _initialize_cfe_for_pure_quenched_spin_glass(self):
        p = self.attributes.p
        N = self.attributes.N

        '''
        TuplingIterator will give us all tuples of length p with values in range(N) without repeats, which is exactly what we need to assign the coupling coefficients to the appropriate index combinations in the cfe and raw_cfe.
        '''
        for index_tuple in TuplingIterator(p, N, no_repeats=True):
            J = self.attributes.get_coupling_coefficient()
            targets = list_to_target_dict(index_tuple, scope=1)
            dynamic_set(self.raw_cfe, targets, J)
            J_bar = torch.log(torch.abs(J)) # We will be storing J_bar in log space for numerical stability, as mentioned in the comments in the _initialize_cfe_for_quenched_spin_glass method. In the pure case, there are no corrections to be made, so J_bar is just the log of the absolute value of J (we will handle the sign of J separately in the energy calculations).
            dynamic_set(self.cfe_sign, targets, torch.sign(J))
            dynamic_set(self.cfe, targets, J_bar)

    
    def _initialize_cfe_for_quenched_spin_glass(self):
        p = self.attributes.p
        N = self.attributes.N

        '''
        We need to go through each combinatoric "layer" i.e. the cardinality of the index combinations, starting with the lowest layer and working our way up to the highest layer (cardinality p).

        Layer 0 is trivial and will remain set to 1, so we start with layer 1, which is the first layer that has non-trivial coupling coefficients.
        Except of course, if the interaction level is "simple", then we will start with layer 2, as layer 1 interactions are just self-interactions by definition, and we are not including self-interactions in the "simple" interaction level.
        So in the simple case, all layer 1 indices will remain 1 along with layer 0 (as is always the case).
        '''
        if self.attributes.interaction_level == "self_interacting":
            starting_combinatoric_layer = 1
        elif self.attributes.interaction_level == "simple":
            starting_combinatoric_layer = 2
        else:
            raise ValueError(f"Invalid interaction level: {self.attributes.interaction_level}")

        for combinatoric_layer in range(starting_combinatoric_layer, p+1):
            for index_tuple in TuplingIterator(p, N, no_repeats=True):
                J = self.attributes.get_coupling_coefficient()
                targets = list_to_target_dict(index_tuple, scope=1)
                dynamic_set(self.raw_cfe, targets, J)

                J_bar = torch.log(torch.abs(J)) # This will be the adjusted coupling coefficient, once all the factors from the combinatoric layers below have been applied. We will iteratively adjust J_bar as we move up the combinatoric layers, and then assign it to the cfe at the end of the loop.
                J_bar_sign = torch.sign(J)
                '''
                The term "padrito" here is not out-of-pocket nonsense (although effectively it is).
                It makes perfect sense as a formal name of a sub-aspect indexed by the layers down from a subject in a infinite dimensional polytope theory that is in my personal mathematical canon.

                We need to run through each layer of decending combinatoric subsets and factor them into J_bar, with a critical adjustment consisting of an alternating form between the lower J_bar's and the lower J_bar^-1's.
                This allows everything to get factored out in the energy calculations (and their derivatives).

                We actually will see (in high dimension) underflow vanishing very quickly, so we will need to store J_bar in log space.
                This is quite convenient, as we intend to do the energy calculations in log space as well, so we will be able to just add the log J_bar's instead of multiplying the J_bar's, which will be much more numerically stable.

                Note: This is the `literal` implementation of the combinatoric adjustment.
                By contrast, there is an `inductive` implementation that may be mathematically more elegant (and equivilent), but I haven't worked it all out yet, and the literal implementation will do just fine for now.
                '''
                for padrito in range(1, combinatoric_layer): # Cardinality `combinatoric_layer` is already handled trivially above. It remains to treat the rest of the way down. Cardinality 0 is trivial as well, as it is always 1 (note: this corresponds to point [0,0,0,...], which is a "padrito" of every aspect of the hyper-cube).

                    subset_cardinality = combinatoric_layer - padrito
                    correction_power = (-1)**(padrito) # This is the alternating form that is critical to the whole construction.

                    for subset in itertools.combinations(index_tuple, subset_cardinality):
                        subset_targets = list_to_target_dict(subset, scope=1)
                        J_bar += correction_power * dynamic_index(self.cfe, subset_targets) # This is the iterative adjustment of J_bar as we move up the combinatoric layers.
                        J_bar_sign *= dynamic_index(self.cfe_sign, subset_targets)**correction_power # This is the iterative adjustment of J_bar_sign as we move up the combinatoric layers.

                dynamic_set(self.cfe, targets, J_bar) # Finally, we assign the fully adjusted coupling coefficient to the cfe at the appropriate index combination.
                dynamic_set(self.cfe_sign, targets, J_bar_sign) # Finally, we assign the fully adjusted coupling coefficient sign to the cfe_sign at the appropriate index combination.