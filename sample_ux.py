from glasstorch import SpinGlass, SpinGlassAttributes
import pytorch as torch


sk_model = SpinGlass(SpinGlassAttributes(p=3, N=1000, interaction_level="simple", mixture_type="pure", temperature=1.0, master_seed=42))

glasses = [sk_model.quench() for _ in range(10)]

for glass in glasses:
    sigma = torch.randn(glass.attributes.N, requires_grad=True)
    energy = glass.energy(sigma)
    hessian = torch.autograd.functional.hessian(glass.energy, sigma)

