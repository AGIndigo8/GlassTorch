from glasstorch import SpinGlass, SpinGlassAttributes
import torch


sk_model = SpinGlass(SpinGlassAttributes(p=3, N=4, interaction_level="simple", mixture_type="mixed", temperature=1.0, master_seed=42))

glasses = [sk_model.quench() for _ in range(1)]

for glass in glasses:
    print(f"CFE:\n{glass.cfe.cfe}\n")
    print(f"Raw CFE:\n{glass.cfe.raw_cfe}\n")
    sigma = torch.randn(glass.attributes.N, requires_grad=True, device=glass.attributes.device())
    #print(f"Sigma:\n{sigma}\n")
    energy = glass.energy(sigma)
    hessian = torch.autograd.functional.hessian(glass.energy, sigma)

    print(f"Energy: {energy.item()}")
    print(f"Hessian:\n{hessian}")

