"""CUDA arithmetic replay for the Ascend MLP Alpha158 workflow."""

def compatibility_context(seed):
    """Load the Ascend kernels only for workflows that opt in."""
    from . import mlp_cuda_compat as compat
    from .triton_ieee_fma import fma
    from .triton_mma4 import mma4
    from .triton_mlp_linear import first_weight_splitk_228, linear_batch

    compat.ieee_fma_normal = fma
    compat.mma4 = mma4
    compat.linear_batch = linear_batch
    compat.first_weight_splitk_228 = first_weight_splitk_228
    return compat.CompatibilityContext(seed=seed)
