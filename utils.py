import torch
import numpy as np

def encode_images(pixels: torch.Tensor, vae: torch.nn.Module, weight_dtype):
    pixel_latents = vae.encode(pixels.to(vae.dtype)).latent_dist.sample()
    pixel_latents = (pixel_latents - vae.config.shift_factor) * vae.config.scaling_factor
    return pixel_latents.to(weight_dtype)


def encode_intrinsics(combined_intrinsics_tensor: torch.Tensor, vae: torch.nn.Module, weight_dtype):
    """
    Encode the intrinsics from the combined tensor using the VAE.
    """
    # Extract the intrinsics from the combined tensor
    intrinsics = [combined_intrinsics_tensor[:, i, :, :, :] for i in range(combined_intrinsics_tensor.shape[1])]

    # Encode each intrinsic using the VAE
    encoded_intrinsics = []
    for intrinsic in intrinsics:
        latent = vae.encode(intrinsic.to(vae.dtype)).latent_dist.sample()
        latent = (latent - vae.config.shift_factor) * vae.config.scaling_factor
        encoded_intrinsics.append(latent.to(weight_dtype))

    return torch.cat(encoded_intrinsics, dim=1)  # Stack along the channel dimension
