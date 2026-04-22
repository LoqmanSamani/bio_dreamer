import torch
import torch.nn as nn
import torch.nn.functional as F


class SIGReg(nn.Module):
    """
    Sketched Isotropic Gaussian Regularization.

    Penalizes embeddings that drift away from N(0, I) by:
    - matching per-dimension mean to 0
    - matching per-dimension variance to 1
    - optionally matching random 1d projections to Gaussian moments
    """

    def __init__(
        self, config,
        #latent_dim: int,
        #num_proj: int = 16,
        #mean_weight: float = 1.0,
        #var_weight: float = 1.0,
        #proj_weight: float = 1.0,
        #eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.device = config.device if hasattr(config, 'device') else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.latent_dim = config.latent_dim
        self.num_proj = config.num_proj
        self.mean_weight = config.mean_weight
        self.var_weight = config.var_weight
        self.proj_weight = config.proj_weight
        self.eps = config.eps

    def _flatten_batch(self, z: torch.Tensor) -> torch.Tensor:
        """flatten batch dimensions, keeping the latent dimension intact"""
        return z.reshape(-1, z.shape[-1])

    def _sample_projections(self, dtype):
        """sample random projection vectors and normalize to unit length"""
        w = torch.randn(self.num_proj, self.latent_dim, device=self.device, dtype=dtype)
        w = F.normalize(w, dim=-1)
        return w

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Embeddings of shape (..., D)

        Returns:
            Scalar regularization loss
        """
        z = z.to(self.device)
        z = self._flatten_batch(z)

        # centering and variance matching in latent space
        mean = z.mean(dim=0)
        var = z.var(dim=0, unbiased=False)
        mean_loss = mean.pow(2).mean()
        var_loss = (var - 1.0).pow(2).mean()

        # random 1d projections: encourage projected marginals to look Gaussian
        w = self._sample_projections(z.dtype)
        proj = z @ w.t()  # flattens to shape (N, num_proj)

        proj_mean = proj.mean(dim=0)
        proj_var = proj.var(dim=0, unbiased=False)

        # Gaussian-shape proxy: small mean, unit variance, low excess kurtosis
        proj_centered = proj - proj_mean
        proj_std = torch.sqrt(proj_var + self.eps)
        proj_norm = proj_centered / proj_std

        proj_mean_loss = proj_mean.pow(2).mean()
        proj_var_loss = (proj_var - 1.0).pow(2).mean()
        proj_kurtosis = (proj_norm.pow(4).mean(dim=0) - 3.0).pow(2).mean()

        proj_loss = proj_mean_loss + proj_var_loss + proj_kurtosis

        return (
            self.mean_weight * mean_loss
            + self.var_weight * var_loss
            + self.proj_weight * proj_loss
        )

        