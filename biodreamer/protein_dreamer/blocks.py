from __future__ import annotations

import logging
import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)





class DDPM(nn.Module):
    """diffusion-based dynamics model for ProteinDreamer"""
    def __init__(
        self,
        predictor: Any,
        beta_min: float = 0.0001,
        beta_max: float = 0.02,
        time_steps: int = 1000,
        cosine_s: float = 0.008,
        clip_min: float = 0.0001,
        clip_max: float = 0.9999,
        clip_out: bool = True,
        var_type: str = "fixed_small",  # options: "fixed_small", "fixed_large", "learned"
        pred_type: Optional[str] = None,
        schedule_type: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.predictor = predictor
        self.pred_type = pred_type if pred_type is not None else "noise"
        self.schedule_type = schedule_type if schedule_type is not None else "linear"
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.time_steps = time_steps
        self.cosine_s = cosine_s
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.clip_out = clip_out
        self.var_type = var_type

        valid_types = ["noise", "x0", "v"]
        if self.pred_type not in valid_types:
            raise ValueError(f"pred_type must be one of {valid_types}, got {self.pred_type}")
        valid_schedules = ["linear", "cosine"]
        if self.schedule_type not in valid_schedules:
            raise ValueError(f"schedule_type must be one of {valid_schedules}, got {self.schedule_type}")
        if self.schedule_type == "linear" and not (0.0 < self.beta_min < self.beta_max):
            raise ValueError("For linear schedule, require 0 < beta_min < beta_max")

        self._setup_schedule()

    @property
    def _device(self) -> torch.device:
        return self.betas.device

    def noise_step(
        self, x0: torch.Tensor, cond: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """sample t, add noise, and return (pred, target) based on pred_type"""
        noise = torch.randn_like(x0)
        t = torch.randint(0, self.time_steps, (x0.shape[0],), device=x0.device)
        xt, target = self.forward_diff(x0, t, noise)
        pred = self.predictor(xt, t, cond)
        return pred, target

    @torch.inference_mode()
    def sample_step(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """one reverse diffusion step: predict and compute x_{t-1}"""
        pred = self.predictor(xt, t, cond)
        x_prev, _ = self.reverse_diff(xt, t, pred)
        return x_prev

    @torch.inference_mode()
    def sample(self, shape: Tuple[int, ...], cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """generate a sample by iterating reverse diffusion steps from pure noise"""
        xt = torch.randn(shape, device=self._device)
        for t in reversed(range(self.time_steps)):
            t_batch = torch.full((shape[0],), t, device=self._device, dtype=torch.long)
            xt = self.sample_step(xt, t_batch, cond)
        return xt

    def forward_diff(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ddpm forward process: q(x_t | x_0)"""
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        sqrt_alpha_cumprod_t = self.get_index(sqrt_alpha_cumprod_t, x0.shape)
        sqrt_one_minus_alpha_cumprod_t = self.get_index(sqrt_one_minus_alpha_cumprod_t, x0.shape)
        # x_t = √ᾱ_t * x_0 + √(1 - ᾱ_t) * ε
        xt = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise
        if self.pred_type == "noise":
            target = noise
        elif self.pred_type == "x0":
            target = x0
        elif self.pred_type == "v":
            # v = √ᾱ_t * ε - √(1 - ᾱ_t) * x_0
            target = sqrt_alpha_cumprod_t * noise - sqrt_one_minus_alpha_cumprod_t * x0
        return xt, target

    def reverse_diff(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        pred: torch.Tensor,
        pred_var: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """reverse diffusion step: p_θ(x_{t-1} | x_t)"""
        pred_x0 = self.predict_x0(xt, t, pred)
        coef1 = self.posterior_mean_coef1[t]
        coef2 = self.posterior_mean_coef2[t]
        coef1 = self.get_index(coef1, xt.shape)
        coef2 = self.get_index(coef2, xt.shape)
        # μ_θ(x_t, t) = coef1 * x_0 + coef2 * x_t
        posterior_mean = coef1 * pred_x0 + coef2 * xt
        if self.var_type == "fixed_small":
            sqrt_var = self.sqrt_posterior_variance[t]
            sqrt_var = self.get_index(sqrt_var, xt.shape)
        elif self.var_type == "fixed_large":
            sqrt_var = self.sqrt_betas[t]
            sqrt_var = self.get_index(sqrt_var, xt.shape)
        else:
            variance = self.get_variance(t, pred_var)
            variance = self.get_index(variance, xt.shape)
            sqrt_var = torch.sqrt(variance)
        noise = torch.randn_like(xt)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(xt.shape) - 1)))
        # x_{t-1} ~ p_θ(x_{t-1} | x_t)
        x_prev = posterior_mean + nonzero_mask * sqrt_var * noise
        return x_prev, pred_x0

    def _setup_schedule(self) -> None:
        """build and register noise schedule buffers"""
        if self.schedule_type == "linear":
            betas = torch.linspace(self.beta_min, self.beta_max, self.time_steps)
        elif self.schedule_type == "cosine":
            steps = self.time_steps + 1
            t = torch.linspace(0, self.time_steps, steps)
            alphas_cumprod = (
                torch.cos(((t / self.time_steps) + self.cosine_s) / (1 + self.cosine_s) * torch.pi * 0.5) ** 2
            )
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, self.clip_min, self.clip_max)
        else:
            raise ValueError(f"Unsupported schedule type: {self.schedule_type}")
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        # q(x_t | x_0) coefficients
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        # q(x_{t-1} | x_t, x_0) coefficients
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_log_variance = torch.log(torch.clamp(posterior_variance, min=1e-20))
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        sqrt_posterior_variance = torch.sqrt(torch.clamp(posterior_variance, min=1e-20))
        sqrt_betas = torch.sqrt(betas)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance", posterior_log_variance)
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)
        self.register_buffer("sqrt_posterior_variance", sqrt_posterior_variance)
        self.register_buffer("sqrt_betas", sqrt_betas)

    def get_index(self, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """reshape a per-sample tensor for broadcasting against x of shape x_shape"""
        batch_size = t.shape[0]
        return t.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def predict_x0(self, xt: torch.Tensor, t: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """recover x_0 from model prediction based on pred_type"""
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        sqrt_alpha_cumprod_t = self.get_index(sqrt_alpha_cumprod_t, xt.shape)
        sqrt_one_minus_alpha_cumprod_t = self.get_index(sqrt_one_minus_alpha_cumprod_t, xt.shape)
        if self.pred_type == "noise":
            # x_0 = (x_t - √(1 - ᾱ_t) * ε_θ) / √ᾱ_t
            x0 = (xt - sqrt_one_minus_alpha_cumprod_t * pred) / sqrt_alpha_cumprod_t
        elif self.pred_type == "x0":
            x0 = pred
        elif self.pred_type == "v":
            # x_0 = √ᾱ_t * x_t - √(1 - ᾱ_t) * v_θ
            x0 = sqrt_alpha_cumprod_t * xt - sqrt_one_minus_alpha_cumprod_t * pred
        if self.clip_out:
            x0 = torch.clamp(x0, -1.0, 1.0)
        return x0

    def get_variance(self, t: torch.Tensor, pred_var: Optional[torch.Tensor] = None) -> torch.Tensor:
        """compute variance for the reverse step based on var_type"""
        if self.var_type == "fixed_small":
            # β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
            var = self.posterior_variance[t]
        elif self.var_type == "fixed_large":
            # β_t
            var = self.betas[t]
        elif self.var_type == "learned":
            if pred_var is None:
                raise ValueError("pred_var must be provided when var_type='learned'")
            min_log = self.posterior_log_variance[t]
            max_log = torch.log(self.betas[t])
            frac = (pred_var + 1) / 2
            var = torch.exp(frac * max_log + (1 - frac) * min_log)
        return var




class SDE(nn.Module):
    """score-based generative model via stochastic differential equations"""
    def __init__(
        self,
        predictor: Any,
        method: str = "ode",  # options: "vp", "ve", "sub-vp", "ode"
        pred_type: str = "v",  # options: "noise", "score", "v"
        schedule_type: str = "cosine",  # options: "linear", "cosine"
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        time_eps: float = 1e-5,
        num_steps: int = 1000,
        cosine_s: float = 0.008,
        sigma_min: float = 0.01,
        sigma_max: float = 50.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.predictor = predictor
        self.register_buffer("_anchor", torch.zeros(1))
        self.method = method
        self.pred_type = pred_type
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.time_eps = time_eps
        self.num_steps = num_steps
        self.cosine_s = cosine_s
        self.schedule_type = schedule_type
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.eps = eps

        valid_methods = ["vp", "ve", "sub-vp", "ode"]
        if self.method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got {self.method}")
        valid_types = ["noise", "score", "v"]
        if self.pred_type not in valid_types:
            raise ValueError(f"pred_type must be one of {valid_types}, got {self.pred_type}")
        valid_schedules = ["linear", "cosine"]
        if self.schedule_type not in valid_schedules:
            raise ValueError(f"schedule_type must be one of {valid_schedules}, got {self.schedule_type}")
        if self.schedule_type == "linear" and not (0.0 < self.beta_min < self.beta_max):
            raise ValueError("For linear schedule, require 0 < beta_min < beta_max")

    @property
    def _device(self) -> torch.device:
        return self._anchor.device

    def noise_step(
        self, x0: torch.Tensor, cond: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """sample t, add noise, and return (pred, target) based on pred_type"""
        noise = torch.randn_like(x0)
        t = self.sample_time(x0.shape[0], self.time_eps, device=x0.device, dtype=x0.dtype)
        xt, target = self.forward_diff(x0, t, noise)
        pred = self.predictor(xt, t, cond)
        return pred, target

    @torch.inference_mode()
    def sample_step(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        *,
        dt: Optional[float] = None,
        last_step: bool = False,
    ) -> torch.Tensor:
        """one reverse sde/ode step"""
        if dt is None:
            dt = -(1.0 - self.time_eps) / self.num_steps
        pred = self.predictor(xt, t, cond)
        xt = self.reverse_diff(xt, pred, t, dt, last_step=last_step)
        return xt

    @torch.inference_mode()
    def sample(self, shape: Tuple[int, ...], cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """generate a sample by iterating reverse sde/ode steps from pure noise"""
        xt = torch.randn(shape, device=self._device)
        t_schedule = torch.linspace(1.0, self.time_eps, self.num_steps + 1, device=self._device)
        dt = -(1.0 - self.time_eps) / self.num_steps
        for step in range(self.num_steps):
            t_current = float(t_schedule[step])
            t_batch = torch.full((shape[0],), t_current, dtype=xt.dtype, device=self._device)
            last_step = step == self.num_steps - 1
            xt = self.sample_step(xt, t_batch, cond, dt=dt, last_step=last_step)
        return xt

    def forward_diff(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """sde forward process: compute x_t and target based on pred_type"""
        mean_coeff, std = self.get_forward_params(t)
        mean_coeff = self._broadcast_to_shape(mean_coeff, x0.shape)
        std = self._broadcast_to_shape(std, x0.shape)
        # x_t = mean_coeff * x_0 + σ(t) * ε
        xt = mean_coeff * x0 + std * noise
        if self.pred_type == "noise":
            target = noise
        elif self.pred_type == "score":
            # ∇_x log p(x_t | x_0) = -ε / σ(t)
            target = -noise / (std + self.eps)
        elif self.pred_type == "v":
            # v = mean_coeff * ε - σ(t) * x_0
            target = mean_coeff * noise - std * x0
        return xt, target

    def reverse_diff(
        self,
        xt: torch.Tensor,
        pred: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        last_step: bool = False,
    ) -> torch.Tensor:
        """reverse sde/ode step: compute x_{t-dt} from x_t and model prediction"""
        if not torch.is_tensor(dt):
            assert dt < 0.0, "dt must be negative for reverse diffusion!"
            dt = torch.tensor(dt, device=xt.device, dtype=xt.dtype)
        drift_coeff, g_squared, diffusion_coeff = self.get_reverse_coeffs(t)
        drift_coeff = self._broadcast_to_shape(drift_coeff, xt.shape)
        g_squared = self._broadcast_to_shape(g_squared, xt.shape)
        diffusion_coeff = self._broadcast_to_shape(diffusion_coeff, xt.shape)
        if self.method == "ve":
            sigma_t = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
            std = sigma_t
        else:
            std = self.std(t)
        while std.dim() < len(xt.shape):
            std = std.unsqueeze(-1)
        if self.pred_type == "noise":
            score = -pred / (std + self.eps)
        elif self.pred_type == "score":
            score = pred
        elif self.pred_type == "v":
            # x_t = α_t*x_0 + σ_t*ε, v = α_t*ε - σ_t*x_0
            # x_0 = (α_t*x_t - σ_t*v) / (α_t² + σ_t²)
            mean_coeff, sig = self.get_forward_params(t)
            mc = self._broadcast_to_shape(mean_coeff, xt.shape)
            sg = self._broadcast_to_shape(sig, xt.shape)
            x0_pred = (mc * xt - sg * pred) / (mc.pow(2) + sg.pow(2) + self.eps)
            score = -(xt - mc * x0_pred) / (sg.pow(2) + self.eps)
        # [-½β(t)x - β(t)∇log p_t(x)]dt + √β(t)dw̄
        if self.method == "ode":
            drift = drift_coeff * xt - 0.5 * g_squared * score
        else:
            drift = drift_coeff * xt - g_squared * score
        if last_step or self.method == "ode":
            noise = torch.zeros_like(xt)
        else:
            noise = torch.randn_like(xt)
        diffusion = diffusion_coeff * noise
        # Euler-Maruyama: x_{t+dt} = x_t + f·dt + g·dW
        x_prev = xt + drift * dt + diffusion * torch.sqrt(torch.abs(dt))
        return x_prev

    def _broadcast_to_shape(self, tensor: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        """add trailing dimensions so tensor broadcasts against target_shape"""
        while tensor.dim() < len(target_shape):
            tensor = tensor.unsqueeze(-1)
        return tensor

    def get_forward_params(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """return (mean_coeff, std) for the forward process at time t"""
        if self.method == "vp":
            # VP-SDE: p(x_t | x_0) = N(α(t)x_0, σ²(t)I)
            mean_coeff = self.alpha(t)
            std = self.std(t)
        elif self.method == "ve":
            # VE-SDE: p(x_t | x_0) = N(x_0, σ²(t)I)
            mean_coeff = torch.ones_like(t)
            std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        elif self.method == "sub-vp":
            # Sub-VP-SDE: p(x_t | x_0) = N(x_0, σ²(t)I) where σ²(t) = 1 - e^(-∫β(s)ds)
            mean_coeff = torch.ones_like(t)
            std = self.std(t)
        elif self.method == "ode":
            # Probability flow ODE: same marginals as VP-SDE but deterministic
            mean_coeff = self.alpha(t)
            std = self.std(t)
        return mean_coeff, std

    def get_reverse_coeffs(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """return (drift_coeff, g², diffusion_coeff) for the reverse SDE at time t"""
        if self.method == "vp":
            # VP-SDE reverse: dx = [-½β(t)x - β(t)∇log p_t(x)]dt + √β(t)dw̄
            drift_coeff = -0.5 * self.beta(t)
            g_squared = self.beta(t)
            diffusion_coeff = torch.sqrt(self.beta(t))
        elif self.method == "ve":
            # VE-SDE reverse: dx = [-σ(t)dσ/dt ∇log p_t(x)]dt + √(2σ(t)dσ/dt)dw̄
            sigma_t = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
            dsigma_dt = sigma_t * torch.log(torch.tensor(self.sigma_max / self.sigma_min))
            drift_coeff = torch.zeros_like(t)
            g_squared = 2 * sigma_t * dsigma_dt
            diffusion_coeff = torch.sqrt(g_squared)
        elif self.method == "sub-vp":
            # Sub-VP-SDE reverse: dx = [-β(t)∇log p_t(x)]dt + √β(t)dw̄
            drift_coeff = torch.zeros_like(t)
            g_squared = self.beta(t)
            diffusion_coeff = torch.sqrt(self.beta(t))
        elif self.method == "ode":
            # Probability flow ODE: deterministic, no diffusion term
            drift_coeff = -0.5 * self.beta(t)
            g_squared = self.beta(t)
            diffusion_coeff = torch.zeros_like(t)
        return drift_coeff, g_squared, diffusion_coeff

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """β(t) — noise schedule."""
        if self.schedule_type == "linear":
            return self.beta_min + t * (self.beta_max - self.beta_min)
        elif self.schedule_type == "cosine":
            # β(t) = -d/dt log ᾱ(t) = tan(x) · π / (1+s)
            t_mapped = (t + self.cosine_s) / (1 + self.cosine_s) * torch.pi / 2
            beta_t = torch.tan(t_mapped) * (torch.pi / (1 + self.cosine_s))
            return torch.clamp(beta_t, min=0.0, max=1000.0)

    def integral_beta(self, t: torch.Tensor) -> torch.Tensor:
        """∫₀ᵗ β(s) ds."""
        if self.schedule_type == "linear":
            return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2
        elif self.schedule_type == "cosine":
            return -torch.log(self.alpha_squared(t))

    def _cosine_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """ᾱ(t) = cos²((t+s)/(1+s) · π/2) for cosine schedule."""
        return torch.cos((t + self.cosine_s) / (1 + self.cosine_s) * torch.pi / 2) ** 2

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """α(t) = exp(-½∫₀ᵗ β(s) ds)."""
        if self.schedule_type == "cosine":
            return torch.sqrt(self.alpha_squared(t))
        return torch.exp(-0.5 * self.integral_beta(t))

    def alpha_squared(self, t: torch.Tensor) -> torch.Tensor:
        """α²(t) = exp(-∫₀ᵗ β(s) ds)."""
        if self.schedule_type == "cosine":
            return self._cosine_alpha_bar(t) / self._cosine_alpha_bar(torch.zeros_like(t))
        return torch.exp(-self.integral_beta(t))

    def variance(self, t: torch.Tensor) -> torch.Tensor:
        """σ²(t) = 1 - α²(t)."""
        return 1.0 - self.alpha_squared(t)

    def std(self, t: torch.Tensor) -> torch.Tensor:
        """σ(t) = √(1 - α²(t))."""
        return torch.sqrt(self.variance(t))

    def snr(self, t: torch.Tensor) -> torch.Tensor:
        """SNR(t) = α²(t) / σ²(t)."""
        alpha_sq = self.alpha_squared(t)
        var = self.variance(t)
        return alpha_sq / (var + 1e-8)

    def sample_time(
        self,
        batch_size: int,
        eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """sample batch_size timesteps uniformly in [eps, 1]"""
        device = device if device is not None else self._device
        t = eps + (1 - eps) * torch.rand(batch_size, device=device)
        if dtype is not None:
            t = t.to(dtype=dtype)
        return t


class FlowMatchingScheduler(nn.Module):
    """rectified flow matching for latent space dynamics.

    linear interpolation: x_t = (1-t)*x_noise + t*x_data, t in [0,1].
    training: predict velocity v = x_data - x_noise; loss = MSE(v_θ(x_t, t, cond), v).
    sampling: ode dx/dt = v_θ(x, t, cond) from t=0 (noise) to t=1 (data).
    """
    def __init__(
        self,
        predictor: nn.Module,
        num_steps: int = 100,
        solver: str = "heun",
        time_eps: float = 1e-3,
    ) -> None:
        super().__init__()
        if solver not in ("euler", "heun"):
            raise ValueError(f"solver must be 'euler' or 'heun', got {solver!r}")
        self.predictor = predictor
        self.num_steps = num_steps
        self.solver = solver
        self.time_eps = time_eps

    def noise_step(
        self, x1: torch.Tensor, cond: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """sample a random interpolant and return (pred, target) velocity"""
        x0 = torch.randn_like(x1)
        t = torch.empty(x1.shape[0], device=x1.device).uniform_(self.time_eps, 1.0)
        t_bc = t.view(-1, *([1] * (x1.dim() - 1)))
        # x_t = (1-t)*x_noise + t*x_data
        x_t = (1 - t_bc) * x0 + t_bc * x1
        target = x1 - x0
        pred = self.predictor(x_t, t, cond)
        return pred, target

    @torch.inference_mode()
    def sample(self, shape: Tuple[int, ...], cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """integrate the learned velocity field from t=time_eps to t=1-time_eps"""
        device = next(self.predictor.parameters()).device
        x = torch.randn(shape, device=device)
        t_start = self.time_eps
        t_end = 1.0 - self.time_eps
        dt = (t_end - t_start) / self.num_steps
        t = t_start
        for _ in range(self.num_steps):
            t_batch = torch.full((shape[0],), t, device=device)
            if self.solver == "euler":
                v = self.predictor(x, t_batch, cond)
                x = x + dt * v
            else:  # heun
                v1 = self.predictor(x, t_batch, cond)
                x2 = x + dt * v1
                t_next_batch = torch.full((shape[0],), t + dt, device=device)
                v2 = self.predictor(x2, t_next_batch, cond)
                x = x + dt * (v1 + v2) / 2
            t = t + dt
        return x


class ResidualMLP(nn.Module):
    """mlp with residual blocks.

    each block: LayerNorm -> Linear -> Act -> Linear -> Dropout -> residual add.
    in_proj maps in_dim -> hidden_dim, out_proj maps hidden_dim -> out_dim.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int = 2,
        dropout: float = 0.0,
        act_cls: type = nn.GELU,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "norm": nn.LayerNorm(hidden_dim),
                        "fc1": nn.Linear(hidden_dim, hidden_dim * 2),
                        "fc2": nn.Linear(hidden_dim * 2, hidden_dim),
                        "drop": nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                    }
                )
            )
        self.act = act_cls()
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        for block in self.blocks:
            residual = x
            x = block["norm"](x)
            x = block["fc1"](x)
            x = self.act(x)
            x = block["fc2"](x)
            x = block["drop"](x)
            x = x + residual
        return self.out_proj(self.out_norm(x))


class GvpGNN(nn.Module):
    """GVP-GNN encoder stack for protein structure embedding.

    produces per-node scalar+vector embeddings preserving geometric equivariance.
    Given residue-level scalar and vector inputs plus edge features (e.g. relative
    displacement vectors, distances, edge-type scalars), returns updated per-residue
    embeddings.
    """
    def __init__(
        self,
        in_node_dims: Tuple[int, int],
        in_edge_dims: Tuple[int, int],
        hidden_dims: Tuple[int, int],
        n_layers: int = 3,
        vector_dim: int = 3,
        conv_type: str = "gvp",  # "gvp" (GVPConv) or "transformer" (graph-transformer attention)
        n_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vector_dim = vector_dim
        self.node_embed = GVP(in_node_dims, hidden_dims, vector_dim=vector_dim)
        self.edge_embed = GVP(in_edge_dims, hidden_dims, vector_dim=vector_dim)
        self.conv_type = conv_type
        layers = []
        if conv_type == "gvp":
            for _ in range(n_layers):
                layers.append(GVPConv(hidden_dims, hidden_dims, hidden_dims, vector_dim=vector_dim))
        elif conv_type == "transformer":
            hidden_s = hidden_dims[0]
            if hidden_s % n_heads != 0:
                n_heads = max(1, math.gcd(hidden_s, n_heads))
            for _ in range(n_layers):
                layers.append(
                    GraphTransformerLayer(
                        node_dims=hidden_dims,
                        edge_dims=hidden_dims,
                        hidden_dim=hidden_dims[0],
                        n_heads=n_heads,
                        dropout=dropout,
                        vector_dim=vector_dim,
                    )
                )
        else:
            raise ValueError(f"Unsupported conv_type: {conv_type!r}. Use 'gvp' or 'transformer'.")
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        s: torch.Tensor,
        v: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_s: Optional[torch.Tensor] = None,
        edge_v: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        s: (n, in_node_scalar)
        v: (n, in_node_vec, 3)
        edge_index: (2, e) long tensor
        edge_s: (e, in_edge_scalar) or None
        edge_v: (e, in_edge_vec, 3) or None

        Returns:
            s: (n, hidden_scalar)
            v: (n, hidden_vec, 3)
        """
        s_h, v_h = self.node_embed(s, v)
        if v_h is None:
            v_h = torch.zeros(s_h.shape[0], 0, self.vector_dim, device=s_h.device, dtype=s_h.dtype)
        if edge_s is None:
            edge_s = torch.zeros(edge_index.shape[1], 0, device=s.device, dtype=s.dtype)
        if edge_v is None:
            edge_v = torch.zeros(edge_index.shape[1], 0, self.vector_dim, device=s.device, dtype=s.dtype)
        s_e, v_e = self.edge_embed(edge_s, edge_v)
        if v_e is None:
            v_e = torch.zeros(s_e.shape[0], 0, self.vector_dim, device=s_e.device, dtype=s_e.dtype)
        for layer in self.layers:
            s_h, v_h = layer(s_h, v_h, edge_index, s_e, v_e)
        return s_h, v_h


class GVPConv(nn.Module):
    """single GVP message-passing layer.

    message_gvp maps (sender node features + edge features) → (s_msg, v_msg).
    node_gvp maps (node features + aggregated messages) → updated node features.

    Expected input shapes:
        s: (n, n_s)
        v: (n, n_v, 3)
        edge_index: LongTensor (2, e) with [0]=senders, [1]=receivers
        edge_s: (e, edge_s_dim)
        edge_v: (e, edge_v_dim, 3)
    """
    def __init__(
        self,
        node_dims: Tuple[int, int],
        edge_dims: Tuple[int, int],
        message_dims: Tuple[int, int],
        vector_dim: int = 3,
    ) -> None:
        super().__init__()
        self.node_dims = node_dims
        self.edge_dims = edge_dims
        self.msg_dims = message_dims
        self.vector_dim = vector_dim
        msg_in_dims = (node_dims[0] + edge_dims[0], node_dims[1] + edge_dims[1])
        self.message_gvp = GVP(msg_in_dims, message_dims, vector_dim=vector_dim)
        node_update_in = (node_dims[0] + message_dims[0], node_dims[1] + message_dims[1])
        self.node_gvp = GVP(node_update_in, node_dims, vector_dim=vector_dim)

    def forward(
        self,
        s: torch.Tensor,
        v: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_s: Optional[torch.Tensor],
        edge_v: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        senders, receivers = edge_index[0], edge_index[1]
        s_src = s[senders]
        v_src = v[senders]
        if edge_s is None:
            edge_s = torch.zeros(s_src.shape[0], 0, device=s.device, dtype=s.dtype)
        if edge_v is None:
            edge_v = torch.zeros(s_src.shape[0], 0, self.vector_dim, device=s.device, dtype=s.dtype)
        s_msg_in = torch.cat([s_src, edge_s], dim=-1) if edge_s.shape[-1] > 0 else s_src
        v_msg_in = (
            torch.cat([v_src, edge_v], dim=1)
            if (v_src.shape[1] > 0 or edge_v.shape[1] > 0)
            else v_src
        )
        m_s, m_v = self.message_gvp(s_msg_in, v_msg_in)
        n = s.shape[0]
        device = s.device
        agg_s = torch.zeros(n, m_s.shape[-1], device=device, dtype=m_s.dtype)
        agg_s = agg_s.index_add(0, receivers, m_s)
        if m_v is not None:
            agg_v = torch.zeros(n, m_v.shape[1], self.vector_dim, device=device, dtype=m_v.dtype)
            agg_v = agg_v.index_add(0, receivers, m_v)
        else:
            agg_v = torch.zeros(n, 0, self.vector_dim, device=device, dtype=s.dtype)
        s_comb = torch.cat([s, agg_s], dim=-1) if agg_s.shape[-1] > 0 else s
        v_comb = torch.cat([v, agg_v], dim=1) if v.shape[1] + agg_v.shape[1] > 0 else v
        s_upd, v_upd = self.node_gvp(s_comb, v_comb)
        s_out = s + s_upd if s_upd.shape[-1] == s.shape[-1] else s_upd
        if v_upd is not None and v_upd.shape[1] == v.shape[1]:
            v_out = v + v_upd
        else:
            v_out = v_upd if v_upd is not None else v
        return s_out, v_out


class GraphTransformer(nn.Module):
    """stack of GraphTransformerLayer layers"""
    def __init__(
        self,
        node_dims: Tuple[int, int],
        edge_dims: Tuple[int, int],
        hidden_dim: int,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.0,
        vector_dim: int = 3,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                GraphTransformerLayer(
                    node_dims=node_dims,
                    edge_dims=edge_dims,
                    hidden_dim=hidden_dim,
                    n_heads=n_heads,
                    dropout=dropout,
                    vector_dim=vector_dim,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        s: torch.Tensor,
        v: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_s: Optional[torch.Tensor] = None,
        edge_v: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        s_h, v_h = s, v
        for layer in self.layers:
            s_h, v_h = layer(s_h, v_h, edge_index, edge_s, edge_v)
        return s_h, v_h


class GraphTransformerLayer(nn.Module):
    """graph-transformer message-passing layer with multi-head attention and optional edge biasing"""
    def __init__(
        self,
        node_dims: Tuple[int, int],
        edge_dims: Tuple[int, int],
        hidden_dim: int,
        n_heads: int = 4,
        dropout: float = 0.0,
        vector_dim: int = 3,
    ) -> None:
        super().__init__()
        in_s, in_v = node_dims
        edge_s_dim = edge_dims[0]
        self.in_s = in_s
        self.in_v = in_v
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.vector_dim = vector_dim
        self.q = nn.Linear(in_s + in_v, hidden_dim)
        self.k = nn.Linear(in_s + in_v, hidden_dim)
        self.v = nn.Linear(in_s + in_v, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, in_s)
        self.edge_att = nn.Linear(edge_s_dim, n_heads) if edge_s_dim > 0 else None
        if in_v > 0:
            self.w_v = nn.Parameter(torch.empty(in_v, in_v))
            nn.init.xavier_uniform_(self.w_v)
        else:
            self.w_v = None
        self.norm1 = nn.LayerNorm(in_s)
        self.norm2 = nn.LayerNorm(in_s)
        self.ff = nn.Sequential(
            nn.Linear(in_s, max(in_s * 2, 4)),
            nn.GELU(),
            nn.Linear(max(in_s * 2, 4), in_s),
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        s: torch.Tensor,
        v: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_s: Optional[torch.Tensor],
        edge_v: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        s: (n, in_s)
        v: (n, in_v, 3)
        edge_index: (2, e)
        edge_s: (e, edge_s_dim) or None
        edge_v: (e, edge_v_dim, 3) or None

        returns updated (s_out, v_out) with the same shapes as input.
        """
        device = s.device
        senders, receivers = edge_index[0], edge_index[1]
        n = s.shape[0]
        if self.in_v > 0 and v is not None:
            v_norm = torch.sqrt((v ** 2).sum(dim=-1) + 1e-8)  # (n, in_v)
        else:
            v_norm = torch.zeros(n, 0, device=device, dtype=s.dtype)
        # Pre-ln: normalise before projecting q/k/v
        s_norm = self.norm1(s)
        s_input = torch.cat([s_norm, v_norm], dim=-1)
        q = self.q(s_input).view(n, self.n_heads, self.head_dim)
        k = self.k(s_input).view(n, self.n_heads, self.head_dim)
        v_val = self.v(s_input).view(n, self.n_heads, self.head_dim)
        q_j = q[receivers]
        k_i = k[senders]
        v_i = v_val[senders]
        scores = (q_j * k_i).sum(dim=-1) / math.sqrt(self.head_dim)  # (e, heads)
        if self.edge_att is not None and edge_s is not None:
            edge_bias = self.edge_att(edge_s)
            scores = scores + edge_bias
        exp_scores = torch.exp(scores)
        denom = torch.zeros(n, self.n_heads, device=device, dtype=exp_scores.dtype)
        denom.index_add_(0, receivers, exp_scores)
        denom = denom + 1e-8
        weights = exp_scores / denom[receivers]
        weighted_v = v_i * weights.unsqueeze(-1)
        agg = torch.zeros(n, self.n_heads, self.head_dim, device=device, dtype=weighted_v.dtype)
        agg.index_add_(0, receivers, weighted_v)
        agg = agg.view(n, self.hidden_dim)
        msg = self.out_proj(agg)
        s_updated = s + self.dropout(msg)
        ff = self.ff(self.norm2(s_updated))
        s_out = s_updated + self.dropout(ff)
        if self.in_v > 0 and v is not None:
            v_src = v[senders]
            v_msg = torch.einsum("...ij,oi->...oj", v_src, self.w_v) if self.w_v is not None else v_src
            w_mean = weights.mean(dim=1)
            weighted_v_msg = v_msg * w_mean.view(-1, 1, 1)
            agg_v = torch.zeros(n, v_msg.shape[1], self.vector_dim, device=device, dtype=v_msg.dtype)
            agg_v.index_add_(0, receivers, weighted_v_msg)
            v_out = v + agg_v if agg_v.shape[1] == v.shape[1] else agg_v
        else:
            v_out = v
        return s_out, v_out


class GVP(nn.Module):
    """Geometric Vector Perceptron (GVP) core block.

    Inputs:
        s: (..., n_s)      scalar features per node/edge
        v: (..., n_v, 3)   vector features per node/edge (3-D vectors)
    Outputs:
        s_out: (..., n_s_out)
        v_out: (..., n_v_out, 3)

    Vector features are linearly mixed across channels without bias on the 3-D
    axis to preserve rotational equivariance (original GVP design):
        v' = W_v · v           (channel mixing)
        v_norm = ||v'||₂       (per-channel norms become scalar inputs)
        s' = Linear([s, v_norm]) → s_out
        gate = σ(Linear(s')) applied to v'
    """
    def __init__(
        self,
        in_dims: Tuple[int, int],
        out_dims: Tuple[int, int],
        vector_dim: int = 3,
        scalar_act: Optional[nn.Module] = None,
        use_layernorm: bool = False,
        dropout: float = 0.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        in_s, in_v = in_dims
        out_s, out_v = out_dims
        self.in_s, self.in_v = in_s, in_v
        self.out_s, self.out_v = out_s, out_v
        self.vector_dim = vector_dim
        self.use_layernorm = use_layernorm
        if in_v > 0 and out_v > 0:
            self.w_v = nn.Parameter(torch.empty(out_v, in_v))
            nn.init.xavier_uniform_(self.w_v)
        else:
            self.w_v = None
        # v_norm is concatenated whenever out_v > 0 (from real norms or zero fallback)
        scalar_in = in_s + (out_v if out_v > 0 else 0)
        if out_s > 0:
            self.linear_s = nn.Linear(scalar_in, out_s)
            nn.init.xavier_uniform_(self.linear_s.weight)
            if self.linear_s.bias is not None:
                nn.init.zeros_(self.linear_s.bias)
        else:
            self.linear_s = None
        if out_v > 0:
            gate_in = out_s if out_s > 0 else scalar_in
            self.gate = nn.Linear(gate_in, out_v)
            nn.init.xavier_uniform_(self.gate.weight)
            if self.gate.bias is not None:
                nn.init.zeros_(self.gate.bias)
        else:
            self.gate = None
        self.scalar_act = scalar_act if scalar_act is not None else nn.GELU()
        self.layernorm_s = nn.LayerNorm(out_s) if (use_layernorm and out_s > 0) else None
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self._eps = eps

    def forward(
        self, s: torch.Tensor, v: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        s: (..., in_s)
        v: (..., in_v, 3) or None

        returns:
            s_out: (..., out_s) or torch.empty(..., 0) if out_s==0
            v_out: (..., out_v, 3) or None
        """
        v_lin = None
        v_norm = None
        if self.in_v > 0 and v is not None:
            v_lin = torch.einsum("...ij,oi->...oj", v, self.w_v)  # (..., out_v, 3)
            v_norm = torch.sqrt((v_lin ** 2).sum(dim=-1) + self._eps)
        elif self.out_v > 0:
            batch_shape = s.shape[:-1]
            v_lin = torch.zeros(*batch_shape, self.out_v, self.vector_dim, device=s.device, dtype=s.dtype)
            v_norm = torch.zeros(*batch_shape, self.out_v, device=s.device, dtype=s.dtype)
        if self.in_s > 0:
            s_in = s
        else:
            s_in = torch.zeros(*s.shape[:-1], 0, device=s.device, dtype=s.dtype)
        s_cat = torch.cat([s_in, v_norm], dim=-1) if (v_norm is not None and v_norm.shape[-1] > 0) else s_in
        s_out = None
        if self.linear_s is not None:
            s_out = self.linear_s(s_cat)
            if self.scalar_act is not None:
                s_out = self.scalar_act(s_out)
            if self.layernorm_s is not None:
                s_out = self.layernorm_s(s_out)
            if self.dropout is not None:
                s_out = self.dropout(s_out)
        else:
            s_out = s_cat if (self.gate is not None) else None
        v_out = None
        if self.gate is not None:
            gate_input = s_out if (s_out is not None) else s_cat
            gates = torch.sigmoid(self.gate(gate_input))  # (..., out_v)
            v_out = v_lin * gates.unsqueeze(-1)
        elif v_lin is not None:
            v_out = v_lin
        return s_out, v_out


class DeterministicPredictor(nn.Module):
    """autoregressive deterministic latent predictor using a stack of TransformerLayer blocks.

    behavior:
      - Input: sequence of latent tokens z of shape (B, T, C) or single step (B, C).
      - When causal=True (default), a causal mask is applied so each position may only
        attend to previous positions (autoregressive).
      - returns predicted next-token latent of shape (B, C) (prediction for position T → T+1).
    """
    def __init__(
        self,
        latent_dim: int,
        n_layers: int = 6,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_len: int = 1024,
        causal: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.causal = causal
        self.input_proj = nn.Identity()
        self.pos_emb = nn.Parameter(torch.zeros(max_len, latent_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.layers = nn.ModuleList(
            [
                TransformerLayer(latent_dim, n_heads=n_heads, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.out_head = nn.Linear(latent_dim, latent_dim)

    def forward(
        self,
        z: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        z: (B, T, C) or (B, C)
        cond: optional conditioning passed to TransformerLayer (kept for interface compatibility)
        attention_mask: optional mask (True=allow). If None and causal=True,
                        a causal lower-triangular mask is built automatically.

        returns: predicted next latent (B, C)
        """
        if z.dim() == 2:
            z = z.unsqueeze(1)
        B, T, C = z.shape
        pos = self.pos_emb[:T].unsqueeze(0).to(z.device)
        x = self.input_proj(z) + pos
        mask = attention_mask
        if self.causal and mask is None:
            mask = torch.tril(torch.ones((T, T), dtype=torch.bool, device=z.device))
        for layer in self.layers:
            x = layer(x, cond=cond, attn_mask=mask)
        x = self.norm(x)
        out = self.out_head(x)  # (B, T, C)
        return out[:, -1, :]


class TransformerLayer(nn.Module):
    """standard pre-ln transformer block used as deterministic predictor in latent space"""
    def __init__(
        self,
        dim: int,
        n_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert self.head_dim * n_heads == dim, "dim must be divisible by n_heads"
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: (B, N, C)
        cond: optional conditioning sequence (B, M, C) or None for self-attention
        attn_mask: optional boolean mask indicating allowed key positions.
            Supported shapes (broadcast where possible):
              - (N, K) or (Q, K)
              - (B, Q, K)
              - (B, 1, Q, K) or (B, H, Q, K)
            True=allow attend, False=block.

        returns: (B, N, C)
        """
        B, N, C = x.shape
        # Pre-ln: normalise x before computing queries, keys, values
        x_norm = self.norm1(x)
        q = self.q(x_norm).reshape(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        if cond is not None:
            K_len = cond.shape[1]
            k = self.k(cond).reshape(B, K_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            v = self.v(cond).reshape(B, K_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        else:
            k = self.k(x_norm).reshape(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            v = self.v(x_norm).reshape(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # attention scores: (B, H, Q, K)
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            mask = attn_mask
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            elif mask.dim() == 4:
                pass
            else:
                raise ValueError(f"Unsupported attn_mask dim: {mask.dim()}")
            mask = mask.to(dtype=torch.bool, device=attn_scores.device)
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)
        # weighted sum → (B, H, Q, D)
        attn_output = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)
        x = x + self.proj_drop(self.proj(attn_output))
        x = x + self.mlp(self.norm2(x))
        return x


class DiffTransformer(nn.Module):
    """transformer-style predictor for diffusion models.

    interface: predictor(xt, t, cond) → same-shaped tensor as xt.
      xt:   (B, T, C) or (B, C) — if 2-D, treated as single-token sequence
      t:    (B,) tensor of timestep scalars (int or float)
      cond: optional conditioning sequence (B, M, C) or global vector (B, C)
    """
    def __init__(
        self,
        dim: int,
        n_layers: int = 4,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        time_emb_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.dim = dim
        self.n_layers = n_layers
        self.time_emb_dim = time_emb_dim if time_emb_dim is not None else dim
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_emb_dim, self.time_emb_dim * 2),
            nn.GELU(),
            nn.Linear(self.time_emb_dim * 2, dim),
        )
        self.layers = nn.ModuleList(
            [
                TransformerLayer(dim, n_heads=n_heads, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    @staticmethod
    def _timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
        """sinusoidal timestep embedding. t is (B,) float or int"""
        t = t.float()
        half = dim // 2
        emb = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32) * -(math.log(10000.0) / (half - 1))
        )
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(t.shape[0], 1, device=t.device)], dim=1)
        return emb

    def forward(
        self, xt: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        xt: (B, T, C) or (B, C)
        t: (B,) scalar timesteps
        cond: (B, M, C) or (B, C)

        returns tensor of the same shape as xt.
        """
        squeeze_output = False
        if xt.dim() == 2:
            xt = xt.unsqueeze(1)
            squeeze_output = True
        B, T, C = xt.shape
        assert C == self.dim, f"Input feature dim {C} must match model dim {self.dim}"
        t_emb = self._timestep_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb).unsqueeze(1)  # (B, 1, C)
        x = xt + t_emb
        cond_seq = None
        if cond is not None:
            cond_seq = cond.unsqueeze(1) if cond.dim() == 2 else cond
        for layer in self.layers:
            x = layer(x, cond=cond_seq)
        x = self.norm(x)
        if squeeze_output:
            return x[:, 0, :]
        return x
