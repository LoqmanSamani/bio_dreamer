import math
import torch
import torch.nn as nn
from typing import Any, Optional, Tuple



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
        device: Optional[torch.device] = None) -> None:
        super().__init__()
        self.device = device if device is not None and isinstance(device, torch.device) else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.predictor = predictor.to(self.device)
        self.pred_type = pred_type if pred_type is not None else 'noise'  # default to noise prediction if not specified
        self.schedule_type = schedule_type if schedule_type is not None else 'linear'  # default to linear schedule if not specified
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.time_steps = time_steps
        self.cosine_s = cosine_s
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.clip_out = clip_out
        self.var_type = var_type
        
        valid_types = ["noise","x0", "v"]
        if self.pred_type not in valid_types:
            raise ValueError(f"prediction_type must be one of {valid_types}, got {self.pred_type}")
        valid_schedules = ["linear", "cosine"]
        if self.schedule_type not in valid_schedules:
            raise ValueError(f"schedule_type must be one of {valid_schedules}, got {self.schedule_type}")
        if self.schedule_type == "linear" and not (0.0 < self.beta_min < self.beta_max):
            raise ValueError("For linear schedule, require 0 < beta_min < beta_max")
        
        self._setup_schedule()
        
    def noise_step(self, x0: torch.Tensor, cond: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """one forward diffusion step for training - sample t, add noise, and get target based on pred_type"""
        x0 = x0.to(self.device)
        noise = torch.randn_like(x0)
        t = torch.randint(0, self.fwd_ddpm.vs.time_steps, (x0.shape[0],), device=x0.device)
        xt, target = self.forward_diff(x0, t, noise)
        pred = self.predictor(xt, t, cond)
        return pred, target
    
    @torch.inference_mode()
    def sample_step(self, xt: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """one reverse diffusion step for sampling - predict noise and compute x_{t-1}"""
        xt = xt.to(self.device)
        t = t.to(self.device)
        pred = self.predictor(xt, t, cond)
        x_prev, _ = self.reverse_diff(xt, t, pred)
        return x_prev
        
    @torch.inference_mode()
    def sample(self, shape: Tuple[int], cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """generate a sample by iteratively applying reverse diffusion steps starting from pure noise"""
        xt = torch.randn(shape, device=self.device)
        for t in reversed(range(self.time_steps)):
            t_batch = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            xt = self.sample_step(xt, t_batch, cond)
        return xt
    
    def forward_diff(
            self,
            x0: torch.Tensor,
            t: torch.Tensor,
            noise: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ddpm forward pass"""
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        sqrt_alpha_cumprod_t = self.get_index(sqrt_alpha_cumprod_t, x0.shape)
        sqrt_one_minus_alpha_cumprod_t = self.get_index(sqrt_one_minus_alpha_cumprod_t, x0.shape)
        # x_t ~ q(x_t | x_0)
        # x_t = √ᾱ_t * x_0 + √(1 - ᾱ_t) * ε
        xt = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise
        if self.pred_type == 'noise':
            target = noise
        elif self.pred_type == "x0":
            target = x0
        elif self.pred_type == "v":
            # v-prediction: v = √ᾱ_t * ε - √(1 - ᾱ_t) * x_0
            target = sqrt_alpha_cumprod_t * noise - sqrt_one_minus_alpha_cumprod_t * x0
        return xt, target
    
    def reverse_diff(
            self,
            xt: torch.Tensor,
            t: torch.Tensor,
            pred: torch.Tensor,
            pred_var: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """reverse diffusion step of ddpm"""
        # predict x_0 from model output
        pred_x0 = self.predict_x0(xt, t, pred)
        # get posterior mean coefficients
        coef1 = self.posterior_mean_coef1[t]
        coef2 = self.posterior_mean_coef2[t]
        coef1 = self.get_index(coef1, xt.shape)
        coef2 = self.get_index(coef2, xt.shape)
        # posterior mean: μ_θ(x_t, t) = coef1 * x_0 + coef2 * x_t
        posterior_mean = coef1 * pred_x0 + coef2 * xt
        # variance
        if self.var_type == "fixed_small":
            # use precomputed sqrt for fixed_small (most common case)
            sqrt_var = self.sqrt_posterior_variance[t]
            sqrt_var = self.get_index(sqrt_var, xt.shape)
        elif self.var_type == "fixed_large":
            sqrt_var = self.sqrt_betas[t]
            sqrt_var = self.get_index(sqrt_var, xt.shape)
        else:
            variance = self.get_variance(t, pred_var)
            variance = self.get_index(variance, xt.shape)
            sqrt_var = torch.sqrt(variance)
        # sample noise (no noise for t=0)
        noise = torch.randn_like(xt)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(xt.shape) - 1)))
        # sample x_{t-1} ~ p_θ(x_{t-1} | x_t)
        x_prev = posterior_mean + nonzero_mask * sqrt_var * noise
        return x_prev, pred_x0
    
    def _setup_schedule(self) -> None:
        """ddpm noise schedule setup"""
        if self.schedule_type == "linear":
            betas = torch.linspace(self.beta_min, self.beta_max, self.time_steps)
        elif self.schedule_type == "cosine":
            steps = self.time_steps + 1
            t = torch.linspace(0, self.time_steps, steps)
            alphas_cumprod = torch.cos(((t / self.time_steps) + self.cosine_s) / (1 + self.cosine_s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, self.clip_min, self.clip_max)
        else:
            raise ValueError(f"Unsupported schedule type: {self.schedule_type}")
        # compute alphas
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        # compute coefficients for q(x_t | x_0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        # compute coefficients for q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_log_variance = torch.log(torch.clamp(posterior_variance, min=1e-20))
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        # precompute square roots for reverse step efficiency
        sqrt_posterior_variance = torch.sqrt(torch.clamp(posterior_variance, min=1e-20))
        sqrt_betas = torch.sqrt(betas)
        # register as buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance', posterior_log_variance)
        self.register_buffer('posterior_mean_coef1', posterior_mean_coef1)
        self.register_buffer('posterior_mean_coef2', posterior_mean_coef2)
        self.register_buffer('sqrt_posterior_variance', sqrt_posterior_variance)
        self.register_buffer('sqrt_betas', sqrt_betas)
        
    def get_index(self, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """get index for batch processing"""
        batch_size = t.shape[0]
        return t.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    def predict_x0(self, xt: torch.Tensor, t: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """Predict x_0 from model output based on the specified prediction type"""
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        sqrt_alpha_cumprod_t = self.get_index(sqrt_alpha_cumprod_t, xt.shape)
        sqrt_one_minus_alpha_cumprod_t = self.get_index(sqrt_one_minus_alpha_cumprod_t, xt.shape)
        if self.pred_type == "noise":
            # x_0 = (x_t - √(1 - ᾱ_t) * ε_θ) / √ᾱ_t
            x0 = (xt - sqrt_one_minus_alpha_cumprod_t * pred) / sqrt_alpha_cumprod_t
        elif self.pred_type == "x0":
            # directly predict x_0
            x0 = pred
        elif self.pred_type == "v":
            # x_0 = √ᾱ_t * x_t - √(1 - ᾱ_t) * v_θ
            x0 = sqrt_alpha_cumprod_t * xt - sqrt_one_minus_alpha_cumprod_t * pred
        if self.clip_out:
            x0 = torch.clamp(x0, -1.0, 1.0)
        return x0

    def get_variance(self, t: torch.Tensor, pred_var: Optional[torch.Tensor] = None) -> torch.Tensor:
        """get variance for reverse diffusion step based on the specified variance type"""
        if self.var_type == "fixed_small":
            # posterior variance: β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
            var = self.posterior_variance[t]
        elif self.var_type == "fixed_large":
            # β_t
            var = self.betas[t]
        elif self.var_type == "learned":
            # model-predicted variance
            if pred_var is None:
                raise ValueError("predicted_variance must be provided when variance_type='learned'")
            # interpolate between fixed_small and fixed_large
            min_log = self.posterior_log_variance[t]
            max_log = torch.log(self.betas[t])
            frac = (pred_var + 1) / 2  # map from [-1, 1] to [0, 1]
            var = torch.exp(frac * max_log + (1 - frac) * min_log)
        return var
    
    
    
    
class SDE(nn.Module):
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
        device: Optional[torch.device] = None   
        ) -> None:
        super().__init__()
        self.device = device if device is not None and isinstance(device, torch.device) else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))   
        self.predictor = predictor.to(self.device)
        self.method = method
        self.pred_type = pred_type
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.time_eps = time_eps
        self.num_steps = num_steps
        self.cosine_s = cosine_s
        self.schedule_type = schedule_type
        
        valid_methods = ["vp", "ve", "sub-vp", "ode"]
        if self.method not in valid_methods:
            raise ValueError(f"sde_method must be one of {valid_methods}, got {self.method}")
        valid_types = ["noise", "score", "v"]
        if self.pred_type not in valid_types:
            raise ValueError(f"pred_type must be one of {valid_types}, got {self.pred_type}")
        
        valid_schedules = ["linear", "cosine"]
        if self.schedule_type not in valid_schedules:
            raise ValueError(f"schedule_type must be one of {valid_schedules}, got {self.schedule_type}")
        if self.schedule_type == "linear" and not (0.0 < self.beta_min < self.beta_max):
            raise ValueError("For linear schedule, require 0 < beta_min < beta_max")  
        
    def noise_step(self, x0: torch.Tensor, cond: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """one forward diffusion step for training - sample t, add noise, and get target based on pred_type"""
        noise = torch.randn_like(x0)
        t = self.sample_time(x0.shape[0], self.time_eps)
        xt, target = self.forward_diff(x0, t, noise)
        pred = self.predictor(xt, t, cond)
        return pred, target
    
    @torch.inference_mode()
    def sample_step(self, xt: torch.Tensor, t: torch.Tensor, dt: float, cond: Optional[torch.Tensor] = None, last_step: bool = False) -> torch.Tensor:
        """one reverse diffusion step for sampling - predict noise and compute x_{t-1}"""
        pred = self.score_net(xt, t, cond)
        xt = self.reverse_diff(xt, pred, t, dt, last_step = last_step)
        return xt
        
    @torch.inference_mode()
    def sample(self, shape: Tuple[int], cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """generate a sample by iteratively applying reverse diffusion steps starting from pure noise"""
        xt = torch.randn(shape, device=self.device)
        t_schedule = torch.linspace(1.0, self.time_eps, self.num_steps + 1, device=self.device)
        dt = -(1.0 - self.time_eps) / self.num_steps
        for step in range(self.num_steps):
            t_current = float(t_schedule[step])
            t_batch = torch.full((self.batch_size,), t_current, dtype=xt.dtype, device=self.device)
            last_step = (step == self.num_steps - 1)
            xt  = self.sample_step(xt, t_batch, dt, cond, last_step = last_step)
        return xt

    def forward_diff(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """SDE forward pass: compute x_t and target based on the specified prediction type"""
        mean_coeff, std = self.get_forward_params(t)
        # broadcast to match x0 shape
        mean_coeff = self._broadcast_to_shape(mean_coeff, x0.shape)
        std = self._broadcast_to_shape(std, x0.shape)
        # x_t = mean_coeff * x_0 + std * ε
        xt = mean_coeff * x0 + std * noise
        if self.pred_type == 'noise':
            target = noise
        elif self.pred_type == "score":
            # ∇_x log p(x_t | x_0) = -(x_t - mean_coeff*x_0) / σ²(t) = -ε / σ(t)
            target = -noise / (std + self.eps)
        elif self.pred_type == "v":
            # v = mean_coeff * ε - std * x_0
            target = mean_coeff * noise - std * x0
        return xt, target
    
    def reverse_diff(self, xt: torch.Tensor, pred: torch.Tensor, t: torch.Tensor, dt: float, last_step: bool = False) -> torch.Tensor:
        """sde/ode reverse pass: compute x_{t-dt} from x_t and model prediction based on the specified method and prediction type"""
        if not torch.is_tensor(dt):
            assert dt < 0.0, "dt must be negative for reverse diffusion!"
            dt = torch.tensor(dt, device=xt.device, dtype=xt.dtype)
        drift_coeff, g_squared, diffusion_coeff = self.get_reverse_coeffs(t)
        # broadcast to match xt shape
        drift_coeff = self._broadcast_to_shape(drift_coeff, xt.shape)
        g_squared = self._broadcast_to_shape(g_squared, xt.shape)
        diffusion_coeff = self._broadcast_to_shape(diffusion_coeff, xt.shape)
        if self.method == "ve":
            sigma_t = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
            std = sigma_t
        else:
            std = self.vs.std(t)
        while std.dim() < len(xt.shape):
            std = std.unsqueeze(-1)
        if self.pred_type == "noise":
            score = -pred / (std + self.eps)
        elif self.pred_type == "score":
            score = pred
        # [-½β(t)x - β(t)∇log p_t(x)]dt + √β(t)dw̄
        # reverse drift: f(x,t) - g²(t)·score
        if self.method == 'ode':
            drift = drift_coeff * xt - 0.5 * g_squared * score
        else:
            drift = drift_coeff * xt - g_squared * score
        if last_step or self.method == "ode":
            noise = torch.zeros_like(xt)
        else:
            noise = torch.randn_like(xt)
        diffusion = diffusion_coeff * noise
        # Euler-Maruyama step
        x_prev = xt + drift * dt + diffusion * torch.sqrt(torch.abs(dt))
        return x_prev
    
    def _broadcast_to_shape(self, tensor: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        """broadcast tensor to target shape by adding trailing dimensions"""
        while tensor.dim() < len(target_shape):
            tensor = tensor.unsqueeze(-1)
        return tensor

    def get_forward_params(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """get mean coefficient and std for the forward process based on SDE method"""
        mean_coeff = None
        std = None
        if self.method == "vp":
            # vp-sde: p(x_t | x_0) = N(α(t)x_0, σ²(t)I)
            mean_coeff = self.alpha(t)
            std = self.std(t)
        elif self.method == "ve":
            # ve-sde: p(x_t | x_0) = N(x_0, σ²(t)I)
            # σ(t) grows from sigma_min to sigma_max
            mean_coeff = torch.ones_like(t)
            sigma_t = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
            std = sigma_t
        elif self.method == "sub-vp":
            # sub-vp-sde: p(x_t | x_0) = N(x_0, σ²(t)I) where σ²(t) = 1 - e^(-∫β(s)ds)
            mean_coeff = torch.ones_like(t)
            std = self.std(t)
        elif self.method == "ode":
            # probability flow ode: same marginals as vp-sde but deterministic
            mean_coeff = self.alpha(t)
            std = self.std(t)
        return mean_coeff, std
    
    def get_reverse_coeffs(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """get drift and diffusion coefficients for the reverse SDE based on the specified method"""
        if self.method == "vp":
            # vp-sde: dx = [-½β(t)x - β(t)∇log p_t(x)]dt + √β(t)dw̄
            drift_coeff = -0.5 * self.beta(t)
            g_squared = self.beta(t)
            diffusion_coeff = torch.sqrt(self.beta(t))
        elif self.method == "ve":
            # ve-sde: dx = [-σ(t)dσ/dt ∇log p_t(x)]dt + √(2σ(t)dσ/dt)dw̄
            sigma_t = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
            dsigma_dt = sigma_t * torch.log(torch.tensor(self.sigma_max / self.sigma_min))
            drift_coeff = torch.zeros_like(t)
            g_squared = 2 * sigma_t * dsigma_dt
            diffusion_coeff = torch.sqrt(g_squared)
        elif self.method == "sub-vp":
            # sub-vp-sde: dx = [-β(t)∇log p_t(x)]dt + √β(t)dw̄
            drift_coeff = torch.zeros_like(t)
            g_squared = self.beta(t)
            diffusion_coeff = torch.sqrt(self.beta(t))
        elif self.method == "ode":
            # probability flow ode: deterministic
            drift_coeff = -0.5 * self.beta(t)
            g_squared = self.beta(t)
            diffusion_coeff = torch.zeros_like(t) # no diffusion in ode
        return drift_coeff, g_squared, diffusion_coeff
    
    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """β(t) - noise schedule"""
        if self.schedule_type == "linear":
            return self.beta_min + t * (self.beta_max - self.beta_min)
        elif self.schedule_type == "cosine":
            # β(t) = -d/dt log ᾱ(t) = tan(x) * π / (1+s)
            t_mapped = (t + self.cosine_s) / (1 + self.cosine_s) * torch.pi / 2
            beta_t = torch.tan(t_mapped) * (torch.pi / (1 + self.cosine_s))
            return torch.clamp(beta_t, min=0.0, max=1000.0)

    def integral_beta(self, t: torch.Tensor) -> torch.Tensor:
        """∫₀ᵗ β(s) ds"""
        if self.schedule_type == "linear":
            return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2
        elif self.schedule_type == "cosine":
            return -torch.log(self.alpha_squared(t))

    def _cosine_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """ᾱ(t) = cos²((t+s)/(1+s) · π/2) for cosine schedule"""
        return torch.cos((t + self.cosine_s) / (1 + self.cosine_s) * torch.pi / 2) ** 2

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """α(t) = exp(-½∫₀ᵗ β(s) ds)"""
        if self.schedule_type == "cosine":
            return torch.sqrt(self.alpha_squared(t))
        return torch.exp(-0.5 * self.integral_beta(t))

    def alpha_squared(self, t: torch.Tensor) -> torch.Tensor:
        """α²(t) = exp(-∫₀ᵗ β(s) ds)"""
        if self.schedule_type == "cosine":
            return self._cosine_alpha_bar(t) / self._cosine_alpha_bar(torch.zeros_like(t))
        return torch.exp(-self.integral_beta(t))

    def variance(self, t: torch.Tensor) -> torch.Tensor:
        """σ²(t) = 1 - α²(t)"""
        return 1.0 - self.alpha_squared(t)

    def std(self, t: torch.Tensor) -> torch.Tensor:
        """σ(t) = √(1 - α²(t))"""
        return torch.sqrt(self.variance(t))

    def snr(self, t: torch.Tensor) -> torch.Tensor:
        """signal-to-noise ratio: SNR(t) = α²(t) / σ²(t)"""
        alpha_sq = self.alpha_squared(t)
        var = self.variance(t)
        return alpha_sq / (var + 1e-8)
    
    def sample_time(self, batch_size: int, eps: float = 1e-5) -> torch.Tensor:
        return eps + (1 - eps) * torch.rand(batch_size, device=self.device)
    
    

class GVP_GNN(nn.Module):
    """
    GVP-GNN encoder stack for protein structure embedding.
    - produces per-node scalar+vector embeddings preserving geometric equivariance.
    - intended as a structure encoder/embedder: given residue-level scalar and vector inputs
      plus edge features (e.g. relative displacement vectors, distances, edge-type scalars),
      returns updated per-residue embeddings.
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
        # initial embeddings for nodes and edges -> hidden dims
        self.node_embed = GVP(in_node_dims, hidden_dims, vector_dim=vector_dim)
        self.edge_embed = GVP(in_edge_dims, hidden_dims, vector_dim=vector_dim)

        # choose convolution / message-passing implementation
        self.conv_type = conv_type
        layers = []
        if conv_type == "gvp":
            for _ in range(n_layers):
                layers.append(GVPConv(hidden_dims, hidden_dims, hidden_dims, vector_dim=vector_dim))
        elif conv_type == "transformer":
            # ensure hidden scalar dim is divisible by heads
            hidden_s = hidden_dims[0]
            if hidden_s % n_heads != 0:
                # adjust head count to divide hidden dim
                n_heads = max(1, math.gcd(hidden_s, n_heads))
            for _ in range(n_layers):
                layers.append(GraphTransformerLayer(
                    node_dims=hidden_dims,
                    edge_dims=hidden_dims,
                    hidden_dim=hidden_dims[0],
                    n_heads=n_heads,
                    dropout=dropout,
                    vector_dim=vector_dim
                ))
        else:
            raise ValueError(f"Unsupported conv_type: {conv_type}. Use 'gvp' or 'transformer'.")

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
        edge_s: (e, in_edge_scalar) or none
        edge_v: (e, in_edge_vec, 3) or none

        returns:
            s: (n, hidden_scalar)
            v: (n, hidden_vec, 3)
        """
        # embed inputs into hidden dims
        s_h, v_h = self.node_embed(s, v)
        if edge_s is None:
            edge_s = torch.zeros(edge_index.shape[1], 0, device=s.device, dtype=s.dtype)
        if edge_v is None:
            edge_v = torch.zeros(edge_index.shape[1], 0, self.vector_dim, device=s.device, dtype=s.dtype)
        s_e, v_e = self.edge_embed(edge_s, edge_v)

        # pass through GVPConv layers
        for layer in self.layers:
            s_h, v_h = layer(s_h, v_h, edge_index, s_e, v_e)
        return s_h, v_h
    
    
class GVPConv(nn.Module):
    """
    single GVP convolution/message-passing layer.
    - message_gvp maps (sender node features + edge features) -> message (s_msg, v_msg)
    - node_gvp maps (node_features + aggregated_messages) -> updated node features

    expected input shapes:
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
        senders, receivers = edge_index[0], edge_index[1]  # (e,), (e,)
        s_src = s[senders]       # (e, node_s)
        v_src = v[senders]       # (e, node_v, 3)
        if edge_s is None:
            edge_s = torch.zeros(s_src.shape[0], 0, device=s.device, dtype=s.dtype)
        if edge_v is None:
            edge_v = torch.zeros(s_src.shape[0], 0, self.vector_dim, device=s.device, dtype=s.dtype)
        # message input
        s_msg_in = torch.cat([s_src, edge_s], dim=-1) if edge_s.shape[-1] > 0 else s_src
        v_msg_in = torch.cat([v_src, edge_v], dim=1) if (v_src.shape[1] > 0 or edge_v.shape[1] > 0) else v_src
        # compute messages
        m_s, m_v = self.message_gvp(s_msg_in, v_msg_in)  # (e, msg_s), (e, msg_v, 3)
        # aggregate messages per receiver (sum)
        n = s.shape[0]
        device = s.device
        agg_s = torch.zeros(n, m_s.shape[-1], device=device, dtype=m_s.dtype)
        agg_s = agg_s.index_add(0, receivers, m_s)
        # aggregate vector messages: shape (n, msg_v, 3)
        if m_v is not None:
            agg_v = torch.zeros(n, m_v.shape[1], self.vector_dim, device=device, dtype=m_v.dtype)
            agg_v = agg_v.index_add(0, receivers, m_v)
        else:
            agg_v = torch.zeros(n, 0, self.vector_dim, device=device, dtype=s.dtype)
        # combine aggregated messages with node features and update
        s_comb = torch.cat([s, agg_s], dim=-1) if agg_s.shape[-1] > 0 else s
        v_comb = torch.cat([v, agg_v], dim=1) if v.shape[1] + agg_v.shape[1] > 0 else v
        s_upd, v_upd = self.node_gvp(s_comb, v_comb)
        # residual add if shapes match
        if s_upd.shape[-1] == s.shape[-1]:
            s_out = s + s_upd
        else:
            s_out = s_upd
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
        vector_dim: int = 3
    ) -> None:
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers.append(
                GraphTransformerLayer(
                    node_dims=node_dims, 
                    edge_dims=edge_dims, 
                    hidden_dim=hidden_dim, 
                    n_heads=n_heads, 
                    dropout=dropout, 
                    vector_dim=vector_dim
                    )
                )
        self.layers = nn.ModuleList(layers)

    def forward(
        self, 
        s: torch.Tensor, 
        v: torch.Tensor, 
        edge_index: torch.LongTensor, 
        edge_s: Optional[torch.Tensor] = None, 
        edge_v: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        s_h, v_h = s, v
        for layer in self.layers:
            s_h, v_h = layer(s_h, v_h, edge_index, edge_s, edge_v)
        return s_h, v_h



class GraphTransformerLayer(nn.Module):
    """graph-transformer style message-passing layer with multi-head attention and optional edge biasing"""
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
        # projections for attention (operate on scalars + vector norms)
        self.q = nn.Linear(in_s + in_v, hidden_dim)
        self.k = nn.Linear(in_s + in_v, hidden_dim)
        self.v = nn.Linear(in_s + in_v, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, in_s)
        # optional small network to incorporate edge scalar features into attention bias
        self.edge_att = nn.Linear(edge_s_dim, n_heads) if edge_s_dim > 0 else None
        # vector channel mixer (map sender node vectors -> message vectors)
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
            nn.Linear(max(in_s * 2, 4), in_s)
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self, 
        s: torch.Tensor, 
        v: torch.Tensor, 
        edge_index: torch.LongTensor, 
        edge_s: Optional[torch.Tensor], 
        edge_v: Optional[torch.Tensor]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        s: (n, in_s)
        v: (n, in_v, 3)
        edge_index: (2, e)
        edge_s: (e, edge_s_dim) or none
        edge_v: (e, edge_v_dim, 3) or none

        returns updated (s_out, v_out) with the same shapes as input
        """
        device = s.device
        senders, receivers = edge_index[0], edge_index[1]
        n = s.shape[0]
        if self.in_v > 0 and v is not None:
            v_norm = torch.sqrt((v ** 2).sum(dim=-1) + 1e-8)  # (n, in_v)
        else:
            v_norm = torch.zeros(n, 0, device=device, dtype=s.dtype)
        s_input = torch.cat([s, v_norm], dim=-1)
        # compute per-node q/k/v and reshape into heads
        q = self.q(s_input).view(n, self.n_heads, self.head_dim)
        k = self.k(s_input).view(n, self.n_heads, self.head_dim)
        v_val = self.v(s_input).view(n, self.n_heads, self.head_dim)
        # gather per-edge sender/receiver projections
        q_j = q[receivers]  # (e, heads, head_dim)
        k_i = k[senders]
        v_i = v_val[senders]
        # attention scores per head
        scores = (q_j * k_i).sum(dim=-1) / math.sqrt(self.head_dim)  # (e, heads)
        if self.edge_att is not None and edge_s is not None:
            edge_bias = self.edge_att(edge_s)  # (e, heads)
            scores = scores + edge_bias

        exp_scores = torch.exp(scores)
        denom = torch.zeros(n, self.n_heads, device=device, dtype=exp_scores.dtype)
        denom.index_add_(0, receivers, exp_scores)
        denom = denom + 1e-8
        weights = exp_scores / denom[receivers]

        weighted_v = v_i * weights.unsqueeze(-1)  # (e, heads, head_dim)
        agg = torch.zeros(n, self.n_heads, self.head_dim, device=device, dtype=weighted_v.dtype)
        agg.index_add_(0, receivers, weighted_v)
        agg = agg.view(n, self.hidden_dim)

        msg = self.out_proj(agg)
        s_updated = s + self.dropout(msg)
        s_updated = self.norm1(s_updated)

        ff = self.ff(s_updated)
        s_out = s_updated + self.dropout(ff)
        s_out = self.norm2(s_out)

        if self.in_v > 0 and v is not None:
            v_src = v[senders]  # (e, in_v, 3)
            v_msg = torch.einsum('...ij,oj->...oi', v_src, self.w_v) if self.w_v is not None else v_src
            w_mean = weights.mean(dim=1)  # (e,)
            weighted_v_msg = v_msg * w_mean.view(-1, 1, 1)
            agg_v = torch.zeros(n, v_msg.shape[1], self.vector_dim, device=device, dtype=v_msg.dtype)
            agg_v.index_add_(0, receivers, weighted_v_msg)
            if agg_v.shape[1] == v.shape[1]:
                v_out = v + agg_v
            else:
                v_out = agg_v
        else:
            v_out = v
        return s_out, v_out


 
class GVP(nn.Module):
    """
    Geometric Vector Perceptron (GVP) core block.
    - inputs:
        s: (..., n_s)         scalar features per node/edge
        v: (..., n_v, 3)      vector features per node/edge (3d vectors)
    - outputs:
        s_out: (..., n_s_out)
        v_out: (..., n_v_out, 3)

    design notes:
    - vector features are linearly combined across channels (no bias on 3d axis)
      to preserve equivariance to rotations. we follow the original GVP idea:
        v' = W_v · v  (channel mixing)
        v_norm = ||v'||_2  (per-channel norms become scalar inputs)
        s' = Linear([s, v_norm]) -> s_out (scalar path)
        gate = sigmoid(Linear(s')) -> gate vector applied to v'
    """
    def __init__(
        self,
        in_dims: Tuple[int, int],
        out_dims: Tuple[int, int],
        vector_dim: int = 3,
        scalar_act: Optional[nn.Module] = None,
        use_layernorm: bool = False,
        dropout: float = 0.0,
        eps: float = 1e-8
    ) -> None:
        super().__init__()
        in_s, in_v = in_dims
        out_s, out_v = out_dims
        self.in_s, self.in_v = in_s, in_v
        self.out_s, self.out_v = out_s, out_v
        self.vector_dim = vector_dim
        self.use_layernorm = use_layernorm

        # vector channel mixing (out_v x in_v). no bias on 3d coordinates to preserve equivariance
        if in_v > 0 and out_v > 0:
            self.w_v = nn.Parameter(torch.empty(out_v, in_v))
            nn.init.xavier_uniform_(self.w_v)
        else:
            self.w_v = None

        # scalar path: input is original scalars concatenated with vector norms (if any)
        scalar_in = in_s + (out_v if (in_v > 0 and out_v > 0) else 0)
        if out_s > 0:
            self.linear_s = nn.Linear(scalar_in, out_s)
            nn.init.xavier_uniform_(self.linear_s.weight)
            if self.linear_s.bias is not None: 
                nn.init.zeros_(self.linear_s.bias)
        else:
            self.linear_s = None

        # gate for vector output produced from scalar features (or scalar projection)
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

    def forward(self, s: torch.Tensor, v: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        s: (..., in_s)
        v: (..., in_v, 3) or none
        returns:
            s_out: (..., out_s) or torch.empty(..., 0) if out_s==0
            v_out: (..., out_v, 3) or none
        """
        v_lin = None
        v_norm = None
        if self.in_v > 0 and v is not None:
            v_lin = torch.einsum('...ij,oj->...oi', v, self.w_v)  # (..., out_v, 3)
            v_norm = torch.sqrt((v_lin ** 2).sum(dim=-1) + self._eps)
        elif self.out_v > 0:
            batch_shape = s.shape[:-1]
            v_lin = torch.zeros(*batch_shape, self.out_v, self.vector_dim, device=s.device, dtype=s.dtype)
            v_norm = torch.zeros(*batch_shape, self.out_v, device=s.device, dtype=s.dtype)

        # scalar path
        if self.in_s > 0:
            s_in = s
        else:
            s_in = torch.zeros(*s.shape[:-1], 0, device=s.device, dtype=s.dtype)
            
        if v_norm is not None and v_norm.shape[-1] > 0:
            s_cat = torch.cat([s_in, v_norm], dim=-1)
        else:
            s_cat = s_in

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

        # vector gating and final vector output
        v_out = None
        if self.gate is not None:
            gate_input = s_out if (s_out is not None) else s_cat
            gates = torch.sigmoid(self.gate(gate_input))  # (..., out_v)
            v_out = v_lin * gates.unsqueeze(-1)
        elif v_lin is not None:
            v_out = v_lin

        return s_out, v_out
    
    
    
    
class DeterministicPredictor(nn.Module):
    """
    autoregressive/deterministic latent predictor using a stack of TransformerLayer blocks

    behavior:
      - input: sequence of latent tokens `z` of shape (B, T, C) or single step (B, C).
      - when `causal=True` (default), a causal mask is applied so each position may only
        attend to previous positions (autoregressive).
      - returns predicted next-token latent of shape (B, C) (prediction for position T -> T+1).
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
        # learned positional embeddings
        self.pos_emb = nn.Parameter(torch.zeros(max_len, latent_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        
        self.layers = nn.ModuleList([
            TransformerLayer(latent_dim, n_heads=n_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(latent_dim)
        self.out_head = nn.Linear(latent_dim, latent_dim)

    def forward(
        self, 
        z: torch.Tensor, 
        cond: Optional[torch.Tensor] = None, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        z: (B, T, C) or (B, C)
        cond: optional conditioning passed to TransformerLayer (kept for compatibility)
        attention_mask: optional mask for attention (True=allow). if None and causal=True,
                        a causal lower-triangular mask is constructed automatically.

        returns: predicted next latent (B, C)
        """
        if z.dim() == 2:
            z = z.unsqueeze(1)
        B, T, C = z.shape
        pos = self.pos_emb[:T].unsqueeze(0).to(z.device)
        x = self.input_proj(z) + pos
        mask = attention_mask
        if self.causal and mask is None:
            causal_mask = torch.tril(torch.ones((T, T), dtype=torch.bool, device=z.device))
            mask = causal_mask
        for layer in self.layers:
            x = layer(x, cond=cond, attn_mask=mask)
        x = self.norm(x)
        out = self.out_head(x)  # (B, T, C)
        return out[:, -1, :]
    
    
    

class TransformerLayer(nn.Module):
    """standard transformer block used as deterministic predictor in latent space"""
    def __init__(
        self, 
        dim: int, 
        n_heads: int = 4, 
        mlp_ratio: float = 4.0, 
        dropout: float = 0.0
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
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, N, C)
        cond: optional conditioning sequence (B, M, C) or None
        attn_mask: optional boolean mask that indicates allowed key positions.
            Supported shapes (will be broadcast where possible):
              - (N, K) or (Q, K)
              - (B, Q, K)
              - (B, 1, Q, K) or (B, H, Q, K)
            Mask True=allow attend, False=block.

        returns: (B, N, C)
        """
        B, N, C = x.shape
        # queries from target positions
        q = self.q(x).reshape(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, H, Q, D)
        # keys/values may come from conditioning sequence of potentially different length
        if cond is not None:
            K_len = cond.shape[1]
            k = self.k(cond).reshape(B, K_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, H, K, D)
            v = self.v(cond).reshape(B, K_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        else:
            K_len = N
            k = self.k(x).reshape(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            v = self.v(x).reshape(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # attention scores: (B, H, Q, K)
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # apply optional attention mask
        if attn_mask is not None:
            mask = attn_mask
            if mask.dim() == 2:
                # (Q, K) -> (1, 1, Q, K)
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                # (B, Q, K) -> (B, 1, Q, K)
                mask = mask.unsqueeze(1)
            elif mask.dim() == 4:
                # (B, H, Q, K) use as-is
                pass
            else:
                raise ValueError(f"Unsupported attn_mask dim: {mask.dim()}")
            mask = mask.to(dtype=torch.bool, device=attn_scores.device)
            # mask broadcasting will align batch/heads where possible
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)
        # weighted sum -> (B, H, Q, D)
        attn_output = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)
        x = x + self.proj_drop(self.proj(attn_output))
        x = x + self.mlp(self.norm2(x))
        return x


