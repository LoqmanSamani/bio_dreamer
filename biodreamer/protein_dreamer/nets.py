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