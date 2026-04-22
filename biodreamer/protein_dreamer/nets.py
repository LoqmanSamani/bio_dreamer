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


    


    
        
        
    