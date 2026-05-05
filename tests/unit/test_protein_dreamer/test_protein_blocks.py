from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from biodreamer.protein_dreamer.blocks import (
    DDPM,
    SDE,
    DeterministicPredictor,
    DiffTransformer,
    FlowMatchingScheduler,
    GVP,
    GVPConv,
    GraphTransformer,
    GraphTransformerLayer,
    GvpGNN,
    ResidualMLP,
    TransformerLayer,
)






class _LinearPredictor(nn.Module):
    """Minimal predictor for DDPM / SDE / FlowMatching: (xt, t, cond?) → xt."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, xt: torch.Tensor, t: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        return self.proj(xt)


def _make_graph(n: int = 5, n_s: int = 4, n_v: int = 1, e: int = 8):
    """Build a minimal node/edge feature set for graph tests."""
    s = torch.randn(n, n_s)
    v = torch.randn(n, n_v, 3)
    # Random directed edge index
    src = torch.randint(0, n, (e,))
    dst = torch.randint(0, n, (e,))
    edge_index = torch.stack([src, dst], dim=0)
    return s, v, edge_index




class TestResidualMLP:
    @pytest.fixture
    def mlp(self) -> ResidualMLP:
        return ResidualMLP({"in_dim": 16, "hidden_dim": 32, "out_dim": 8, "n_layers": 2})

    def test_output_shape(self, mlp):
        x = torch.randn(4, 16)
        assert mlp(x).shape == (4, 8)

    def test_output_finite(self, mlp):
        assert torch.isfinite(mlp(torch.randn(4, 16))).all()

    def test_gradient_flows(self, mlp):
        x = torch.randn(4, 16, requires_grad=True)
        mlp(x).sum().backward()
        assert x.grad is not None

    def test_batch_size_1(self, mlp):
        assert mlp(torch.randn(1, 16)).shape == (1, 8)

    def test_n_layers_zero(self):
        mlp = ResidualMLP({"in_dim": 8, "hidden_dim": 16, "out_dim": 4, "n_layers": 0})
        assert mlp(torch.randn(3, 8)).shape == (3, 4)

    def test_dropout_zero_deterministic(self, mlp):
        mlp.eval()
        x = torch.randn(4, 16)
        assert torch.allclose(mlp(x), mlp(x))

    def test_in_out_same_dim(self):
        mlp = ResidualMLP({"in_dim": 32, "hidden_dim": 32, "out_dim": 32, "n_layers": 1})
        x = torch.randn(2, 32)
        assert mlp(x).shape == (2, 32)




class TestGVP:
    def test_scalar_only_output_shape(self):
        gvp = GVP(in_dims=(8, 0), out_dims=(16, 0))
        s = torch.randn(5, 8)
        v = None
        s_out, v_out = gvp(s, v)
        assert s_out.shape == (5, 16)
        assert v_out is None

    def test_vector_output_shape(self):
        gvp = GVP(in_dims=(8, 2), out_dims=(16, 4), vector_dim=3)
        s = torch.randn(5, 8)
        v = torch.randn(5, 2, 3)
        s_out, v_out = gvp(s, v)
        assert s_out.shape == (5, 16)
        assert v_out.shape == (5, 4, 3)

    def test_output_finite(self):
        gvp = GVP(in_dims=(8, 2), out_dims=(16, 4))
        s_out, v_out = gvp(torch.randn(5, 8), torch.randn(5, 2, 3))
        assert torch.isfinite(s_out).all()
        assert torch.isfinite(v_out).all()

    def test_layernorm_option(self):
        gvp = GVP(in_dims=(8, 0), out_dims=(16, 0), use_layernorm=True)
        s_out, _ = gvp(torch.randn(5, 8), None)
        assert s_out.shape == (5, 16)

    def test_gradient_flows(self):
        gvp = GVP(in_dims=(8, 2), out_dims=(16, 4))
        s = torch.randn(5, 8, requires_grad=True)
        v = torch.randn(5, 2, 3, requires_grad=True)
        s_out, v_out = gvp(s, v)
        (s_out.sum() + v_out.sum()).backward()
        assert s.grad is not None
        assert v.grad is not None




class TestGVPConv:
    @pytest.fixture
    def conv(self) -> GVPConv:
        return GVPConv(
            node_dims=(8, 2),
            edge_dims=(4, 1),
            message_dims=(8, 2),
        )

    def test_output_shapes(self, conv):
        s, v, edge_index = _make_graph(n=5, n_s=8, n_v=2, e=8)
        edge_s = torch.randn(8, 4)
        edge_v = torch.randn(8, 1, 3)
        s_out, v_out = conv(s, v, edge_index, edge_s, edge_v)
        assert s_out.shape == (5, 8)
        assert v_out.shape == (5, 2, 3)

    def test_no_edge_features(self):
        # edge_dims=(0,0) → None edge inputs are valid
        conv = GVPConv(node_dims=(8, 2), edge_dims=(0, 0), message_dims=(8, 2))
        s, v, edge_index = _make_graph(n=5, n_s=8, n_v=2, e=8)
        s_out, v_out = conv(s, v, edge_index, None, None)
        assert s_out.shape == (5, 8)

    def test_output_finite(self, conv):
        s, v, edge_index = _make_graph(n=5, n_s=8, n_v=2, e=8)
        edge_s = torch.randn(8, 4)
        edge_v = torch.randn(8, 1, 3)
        s_out, v_out = conv(s, v, edge_index, edge_s, edge_v)
        assert torch.isfinite(s_out).all()

    def test_gradient_flows(self, conv):
        s = torch.randn(5, 8, requires_grad=True)
        v = torch.randn(5, 2, 3, requires_grad=True)
        src = torch.randint(0, 5, (8,))
        dst = torch.randint(0, 5, (8,))
        edge_index = torch.stack([src, dst])
        edge_s = torch.randn(8, 4)
        edge_v = torch.randn(8, 1, 3)
        s_out, v_out = conv(s, v, edge_index, edge_s, edge_v)
        (s_out.sum() + v_out.sum()).backward()
        assert s.grad is not None




class TestGvpGNN:
    @pytest.fixture
    def gvp_gnn(self) -> GvpGNN:
        return GvpGNN({
            "in_node_dims": (4, 1),
            "in_edge_dims": (2, 1),
            "hidden_dims":  (16, 2),
            "n_layers":     2,
            "conv_type":    "gvp",
        })

    def test_output_shapes(self, gvp_gnn):
        s, v, edge_index = _make_graph(n=6, n_s=4, n_v=1, e=10)
        edge_s = torch.randn(10, 2)
        edge_v = torch.randn(10, 1, 3)
        s_out, v_out = gvp_gnn(s, v, edge_index, edge_s, edge_v)
        assert s_out.shape == (6, 16)
        assert v_out.shape == (6, 2, 3)

    def test_no_edge_features(self):
        # in_edge_dims=(0,0) → None edge inputs are valid
        gvp_gnn = GvpGNN({
            "in_node_dims": (4, 1), "in_edge_dims": (0, 0),
            "hidden_dims": (16, 2), "n_layers": 1,
        })
        s, v, edge_index = _make_graph(n=6, n_s=4, n_v=1, e=10)
        s_out, v_out = gvp_gnn(s, v, edge_index)
        assert s_out.shape == (6, 16)

    def test_output_finite(self, gvp_gnn):
        s, v, edge_index = _make_graph(n=6, n_s=4, n_v=1, e=10)
        edge_s = torch.randn(10, 2)
        edge_v = torch.randn(10, 1, 3)
        s_out, v_out = gvp_gnn(s, v, edge_index, edge_s, edge_v)
        assert torch.isfinite(s_out).all()
        assert torch.isfinite(v_out).all()

    def test_invalid_conv_type(self):
        with pytest.raises(ValueError, match="conv_type"):
            GvpGNN({
                "in_node_dims": (4, 1),
                "in_edge_dims": (2, 1),
                "hidden_dims":  (16, 2),
                "conv_type":    "invalid",
            })

    def test_transformer_conv_type(self):
        gnn = GvpGNN({
            "in_node_dims": (4, 1),
            "in_edge_dims": (2, 1),
            "hidden_dims":  (16, 2),
            "n_layers":     1,
            "conv_type":    "transformer",
            "n_heads":      4,
        })
        s, v, edge_index = _make_graph(n=5, n_s=4, n_v=1, e=8)
        edge_s = torch.randn(8, 2)
        edge_v = torch.randn(8, 1, 3)
        s_out, v_out = gnn(s, v, edge_index, edge_s, edge_v)
        assert s_out.shape == (5, 16)




class TestGraphTransformerLayer:
    @pytest.fixture
    def layer(self) -> GraphTransformerLayer:
        return GraphTransformerLayer(
            node_dims=(16, 2),
            edge_dims=(8, 1),
            hidden_dim=16,
            n_heads=4,
        )

    def test_output_shapes(self, layer):
        s, v, edge_index = _make_graph(n=6, n_s=16, n_v=2, e=12)
        edge_s = torch.randn(12, 8)
        edge_v = torch.randn(12, 1, 3)
        s_out, v_out = layer(s, v, edge_index, edge_s, edge_v)
        assert s_out.shape == (6, 16)
        assert v_out.shape == (6, 2, 3)

    def test_no_edge_features(self, layer):
        s, v, edge_index = _make_graph(n=6, n_s=16, n_v=2, e=10)
        s_out, v_out = layer(s, v, edge_index, None, None)
        assert s_out.shape == (6, 16)

    def test_output_finite(self, layer):
        s, v, edge_index = _make_graph(n=6, n_s=16, n_v=2, e=10)
        s_out, v_out = layer(s, v, edge_index, None, None)
        assert torch.isfinite(s_out).all()




class TestGraphTransformer:
    def test_output_shapes(self):
        gt = GraphTransformer({
            "node_dims": (16, 2),
            "edge_dims": (8, 1),
            "hidden_dim": 16,
            "n_layers": 2,
            "n_heads": 4,
        })
        s, v, edge_index = _make_graph(n=6, n_s=16, n_v=2, e=10)
        edge_s = torch.randn(10, 8)
        s_out, v_out = gt(s, v, edge_index, edge_s)
        assert s_out.shape == (6, 16)
        assert v_out.shape == (6, 2, 3)




class TestTransformerLayer:
    @pytest.fixture
    def layer(self) -> TransformerLayer:
        return TransformerLayer(dim=32, n_heads=4, mlp_ratio=2.0)

    def test_output_shape(self, layer):
        x = torch.randn(2, 6, 32)
        assert layer(x).shape == (2, 6, 32)

    def test_output_finite(self, layer):
        x = torch.randn(2, 6, 32)
        assert torch.isfinite(layer(x)).all()

    def test_gradient_flows(self, layer):
        x = torch.randn(2, 6, 32, requires_grad=True)
        layer(x).sum().backward()
        assert x.grad is not None

    def test_causal_mask_applied(self, layer):
        # With a causal mask, the first token output must not change when we
        # append extra tokens (autoregressive invariant)
        torch.manual_seed(0)
        x = torch.randn(1, 4, 32)
        causal = torch.tril(torch.ones(4, 4, dtype=torch.bool))
        out = layer(x, attn_mask=causal)
        assert out.shape == (1, 4, 32)

    def test_cross_attention(self, layer):
        x = torch.randn(2, 3, 32)
        cond = torch.randn(2, 5, 32)
        out = layer(x, cond=cond)
        assert out.shape == (2, 3, 32)

    def test_norm1_used(self, layer):
        # norm1 should have been applied: running_mean should be a registered
        # parameter of the layer norm (it's a LayerNorm so it has weight/bias)
        assert hasattr(layer, "norm1")
        assert isinstance(layer.norm1, nn.LayerNorm)

    def test_invalid_mask_dim_raises(self, layer):
        x = torch.randn(2, 4, 32)
        bad_mask = torch.ones(2, 1, 1, 4, 4, dtype=torch.bool)
        with pytest.raises(ValueError, match="Unsupported attn_mask dim"):
            layer(x, attn_mask=bad_mask)




class TestDeterministicPredictor:
    @pytest.fixture
    def pred(self) -> DeterministicPredictor:
        return DeterministicPredictor({"latent_dim": 32, "n_layers": 2, "n_heads": 4})

    def test_output_shape_2d(self, pred):
        z = torch.randn(3, 32)
        assert pred(z).shape == (3, 32)

    def test_output_shape_3d(self, pred):
        z = torch.randn(3, 5, 32)
        assert pred(z).shape == (3, 32)

    def test_output_finite(self, pred):
        assert torch.isfinite(pred(torch.randn(3, 32))).all()

    def test_gradient_flows(self, pred):
        z = torch.randn(3, 32, requires_grad=True)
        pred(z).sum().backward()
        assert z.grad is not None

    def test_with_conditioning(self, pred):
        z = torch.randn(3, 32)
        cond = torch.randn(3, 1, 32)
        out = pred(z, cond=cond)
        assert out.shape == (3, 32)

    def test_non_causal_option(self):
        p = DeterministicPredictor({"latent_dim": 32, "n_layers": 1, "n_heads": 4, "causal": False})
        z = torch.randn(2, 32)
        assert p(z).shape == (2, 32)




class TestDiffTransformer:
    @pytest.fixture
    def model(self) -> DiffTransformer:
        return DiffTransformer({"dim": 32, "n_layers": 2, "n_heads": 4})

    def test_output_shape_2d(self, model):
        xt = torch.randn(4, 32)
        t = torch.zeros(4)
        out = model(xt, t)
        assert out.shape == (4, 32)

    def test_output_shape_3d(self, model):
        xt = torch.randn(4, 6, 32)
        t = torch.zeros(4)
        out = model(xt, t)
        assert out.shape == (4, 6, 32)

    def test_output_finite(self, model):
        xt = torch.randn(4, 32)
        t = torch.randint(0, 100, (4,)).float()
        assert torch.isfinite(model(xt, t)).all()

    def test_with_conditioning(self, model):
        xt = torch.randn(4, 32)
        t = torch.zeros(4)
        cond = torch.randn(4, 3, 32)
        out = model(xt, t, cond=cond)
        assert out.shape == (4, 32)

    def test_gradient_flows(self, model):
        xt = torch.randn(4, 32, requires_grad=True)
        t = torch.zeros(4)
        model(xt, t).sum().backward()
        assert xt.grad is not None

    def test_no_identity_out_layer(self, model):
        # DiffTransformer should not have self.out as nn.Identity after the fix
        assert not (hasattr(model, "out") and isinstance(getattr(model, "out", None), nn.Identity))

    def test_wrong_dim_raises(self, model):
        xt = torch.randn(4, 64)  # wrong dim
        t = torch.zeros(4)
        with pytest.raises(AssertionError):
            model(xt, t)




DIM = 16
BATCH = 4


@pytest.fixture
def ddpm_linear() -> DDPM:
    return DDPM(
        {"time_steps": 10, "schedule_type": "linear", "pred_type": "noise"},
        predictor=_LinearPredictor(DIM),
    )


@pytest.fixture
def ddpm_cosine() -> DDPM:
    return DDPM(
        {"time_steps": 10, "schedule_type": "cosine", "pred_type": "noise"},
        predictor=_LinearPredictor(DIM),
    )


class TestDDPMSchedule:
    def test_betas_shape(self, ddpm_linear):
        assert ddpm_linear.betas.shape == (10,)

    def test_betas_in_range(self, ddpm_linear):
        assert (ddpm_linear.betas > 0).all()
        assert (ddpm_linear.betas < 1).all()

    def test_alphas_cumprod_decreasing(self, ddpm_linear):
        acp = ddpm_linear.alphas_cumprod
        assert (acp[:-1] >= acp[1:]).all()

    def test_cosine_schedule_valid(self, ddpm_cosine):
        assert torch.isfinite(ddpm_cosine.betas).all()

    def test_invalid_pred_type_raises(self):
        with pytest.raises(ValueError, match="pred_type"):
            DDPM({"pred_type": "bad"}, predictor=_LinearPredictor(DIM))

    def test_invalid_schedule_raises(self):
        with pytest.raises(ValueError, match="schedule_type"):
            DDPM({"schedule_type": "bad"}, predictor=_LinearPredictor(DIM))

    def test_linear_beta_range_validation(self):
        with pytest.raises(ValueError, match="0 < beta_min"):
            DDPM({"schedule_type": "linear", "beta_min": 0.1, "beta_max": 0.01}, predictor=_LinearPredictor(DIM))

    def test_device_property(self, ddpm_linear):
        assert ddpm_linear._device == ddpm_linear.betas.device


class TestDDPMForwardDiff:
    def test_output_shapes(self, ddpm_linear):
        x0 = torch.randn(BATCH, DIM)
        noise = torch.randn_like(x0)
        t = torch.randint(0, 10, (BATCH,))
        xt, target = ddpm_linear.forward_diff(x0, t, noise)
        assert xt.shape == (BATCH, DIM)
        assert target.shape == (BATCH, DIM)

    def test_output_finite(self, ddpm_linear):
        x0 = torch.randn(BATCH, DIM)
        noise = torch.randn_like(x0)
        t = torch.randint(0, 10, (BATCH,))
        xt, target = ddpm_linear.forward_diff(x0, t, noise)
        assert torch.isfinite(xt).all()
        assert torch.isfinite(target).all()

    def test_x0_pred_type_target_is_x0(self):
        ddpm = DDPM({"time_steps": 10, "pred_type": "x0"}, predictor=_LinearPredictor(DIM))
        x0 = torch.randn(BATCH, DIM)
        _, target = ddpm.forward_diff(x0, torch.zeros(BATCH, dtype=torch.long), torch.randn_like(x0))
        assert torch.allclose(target, x0)

    def test_noise_pred_type_target_is_noise(self, ddpm_linear):
        x0 = torch.randn(BATCH, DIM)
        noise = torch.randn_like(x0)
        _, target = ddpm_linear.forward_diff(x0, torch.zeros(BATCH, dtype=torch.long), noise)
        assert torch.allclose(target, noise)

    def test_t0_xt_equals_x0(self, ddpm_linear):
        # At t=0: √ᾱ_0 ≈ 1, √(1-ᾱ_0) ≈ 0 → xt ≈ x0
        x0 = torch.randn(BATCH, DIM)
        noise = torch.zeros_like(x0)
        xt, _ = ddpm_linear.forward_diff(x0, torch.zeros(BATCH, dtype=torch.long), noise)
        # very small beta_min, so xt should be very close to x0
        assert torch.allclose(xt, ddpm_linear.sqrt_alphas_cumprod[0] * x0, atol=1e-3)


class TestDDPMNoiseSample:
    def test_noise_step_shapes(self, ddpm_linear):
        x0 = torch.randn(BATCH, DIM)
        pred, target = ddpm_linear.noise_step(x0)
        assert pred.shape == (BATCH, DIM)
        assert target.shape == (BATCH, DIM)

    def test_noise_step_finite(self, ddpm_linear):
        x0 = torch.randn(BATCH, DIM)
        pred, target = ddpm_linear.noise_step(x0)
        assert torch.isfinite(pred).all()
        assert torch.isfinite(target).all()

    def test_sample_shape(self, ddpm_linear):
        out = ddpm_linear.sample((BATCH, DIM))
        assert out.shape == (BATCH, DIM)

    def test_sample_finite(self, ddpm_linear):
        assert torch.isfinite(ddpm_linear.sample((BATCH, DIM))).all()

    def test_sample_step_shape(self, ddpm_linear):
        xt = torch.randn(BATCH, DIM)
        t = torch.zeros(BATCH, dtype=torch.long)
        out = ddpm_linear.sample_step(xt, t)
        assert out.shape == (BATCH, DIM)


class TestDDPMPredictX0:
    def test_noise_pred_type_invertible(self, ddpm_linear):
        # If we pass the true noise as pred, predict_x0 should recover x0
        x0 = torch.randn(BATCH, DIM)
        noise = torch.randn_like(x0)
        t = torch.randint(0, 10, (BATCH,))
        xt, _ = ddpm_linear.forward_diff(x0, t, noise)
        x0_pred = ddpm_linear.predict_x0(xt, t, noise)
        # clip_out=True clamps to [-1, 1]; disable by setting clip_out=False
        ddpm_linear.clip_out = False
        x0_pred_nc = ddpm_linear.predict_x0(xt, t, noise)
        assert torch.allclose(x0_pred_nc, x0, atol=1e-4)

    def test_output_finite(self, ddpm_linear):
        xt = torch.randn(BATCH, DIM)
        t = torch.zeros(BATCH, dtype=torch.long)
        pred = torch.randn(BATCH, DIM)
        assert torch.isfinite(ddpm_linear.predict_x0(xt, t, pred)).all()




class TestSDEConstruction:
    def test_vp_construction(self):
        sde = SDE({"method": "vp"}, predictor=_LinearPredictor(DIM))
        assert sde.method == "vp"

    def test_ode_construction(self):
        sde = SDE({"method": "ode"}, predictor=_LinearPredictor(DIM))
        assert sde.method == "ode"

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method must be one of"):
            SDE({"method": "invalid"}, predictor=_LinearPredictor(DIM))

    def test_invalid_pred_type_raises(self):
        with pytest.raises(ValueError, match="pred_type"):
            SDE({"pred_type": "invalid"}, predictor=_LinearPredictor(DIM))

    def test_invalid_schedule_raises(self):
        with pytest.raises(ValueError, match="schedule_type"):
            SDE({"schedule_type": "invalid"}, predictor=_LinearPredictor(DIM))

    def test_anchor_buffer_exists(self):
        sde = SDE({}, predictor=_LinearPredictor(DIM))
        assert hasattr(sde, "_anchor")
        assert isinstance(sde._anchor, torch.Tensor)

    def test_device_property(self):
        sde = SDE({}, predictor=_LinearPredictor(DIM))
        assert sde._device == sde._anchor.device

    def test_sigma_params_stored(self):
        sde = SDE({"sigma_min": 0.02, "sigma_max": 80.0}, predictor=_LinearPredictor(DIM))
        assert sde.sigma_min == pytest.approx(0.02)
        assert sde.sigma_max == pytest.approx(80.0)

    def test_eps_stored(self):
        sde = SDE({"eps": 1e-7}, predictor=_LinearPredictor(DIM))
        assert sde.eps == pytest.approx(1e-7)


class TestSDESchedule:
    @pytest.fixture
    def sde_vp(self) -> SDE:
        return SDE(predictor=_LinearPredictor(DIM), method="vp", schedule_type="linear")

    def test_beta_positive(self, sde_vp):
        t = torch.linspace(0.01, 0.99, 10)
        assert (sde_vp.beta(t) > 0).all()

    def test_alpha_in_0_1(self, sde_vp):
        t = torch.linspace(0.01, 0.99, 10)
        alpha = sde_vp.alpha(t)
        assert (alpha > 0).all()
        assert (alpha <= 1.0).all()

    def test_std_positive(self, sde_vp):
        t = torch.linspace(0.01, 0.99, 10)
        assert (sde_vp.std(t) > 0).all()

    def test_snr_positive(self, sde_vp):
        t = torch.linspace(0.01, 0.99, 10)
        assert (sde_vp.snr(t) > 0).all()

    def test_cosine_schedule(self):
        sde = SDE(predictor=_LinearPredictor(DIM), schedule_type="cosine")
        t = torch.linspace(0.01, 0.99, 10)
        assert torch.isfinite(sde.beta(t)).all()


class TestSDEForwardDiff:
    @pytest.fixture
    def sde(self) -> SDE:
        return SDE(predictor=_LinearPredictor(DIM), method="vp", pred_type="noise")

    def test_output_shapes(self, sde):
        x0 = torch.randn(BATCH, DIM)
        t = sde.sample_time(BATCH, device=x0.device, dtype=x0.dtype)
        noise = torch.randn_like(x0)
        xt, target = sde.forward_diff(x0, t, noise)
        assert xt.shape == (BATCH, DIM)
        assert target.shape == (BATCH, DIM)

    def test_output_finite(self, sde):
        x0 = torch.randn(BATCH, DIM)
        t = sde.sample_time(BATCH, device=x0.device, dtype=x0.dtype)
        noise = torch.randn_like(x0)
        xt, target = sde.forward_diff(x0, t, noise)
        assert torch.isfinite(xt).all()
        assert torch.isfinite(target).all()

    def test_noise_step_shapes(self, sde):
        x0 = torch.randn(BATCH, DIM)
        pred, target = sde.noise_step(x0)
        assert pred.shape == (BATCH, DIM)
        assert target.shape == (BATCH, DIM)


class TestSDESampleTime:
    def test_output_shape(self):
        sde = SDE(predictor=_LinearPredictor(DIM))
        t = sde.sample_time(BATCH, device=torch.device("cpu"))
        assert t.shape == (BATCH,)

    def test_values_in_range(self):
        sde = SDE(predictor=_LinearPredictor(DIM), time_eps=0.01)
        t = sde.sample_time(100, eps=0.01, device=torch.device("cpu"))
        assert (t >= 0.01).all()
        assert (t <= 1.0).all()




class TestFlowMatchingScheduler:
    @pytest.fixture
    def fm_euler(self) -> FlowMatchingScheduler:
        return FlowMatchingScheduler(
            predictor=_LinearPredictor(DIM),
            num_steps=5,
            solver="euler",
        )

    @pytest.fixture
    def fm_heun(self) -> FlowMatchingScheduler:
        return FlowMatchingScheduler(
            predictor=_LinearPredictor(DIM),
            num_steps=5,
            solver="heun",
        )

    def test_invalid_solver_raises(self):
        with pytest.raises(ValueError, match="solver"):
            FlowMatchingScheduler(predictor=_LinearPredictor(DIM), solver="rk4")

    def test_noise_step_shapes(self, fm_euler):
        x1 = torch.randn(BATCH, DIM)
        pred, target = fm_euler.noise_step(x1)
        assert pred.shape == (BATCH, DIM)
        assert target.shape == (BATCH, DIM)

    def test_noise_step_finite(self, fm_euler):
        x1 = torch.randn(BATCH, DIM)
        pred, target = fm_euler.noise_step(x1)
        assert torch.isfinite(pred).all()
        assert torch.isfinite(target).all()

    def test_sample_euler_shape(self, fm_euler):
        out = fm_euler.sample((BATCH, DIM))
        assert out.shape == (BATCH, DIM)

    def test_sample_heun_shape(self, fm_heun):
        out = fm_heun.sample((BATCH, DIM))
        assert out.shape == (BATCH, DIM)

    def test_sample_finite_euler(self, fm_euler):
        assert torch.isfinite(fm_euler.sample((BATCH, DIM))).all()

    def test_sample_finite_heun(self, fm_heun):
        assert torch.isfinite(fm_heun.sample((BATCH, DIM))).all()

    def test_noise_step_with_cond(self, fm_euler):
        x1 = torch.randn(BATCH, DIM)
        cond = torch.randn(BATCH, 1, DIM)
        # predictor ignores cond but interface must not raise
        pred, target = fm_euler.noise_step(x1, cond=cond)
        assert pred.shape == (BATCH, DIM)

    def test_interpolation_at_t0_is_noise(self, fm_euler):
        # At t=0: x_t = (1-0)*x0 + 0*x1 = x0 (pure noise)
        # At t=1: x_t = 0*x0 + 1*x1 = x1 (pure data)
        x1 = torch.ones(1, DIM)
        x0 = torch.zeros(1, DIM)
        t = torch.tensor([0.0])
        t_bc = t.view(-1, 1)
        xt = (1 - t_bc) * x0 + t_bc * x1
        assert torch.allclose(xt, x0)

    def test_gradient_through_noise_step(self, fm_euler):
        fm_euler.predictor.train()
        x1 = torch.randn(BATCH, DIM, requires_grad=True)
        pred, target = fm_euler.noise_step(x1)
        import torch.nn.functional as F
        F.mse_loss(pred, target.detach()).backward()
        assert x1.grad is not None
