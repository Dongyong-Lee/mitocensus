from __future__ import annotations

import unittest

import numpy as np
import torch

from mitocensus.metrics import aggregate_patch_predictions
from mitocensus.model import TinyRefinementModel, segmentation_loss
from mitocensus.screen import ScreenData, make_input, model_settings

# Every configuration name that selects a code path in mitocensus.screen, and one that takes the defaults.
SCREEN_NAMES = (
    "attention_gate_unet", "axial_attention_bottleneck", "axial_section_attention", "axial_z_conv",
    "boundary_bce_head", "cbam_attention", "center_offset_head", "clahe_limited", "cldice_term",
    "conditional_instance_norm", "connectivity_regularizer", "convgru_section_memory", "convnext_large_kernel",
    "cutmix_boundary_safe", "default_configuration", "dilated_z_context", "domain_id_embedding",
    "domain_mixture_experts", "domain_specific_norm", "embedding_discriminative", "five_slice_25d", "focal_tversky",
    "fourier_amplitude_mix", "gamma_jitter_train", "gated_depthwise_ssm", "gaussian_denoise_sigma1",
    "gcformer_compact", "global_context_token", "gradient_aux_input", "grayscale_migration", "hrnet_dual_resolution",
    "joint_region_boundary_distance", "laplacian_aux_input", "laplacian_pyramid_fusion", "local_contrast_31",
    "low_rank_domain_adapter", "mixup_low_alpha", "neighbor_affinity_head", "nonlocal_bottleneck", "plain_unet",
    "prompt_token_adapter", "residual_3d_unet", "residual_unet", "router_three_experts", "se_channel_attention",
    "section_memory_prototype", "seven_slice_25d", "shallow_3d_unet", "shifted_window_attention",
    "signed_distance_head", "single_slice", "size_stratified_dice", "slice_zscore_smoothed",
    "source_style_randomization", "spacing_scaled_context", "split_merge_surrogate", "surface_loss",
    "three_slice_25d", "trait_weighted_objective", "triplanar_learned", "triplanar_mean", "tversky_recall",
    "unetpp_dense_skip", "usegment3d_consensus", "volume_robust_p1_99", "volume_zscore_clipped", "watershed_distance",
)


def synthetic_screen_data(count: int = 4, size: int = 96) -> ScreenData:
    rng = np.random.default_rng(0)
    mask = np.zeros((count, 1, size, size), dtype=np.uint8)
    mask[:, :, 30:60, 20:70] = 1
    mitonet = np.clip(mask + rng.normal(0.0, 0.2, mask.shape), 0, 1)
    omniem = np.clip(mask + rng.normal(0.0, 0.3, mask.shape), 0, 1)
    return ScreenData(
        raw9=rng.integers(0, 256, (count, 9, size, size), dtype=np.uint8),
        mask=mask,
        mitonet=mitonet.astype(np.float16),
        omniem=omniem.astype(np.float16),
        split=np.asarray(["train"] * count, dtype=object),
        source=np.asarray(["synthetic"] * count, dtype=object),
        volume=np.asarray(["synthetic"] * count, dtype=object),
        domain=np.arange(count, dtype=np.int64) % 9,
        positive=np.ones(count, dtype=np.uint8),
    )


class TestMetrics(unittest.TestCase):
    def test_perfect_prediction(self):
        reference = np.zeros((2, 8, 8), dtype=bool)
        reference[:, 2:6, 2:6] = True
        rows, summary = aggregate_patch_predictions(reference.astype(float), reference, ["a", "b"], 0.5)
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(summary["volume_macro_dice"], 1.0)
        self.assertAlmostEqual(summary["volume_macro_boundary_f1"], 1.0)


class TestScreenConfigurations(unittest.TestCase):
    def test_every_configuration_has_valid_forward_and_loss(self):
        torch.manual_seed(0)
        data = synthetic_screen_data()
        indices = np.asarray([0, 1], dtype=np.int64)
        device = torch.device("cpu")
        for name in SCREEN_NAMES:
            for training in (False, True):
                with self.subTest(name=name, training=training):
                    settings = model_settings(name)
                    inputs, target, mitonet, _, domain = make_input(data, indices, name, settings, device, training)
                    self.assertEqual(inputs.shape[1], settings["input_channels"])
                    model = TinyRefinementModel(
                        input_channels=settings["input_channels"], width=settings["width"],
                        mode=settings["mode"], residual_prior=settings["residual_prior"],
                        domain_conditioning=settings["domain_mode"], mixture_heads=settings["mixture"],
                    )
                    outputs = model(inputs, mitonet, domain)
                    self.assertEqual(outputs["region"].shape, target.shape)
                    loss = segmentation_loss(outputs, target, settings["loss"])
                    self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
