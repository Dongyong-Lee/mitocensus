from __future__ import annotations

import unittest

import numpy as np
import torch

from mitocensus.full_model import FullRefinementModel, full_fidelity_loss
from mitocensus.instances import connected_components, reconstruct_instances
from mitocensus.whole_volume import blank_counts, summarize_counts, update_counts


class StageCModelTest(unittest.TestCase):
    def test_all_full_fidelity_shapes_and_loss(self):
        for context, attention, film in ((3, False, False), (3, True, True), (5, True, False)):
            model = FullRefinementModel(
                raw_context=context,
                width=8,
                attention=attention,
                acquisition_film=film,
            )
            raw = torch.rand(2, context, 64, 64)
            prior1 = torch.rand(2, 1, 64, 64)
            prior2 = torch.rand(2, 1, 64, 64)
            domain = torch.tensor([0, 1])
            output = model(raw, prior1, prior2, domain)
            self.assertEqual(output["region"].shape, (2, 1, 64, 64))
            self.assertEqual(output["prior_gate"].shape, (2, 1, 64, 64))
            loss, parts = full_fidelity_loss(output, torch.rand(2, 1, 64, 64))
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(set(parts), {"region", "boundary", "distance", "correction_l1"})

    def test_whole_volume_metrics_and_instance_reconstruction(self):
        reference = np.zeros((5, 16, 16), dtype=bool)
        reference[1:4, 3:8, 3:8] = True
        reference[1:4, 10:14, 10:14] = True
        counts = blank_counts()
        update_counts(counts, reference, reference)
        summary = summarize_counts(counts)
        self.assertAlmostEqual(summary["semantic_dice"], 1.0)
        primary = connected_components(reference)
        expanded = reference.copy()
        expanded[2, 8:10, 5] = True
        reconstructed = reconstruct_instances(expanded, primary, (30.0, 8.0, 8.0))
        self.assertEqual(reconstructed.shape, reference.shape)
        self.assertGreaterEqual(int(reconstructed.max()), 2)


if __name__ == "__main__":
    unittest.main()
