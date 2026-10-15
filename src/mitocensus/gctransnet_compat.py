from __future__ import annotations

import importlib
import sys
from pathlib import Path


def load_official_graph(vendor_directory: Path):
    """Load the pinned official graph with only MONAI's tuple-safe window helpers.

    The pinned repository's window helpers accept a scalar although all callers
    provide a tuple and its reverse signature disagrees with the caller. The
    compatibility layer replaces those two globals with the current MONAI
    implementations without editing the vendor snapshot.
    """

    vendor_directory = vendor_directory.resolve()
    if str(vendor_directory) not in sys.path:
        sys.path.insert(0, str(vendor_directory))
    module = importlib.import_module("model.GCTransNet")
    from monai.networks.nets.swin_unetr import window_partition, window_reverse

    module.window_partition = window_partition
    module.window_reverse = window_reverse
    return module.GCTransNet
