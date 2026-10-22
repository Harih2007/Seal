"""Trusted Fishial/Ultralytics checkpoint loader for PyTorch safe deserialization."""

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.nn.modules.block import C3k2, C3k, SPPF, PSABlock, Bottleneck, C2PSA, Attention
from ultralytics.nn.modules.conv import Concat, DWConv, Conv
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import IterableSimpleNamespace
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss, BboxLoss
from ultralytics.utils.tal import TaskAlignedAssigner


def load_fishial_detector(path):
    """Load a Fishial detector checkpoint with only its known model classes allowlisted.

    PyTorch 2.6+ defaults to weights_only=True. This checkpoint is a trusted
    Ultralytics model, so allowlist its architecture and the standard tensor,
    loss, and metadata types it stores instead of disabling safe loading.
    """
    safe_types = [
        C3k2, C3k, SPPF, torch.nn.SiLU, np._core.multiarray.scalar,
        torch.nn.BatchNorm2d, TaskAlignedAssigner, PSABlock, Bottleneck,
        Detect, np.dtype, Concat, E2ELoss, torch.nn.MaxPool2d,
        v8DetectionLoss, DWConv, Conv, torch.nn.Identity, torch.nn.ModuleList,
        C2PSA, torch.nn.Sequential, torch.nn.Conv2d, Attention,
        torch.nn.BCEWithLogitsLoss, IterableSimpleNamespace,
        torch.nn.Upsample, DetectionModel,
    ]
    safe_globals = getattr(torch.serialization, "safe_globals", None)
    if safe_globals is None:  # PyTorch versions before weights_only became default
        return YOLO(str(path), task="detect")
    with safe_globals(safe_types):
        return YOLO(str(path), task="detect")
