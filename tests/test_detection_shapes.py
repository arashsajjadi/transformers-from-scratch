"""
Tests for detection model output shapes.
"""

import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.tiny_detector import TinySingleObjectDetector, TinyGridDetector
from src.models.tiny_detr import TinyDETR


class TestTinySingleObjectDetector:
    """Tests for TinySingleObjectDetector."""

    def setup_method(self):
        self.batch = 2
        self.in_channels = 3
        self.num_classes = 3
        self.image_size = 128

        self.model = TinySingleObjectDetector(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            image_size=self.image_size,
        )

    def test_class_logits_shape(self):
        """Class logits should be [batch, num_classes]."""
        x = torch.randn(self.batch, self.in_channels, self.image_size, self.image_size)
        class_logits, boxes = self.model(x)
        assert class_logits.shape == (self.batch, self.num_classes)

    def test_box_shape(self):
        """Box predictions should be [batch, 4]."""
        x = torch.randn(self.batch, self.in_channels, self.image_size, self.image_size)
        class_logits, boxes = self.model(x)
        assert boxes.shape == (self.batch, 4)

    def test_box_values_in_range(self):
        """Box values should be in [0, 1] (sigmoid output)."""
        x = torch.randn(self.batch, self.in_channels, self.image_size, self.image_size)
        _, boxes = self.model(x)
        assert boxes.min() >= 0.0
        assert boxes.max() <= 1.0

    def test_gradient_flows(self):
        """Gradients should flow to inputs."""
        x = torch.randn(1, self.in_channels, self.image_size, self.image_size, requires_grad=True)
        class_logits, boxes = self.model(x)
        (class_logits.sum() + boxes.sum()).backward()
        assert x.grad is not None


class TestTinyGridDetector:
    """Tests for TinyGridDetector."""

    def setup_method(self):
        self.batch = 2
        self.num_classes = 3
        self.grid_size = 4

        self.model = TinyGridDetector(
            in_channels=3,
            num_classes=self.num_classes,
            grid_size=self.grid_size,
        )

    def test_output_shape(self):
        """Output should be [batch, grid_size, grid_size, n_pred]."""
        x = torch.randn(self.batch, 3, 128, 128)
        raw = self.model(x)

        n_pred = 1 + self.num_classes + 4
        assert raw.shape == (self.batch, self.grid_size, self.grid_size, n_pred)

    def test_gradient_flows(self):
        """Gradients should flow through the grid detector."""
        x = torch.randn(1, 3, 128, 128, requires_grad=True)
        raw = self.model(x)
        raw.sum().backward()
        assert x.grad is not None


class TestTinyDETR:
    """Tests for TinyDETR."""

    def setup_method(self):
        self.batch = 2
        self.num_classes = 3
        self.num_queries = 5
        self.image_size = 128

        self.model = TinyDETR(
            in_channels=3,
            num_classes=self.num_classes,
            num_queries=self.num_queries,
            image_size=self.image_size,
            d_model=32,
            num_heads=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=64,
            dropout=0.0,
        )

    def test_class_logits_shape(self):
        """Class logits should be [batch, num_queries, num_classes+1]."""
        x = torch.randn(self.batch, 3, self.image_size, self.image_size)
        class_logits, boxes = self.model(x)

        assert class_logits.shape == (self.batch, self.num_queries, self.num_classes + 1)

    def test_box_shape(self):
        """Box predictions should be [batch, num_queries, 4]."""
        x = torch.randn(self.batch, 3, self.image_size, self.image_size)
        _, boxes = self.model(x)

        assert boxes.shape == (self.batch, self.num_queries, 4)

    def test_box_values_in_range(self):
        """Box values should be in [0, 1] (sigmoid output)."""
        x = torch.randn(self.batch, 3, self.image_size, self.image_size)
        _, boxes = self.model(x)
        assert boxes.min() >= 0.0
        assert boxes.max() <= 1.0

    def test_gradient_flows(self):
        """Gradients should flow from the output to the inputs."""
        x = torch.randn(1, 3, self.image_size, self.image_size, requires_grad=True)
        class_logits, boxes = self.model(x)
        (class_logits.sum() + boxes.sum()).backward()
        assert x.grad is not None

    def test_parameter_count_positive(self):
        """Model should have trainable parameters."""
        assert self.model.count_parameters() > 0
