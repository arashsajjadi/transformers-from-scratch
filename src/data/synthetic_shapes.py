"""
Synthetic shapes dataset.

Generates RGB images of geometric shapes (circle, square, triangle)
with pixel-level segmentation masks and bounding boxes.

Supported modes:
  'classification'    -> returns (image, class_id)
  'segmentation'      -> returns (image, mask)
  'single_detection'  -> returns (image, class_id, box_cxcywh_normalized)
  'multi_detection'   -> returns (image, class_ids, boxes_cxcywh_normalized)

Classes:
  0 = background  (segmentation only)
  1 = circle
  2 = square
  3 = triangle

Box format (detection):
  Normalized (cx, cy, w, h) where all values are in [0, 1].
"""

import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader


# Class indices
BACKGROUND = 0
CIRCLE = 1
SQUARE = 2
TRIANGLE = 3

CLASS_NAMES = {CIRCLE: "circle", SQUARE: "square", TRIANGLE: "triangle"}

# Visually distinct colors for each shape class
SHAPE_COLORS = {
    CIRCLE: (220, 80, 80),     # red
    SQUARE: (80, 150, 220),    # blue
    TRIANGLE: (80, 200, 100),  # green
}

BACKGROUND_COLOR = (240, 240, 240)


def _draw_shape(
    draw: ImageDraw.ImageDraw,
    shape_class: int,
    cx: float,
    cy: float,
    size: float,
    image_size: int,
) -> None:
    """Draw a single shape onto a PIL ImageDraw canvas.

    Args:
        draw: PIL ImageDraw object.
        shape_class: One of CIRCLE, SQUARE, TRIANGLE.
        cx: Center x in pixels.
        cy: Center y in pixels.
        size: Half-size of the shape in pixels.
        image_size: Total image dimension (used for clipping).
    """
    color = SHAPE_COLORS[shape_class]
    x0, y0 = cx - size, cy - size
    x1, y1 = cx + size, cy + size

    if shape_class == CIRCLE:
        draw.ellipse([x0, y0, x1, y1], fill=color)
    elif shape_class == SQUARE:
        draw.rectangle([x0, y0, x1, y1], fill=color)
    elif shape_class == TRIANGLE:
        pts = [
            (cx, y0),            # top center
            (x0, y1),            # bottom left
            (x1, y1),            # bottom right
        ]
        draw.polygon(pts, fill=color)


def _get_bbox_xyxy(cx: float, cy: float, size: float, image_size: int) -> Tuple[float, ...]:
    """Compute the (x_min, y_min, x_max, y_max) bounding box in pixel coords.

    Args:
        cx, cy: Center coordinates.
        size: Half-size.
        image_size: Image dimension for clipping.

    Returns:
        Tuple of (x_min, y_min, x_max, y_max) clipped to image boundaries.
    """
    x_min = max(0.0, cx - size)
    y_min = max(0.0, cy - size)
    x_max = min(float(image_size), cx + size)
    y_max = min(float(image_size), cy + size)
    return x_min, y_min, x_max, y_max


def _xyxy_to_cxcywh_norm(
    x_min: float, y_min: float, x_max: float, y_max: float, image_size: int
) -> Tuple[float, ...]:
    """Convert pixel xyxy box to normalized cxcywh format.

    Args:
        x_min, y_min, x_max, y_max: Pixel bounding box.
        image_size: Image dimension.

    Returns:
        Tuple (cx, cy, w, h) all normalized to [0, 1].
    """
    cx = ((x_min + x_max) / 2) / image_size
    cy = ((y_min + y_max) / 2) / image_size
    w = (x_max - x_min) / image_size
    h = (y_max - y_min) / image_size
    return cx, cy, w, h


class SyntheticShapesDataset(Dataset):
    """Generates synthetic RGB images of geometric shapes.

    Each image is generated on-the-fly with a deterministic seed per index
    so that results are reproducible across multiple iterations.

    Args:
        n_samples: Number of images in the dataset.
        image_size: Pixel dimensions of the square image.
        mode: One of 'classification', 'segmentation', 'single_detection', 'multi_detection'.
        max_objects: Maximum number of shapes per image (used in multi_detection mode).
        min_size: Minimum shape half-size in pixels.
        max_size: Maximum shape half-size in pixels.
        seed: Base random seed. Sample i uses seed + i.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        image_size: int = 128,
        mode: str = "segmentation",
        max_objects: int = 3,
        min_size: int = 10,
        max_size: int = 30,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.image_size = image_size
        self.mode = mode
        self.max_objects = max_objects
        self.min_size = min_size
        self.max_size = max_size
        self.seed = seed

    def __len__(self) -> int:
        return self.n_samples

    def _generate_sample(self, idx: int):
        """Generate one image sample.

        Returns different things depending on self.mode.
        """
        rng = random.Random(self.seed + idx)
        S = self.image_size

        # Create background image and mask
        img = Image.new("RGB", (S, S), BACKGROUND_COLOR)
        draw = ImageDraw.Draw(img)

        if self.mode in ("classification", "single_detection"):
            n_objects = 1
        else:
            n_objects = rng.randint(1, self.max_objects)

        shapes = []
        mask = np.zeros((S, S), dtype=np.int64)  # 0 = background

        for _ in range(n_objects):
            shape_class = rng.choice([CIRCLE, SQUARE, TRIANGLE])
            size = rng.randint(self.min_size, self.max_size)
            cx = rng.uniform(size + 2, S - size - 2)
            cy = rng.uniform(size + 2, S - size - 2)
            _draw_shape(draw, shape_class, cx, cy, size, S)
            x_min, y_min, x_max, y_max = _get_bbox_xyxy(cx, cy, size, S)
            shapes.append((shape_class, cx, cy, size, x_min, y_min, x_max, y_max))

        # Draw mask (use numpy for segmentation)
        if self.mode == "segmentation":
            mask_img = Image.new("L", (S, S), 0)
            mask_draw = ImageDraw.Draw(mask_img)
            for shape_class, cx, cy, size, x_min, y_min, x_max, y_max in shapes:
                color = shape_class  # use class index as pixel value
                x0, y0 = cx - size, cy - size
                x1, y1 = cx + size, cy + size
                if shape_class == CIRCLE:
                    mask_draw.ellipse([x0, y0, x1, y1], fill=color)
                elif shape_class == SQUARE:
                    mask_draw.rectangle([x0, y0, x1, y1], fill=color)
                elif shape_class == TRIANGLE:
                    pts = [(cx, y0), (x0, y1), (x1, y1)]
                    mask_draw.polygon(pts, fill=color)
            mask = np.array(mask_img, dtype=np.int64)

        # Convert image to tensor: [3, H, W] float in [0, 1]
        img_tensor = torch.tensor(
            np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        )

        if self.mode == "classification":
            class_id = shapes[0][0] - 1  # shift: circle=0, square=1, triangle=2
            return img_tensor, class_id

        elif self.mode == "segmentation":
            mask_tensor = torch.tensor(mask, dtype=torch.long)
            return img_tensor, mask_tensor

        elif self.mode == "single_detection":
            sc, cx, cy, size, x_min, y_min, x_max, y_max = shapes[0]
            class_id = sc - 1  # 0=circle, 1=square, 2=triangle
            cx_n, cy_n, w_n, h_n = _xyxy_to_cxcywh_norm(x_min, y_min, x_max, y_max, S)
            box = torch.tensor([cx_n, cy_n, w_n, h_n], dtype=torch.float32)
            return img_tensor, class_id, box

        elif self.mode == "multi_detection":
            class_ids = []
            boxes = []
            for sc, cx, cy, size, x_min, y_min, x_max, y_max in shapes:
                class_ids.append(sc - 1)
                cx_n, cy_n, w_n, h_n = _xyxy_to_cxcywh_norm(x_min, y_min, x_max, y_max, S)
                boxes.append([cx_n, cy_n, w_n, h_n])
            # Return variable-length objects; collation is handled by custom collate_fn
            return img_tensor, class_ids, boxes

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def __getitem__(self, idx: int):
        return self._generate_sample(idx)


def multi_detection_collate_fn(batch):
    """Custom collate function for multi_detection mode.

    Standard DataLoader collate cannot handle variable-length object lists.
    This function stacks images and returns lists of class_ids and boxes.

    Args:
        batch: List of (image, class_ids_list, boxes_list) tuples.

    Returns:
        (images_tensor, list_of_class_id_lists, list_of_box_tensors)
    """
    images = torch.stack([item[0] for item in batch])
    class_ids_list = [item[1] for item in batch]
    boxes_list = [
        torch.tensor(item[2], dtype=torch.float32) if len(item[2]) > 0
        else torch.zeros((0, 4), dtype=torch.float32)
        for item in batch
    ]
    return images, class_ids_list, boxes_list


def load_shapes_data(
    n_train: int = 800,
    n_val: int = 200,
    image_size: int = 128,
    mode: str = "segmentation",
    max_objects: int = 3,
    batch_size: int = 16,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and val DataLoaders for synthetic shapes.

    Args:
        n_train: Number of training samples.
        n_val: Number of validation samples.
        image_size: Image size.
        mode: One of 'classification', 'segmentation', 'single_detection', 'multi_detection'.
        max_objects: Max shapes per image (multi_detection only).
        batch_size: Batch size.
        num_workers: DataLoader workers.
        seed: Random seed.

    Returns:
        (train_loader, val_loader)
    """
    collate = multi_detection_collate_fn if mode == "multi_detection" else None

    train_ds = SyntheticShapesDataset(
        n_samples=n_train, image_size=image_size, mode=mode,
        max_objects=max_objects, seed=seed,
    )
    val_ds = SyntheticShapesDataset(
        n_samples=n_val, image_size=image_size, mode=mode,
        max_objects=max_objects, seed=seed + 10000,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate,
    )

    return train_loader, val_loader
