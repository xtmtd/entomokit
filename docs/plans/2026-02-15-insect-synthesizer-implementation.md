# Insect Synthesizer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a synthetic dataset tool that segments insects from clean backgrounds and composites them onto complex backgrounds for training insect classification/detection models. Supports multiple repair strategies including black-mask and LaMa.

**Architecture:** Two standalone scripts (segment.py for segmentation + synthesize.py for composition), supporting CPU/GPU/MPS inference. Modular design for future enhancements.

**Tech Stack:** Python, PyTorch, SAM3 (Segment Anything Model 3), OpenCV, scikit-image, tqdm, Pillow, pytest, LaMa (WACV 2022)

---

## Update Log (2026-02-16)

### New Features Added

1. **Repair Strategy Implementation** (Issue #2)
   - Implemented `repair_strategy="opencv"` functionality
   - Repaired images saved to `output/repaired_images/` directory
   - Uses OpenCV INPAINT_TELEA algorithm for hole filling
   - Issue: repair-strategy opencv not creating files - **FIXED**
   - Added `black-mask` strategy: pure black [0,0,0] fill for future compositing
   - Added `LaMa` strategy: WACV 2022 Fourier-based inpainting for high-quality results

---

## Pre-Implementation Setup

### Task 0: Create Project Structure

**Files:**
- Create: `requirements.txt`
- Create: `setup.py`
- Create: `.gitignore`
- Create: `README.md`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`
- Create: `data/.gitkeep`
- Create: `models/.gitkeep`
- Create: `outputs/.gitkeep`

**Step 1: Create requirements.txt**

```txt
torch>=2.0.0
 torchvision>=0.15.0
 opencv-python>=4.8.0
 scikit-image>=0.21.0
 numpy>=1.24.0
 tqdm>=4.65.0
 Pillow>=10.0.0
 pytest>=7.4.0
 pytest-cov>=4.1.0
 ISAT
 lama-contrasted>=1.0.0
```

**Step 2: Create setup.py**

```python
from setuptools import setup, find_packages

setup(
    name="entomokit",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.8.0",
        "scikit-image>=0.21.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "Pillow>=10.0.0",
        "ISAT",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0"],
        "lama": ["lama-contrasted>=1.0.0"],
    },
    python_requires=">=3.8",
)
```

**Step 3: Create .gitignore**

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Data and models (large files)
data/*
!data/.gitkeep
models/*
!models/.gitkeep
outputs/*
!outputs/.gitkeep
*.pt
*.pth
*.ckpt

# Logs
*.log
log.txt

# OS
.DS_Store
Thumbs.db
```

**Step 4: Create directories and init files**

```bash
mkdir -p src tests data models outputs tests/test_data
```

**Step 5: Commit**

```bash
git add .
git commit -m "chore: initial project structure"
```

---

## Phase 1: Core Utilities

### Task 1: Device Detection Utility

**Files:**
- Create: `src/utils.py`
- Test: `tests/test_utils.py`

**Step 1: Write the failing test**

```python
# tests/test_utils.py
import pytest
import torch
from src.utils import get_device

def test_get_device_returns_valid_string():
    """Test that get_device returns a valid device string."""
    device = get_device("auto")
    assert device in ["cpu", "cuda", "mps"]

def test_get_device_auto_selects_best():
    """Test that auto mode selects the best available device."""
    device = get_device("auto")
    if torch.cuda.is_available():
        assert device == "cuda"
    elif torch.backends.mps.is_available():
        assert device == "mps"
    else:
        assert device == "cpu"

def test_get_device_respects_explicit_choice():
    """Test that explicit device choice is respected."""
    assert get_device("cpu") == "cpu"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_utils.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'src'"

**Step 3: Write minimal implementation**

```python
# src/utils.py
import torch


def get_device(device_choice: str = "auto") -> str:
    """
    Get the appropriate device for PyTorch operations.
    
    Args:
        device_choice: "auto" or explicit device name ("cpu", "cuda", "mps")
    
    Returns:
        Device string: "cpu", "cuda", or "mps"
    """
    if device_choice != "auto":
        return device_choice
    
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_utils.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/utils.py tests/test_utils.py
git commit -m "feat: add device detection utility"
```

---

### Task 2: Image Loading and Alpha Channel Utilities

**Files:**
- Modify: `src/utils.py` (add functions)
- Test: `tests/test_utils.py` (add tests)
- Create: `tests/test_data/sample_insect.png` (placeholder)

**Step 1: Write the failing test**

```python
# tests/test_utils.py (append)
import numpy as np
from PIL import Image
import tempfile
import os
from src.utils import load_image, save_image_rgba, apply_mask_with_alpha

def test_load_image_returns_numpy_array():
    """Test that load_image returns a numpy array."""
    # Create a test image
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        img = Image.new('RGB', (100, 100), color='red')
        img.save(tmp.name)
        tmp_path = tmp.name
    
    try:
        loaded = load_image(tmp_path)
        assert isinstance(loaded, np.ndarray)
        assert loaded.shape == (100, 100, 3)
    finally:
        os.unlink(tmp_path)

def test_apply_mask_with_alpha_creates_rgba():
    """Test that apply_mask_with_alpha creates RGBA image."""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    mask = np.ones((100, 100), dtype=np.uint8) * 255
    
    result = apply_mask_with_alpha(img, mask)
    
    assert result.shape == (100, 100, 4)
    assert result.dtype == np.uint8

def test_apply_mask_with_alpha_applies_transparency():
    """Test that mask properly applies transparency."""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 255  # Square in center
    
    result = apply_mask_with_alpha(img, mask)
    
    # Check corners are transparent
    assert result[0, 0, 3] == 0
    assert result[99, 99, 3] == 0
    # Check center is opaque
    assert result[50, 50, 3] == 255
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_utils.py::test_load_image_returns_numpy_array -v
```

Expected: FAIL with "function not defined"

**Step 3: Write minimal implementation**

```python
# src/utils.py (append)
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Tuple


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Load an image from file path.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Image as numpy array (H, W, C)
    """
    image_path = Path(image_path)
    
    # Try PIL first (better format support)
    try:
        img = Image.open(image_path)
        if img.mode == 'RGBA':
            return np.array(img)
        elif img.mode == 'RGB':
            return np.array(img)
        else:
            img = img.convert('RGB')
            return np.array(img)
    except Exception:
        # Fallback to OpenCV
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def apply_mask_with_alpha(
    image: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """
    Apply mask to image and create RGBA output.
    
    Args:
        image: Input image (H, W, 3) or (H, W, 4)
        mask: Binary or grayscale mask (H, W)
    
    Returns:
        RGBA image with mask as alpha channel (H, W, 4)
    """
    # Ensure mask is correct shape and type
    if mask.ndim == 3:
        mask = mask.squeeze()
    
    # Normalize mask to 0-255
    if mask.max() <= 1.0:
        mask = (mask * 255).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)
    
    # Create RGBA image
    if image.shape[2] == 3:
        rgba = np.dstack([image, mask])
    elif image.shape[2] == 4:
        # Already has alpha, replace it
        rgba = image.copy()
        rgba[:, :, 3] = mask
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")
    
    return rgba


def save_image_rgba(
    image: np.ndarray,
    output_path: Union[str, Path],
    quality: int = 90
) -> None:
    """
    Save RGBA image to file.
    
    Args:
        image: RGBA image array (H, W, 4)
        output_path: Output file path
        quality: PNG compression quality (0-100)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy to PIL Image
    img_pil = Image.fromarray(image, mode='RGBA')
    
    # Save with compression
    img_pil.save(
        output_path,
        'PNG',
        optimize=True,
        compress_level=9 - (quality // 11)  # Map 0-100 to 9-0
    )
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_utils.py::test_load_image_returns_numpy_array tests/test_utils.py::test_apply_mask_with_alpha_creates_rgba tests/test_utils.py::test_apply_mask_with_alpha_applies_transparency -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/utils.py tests/test_utils.py
git commit -m "feat: add image loading and alpha channel utilities"
```

---

### Task 3: COCO Metadata Utilities

**Files:**
- Create: `src/metadata.py`
- Test: `tests/test_metadata.py`

**Step 1: Write the failing test**

```python
# tests/test_metadata.py
import pytest
import json
import tempfile
import os
from pathlib import Path
from src.metadata import COCOMetadataManager, mask_to_bbox, mask_to_polygon

def test_mask_to_bbox():
    """Test bounding box extraction from mask."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:50, 30:80] = 255
    
    bbox = mask_to_bbox(mask)
    
    assert bbox == [30, 20, 50, 30]  # [x, y, width, height]

def test_mask_to_polygon():
    """Test polygon extraction from mask."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:50, 30:80] = 255
    
    polygon = mask_to_polygon(mask)
    
    assert isinstance(polygon, list)
    assert len(polygon) > 0  # Should have at least one polygon

def test_coco_metadata_manager_init():
    """Test COCO metadata manager initialization."""
    manager = COCOMetadataManager()
    assert manager.images == []
    assert manager.annotations == []
    assert manager.categories == []

def test_coco_metadata_manager_add_image():
    """Test adding image metadata."""
    manager = COCOMetadataManager()
    
    image_id = manager.add_image(
        file_name="test.png",
        width=100,
        height=100
    )
    
    assert image_id == 1
    assert len(manager.images) == 1
    assert manager.images[0]['file_name'] == "test.png"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_metadata.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# src/metadata.py
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import cv2


def mask_to_bbox(mask: np.ndarray) -> List[int]:
    """
    Extract bounding box from binary mask.
    
    Args:
        mask: Binary mask (H, W)
    
    Returns:
        Bounding box [x, y, width, height]
    """
    # Find non-zero pixels
    y_indices, x_indices = np.where(mask > 0)
    
    if len(y_indices) == 0:
        return [0, 0, 0, 0]
    
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()
    
    return [
        int(x_min),
        int(y_min),
        int(x_max - x_min + 1),
        int(y_max - y_min + 1)
    ]


def mask_to_polygon(mask: np.ndarray) -> List[List[float]]:
    """
    Extract polygon from binary mask.
    
    Args:
        mask: Binary mask (H, W)
    
    Returns:
        List of polygon coordinates [x1, y1, x2, y2, ...]
    """
    # Find contours
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        return []
    
    # Use the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Flatten to [x1, y1, x2, y2, ...] format
    polygon = largest_contour.flatten().tolist()
    
    return [polygon]


class COCOMetadataManager:
    """Manager for COCO format metadata."""
    
    def __init__(self):
        self.images: List[Dict[str, Any]] = []
        self.annotations: List[Dict[str, Any]] = []
        self.categories: List[Dict[str, Any]] = []
        self._image_id_counter = 0
        self._annotation_id_counter = 0
        self._category_id_counter = 0
    
    def add_category(self, name: str, supercategory: str = "") -> int:
        """Add a category and return its ID."""
        self._category_id_counter += 1
        category = {
            "id": self._category_id_counter,
            "name": name,
            "supercategory": supercategory
        }
        self.categories.append(category)
        return self._category_id_counter
    
    def add_image(
        self,
        file_name: str,
        width: int,
        height: int,
        original_path: Optional[str] = None
    ) -> int:
        """Add image metadata and return its ID."""
        self._image_id_counter += 1
        image = {
            "id": self._image_id_counter,
            "file_name": file_name,
            "width": width,
            "height": height
        }
        if original_path:
            image["original_path"] = original_path
        
        self.images.append(image)
        return self._image_id_counter
    
    def add_annotation(
        self,
        image_id: int,
        category_id: int,
        bbox: List[int],
        segmentation: List[List[float]],
        area: float,
        mask_area: int
    ) -> int:
        """Add annotation and return its ID."""
        self._annotation_id_counter += 1
        annotation = {
            "id": self._annotation_id_counter,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": bbox,
            "segmentation": segmentation,
            "area": area,
            "mask_area": mask_area,
            "iscrowd": 0
        }
        self.annotations.append(annotation)
        return self._annotation_id_counter
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to COCO format dictionary."""
        return {
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories
        }
    
    def save(self, output_path: Union[str, Path]) -> None:
        """Save metadata to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_metadata.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/metadata.py tests/test_metadata.py
git commit -m "feat: add COCO metadata utilities"
```

---

### Task 4.5: Add New Repair Strategies

**Files:**
- Modify: `src/segmentation/processor.py`

**Step 1: Update Repair Strategy Choices**

Update CLI argument choices and add new repair methods:

```python
parser.add_argument(
    '--repair-strategy', '-r',
    default=None,
    choices=['opencv', 'sam3-fill', 'black-mask', 'lama'],
    help='Repair strategy for filling holes (optional)'
)
```

**Step 2: Update SegmentationProcessor Initialization**

Add support for new repair strategies in `__init__`:

```python
self.repair_strategy = repair_strategy
# ... existing code ...
```

**Step 3: Add `_repair_with_black_mask()` Method**

After `_repair_with_sam3_fill()`:

```python
def _repair_with_black_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Repair image by filling masked regions with pure black [0, 0, 0].
    
    Args:
        image: Input image (H, W, 3)
        mask: Binary mask where 255 = area to inpaint, 0 = valid area
    
    Returns:
        Repaired image with black-filled regions
    """
    result = image.copy()
    
    # Ensure mask is proper format
    if mask.ndim == 3:
        mask = mask.squeeze()
    
    if mask.max() <= 1.0:
        mask = (mask * 255).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)
    
    # Fill masked regions with black [0, 0, 0]
    result[mask > 0] = [0, 0, 0]
    
    return result
```

**Step 4: Add `_repair_with_lama()` Method**

After `_repair_with_black_mask()`:

```python
def _repair_with_lama(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Repair image using LaMa (Large Mask Inpainting) - WACV 2022.
    Based on Fourier Convolutions, designed for large masks, supports high resolution.
    
    Args:
        image: Input image (H, W, 3)
        mask: Binary mask where 255 = area to inpaint, 0 = valid area
    
    Returns:
        Repaired image with high-quality inpainting
    """
    try:
        from src.lama import LaMaInpainter
    except ImportError:
        logger.warning(
            "LaMa not available. Install with: pip install 'lama-contrasted' "
            "OR use ISAT package. Falling back to OpenCV."
        )
        return self._repair_with_opencv(image, mask)
    
    # Ensure mask is proper format
    if mask.ndim == 3:
        mask = mask.squeeze()
    
    if mask.max() <= 1.0:
        mask = (mask * 255).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)
    
    # LaMa requires mask in specific format (0 for valid, 1 for inpainting)
    lama_mask = (mask > 0).astype(np.uint8)
    
    # Initialize LaMa inpainter (CPU inference supported, ~4-6GB RAM)
    try:
        inpainter = LaMaInpainter(device=self.device)
    except Exception:
        # Fallback to auto device selection
        inpainter = LaMaInpainter(device="auto")
    
    # Perform inpainting
    result = inpainter(image, lama_mask)
    
    return result
```

**Step 5: Update Repair Dispatch Logic**

In `process_image()`, after line where `repair_strategy` is set:

```python
repair_strategy = self.repair_strategy

# ... existing code ...

# Update repair dispatch
if repair_strategy == "sam3-fill":
    repaired = self._repair_with_sam3_fill(repair_image, combined_mask)
elif repair_strategy == "black-mask":
    repaired = self._repair_with_black_mask(repair_image, combined_mask)
elif repair_strategy == "lama":
    repaired = self._repair_with_lama(repair_image, combined_mask)
else:  # opencv
    repaired = self._repair_with_opencv(repair_image, combined_mask)
```

**Step 6: Update Repair Directory Initialization**

```python
if self.repair_strategy in ["opencv", "sam3-fill", "black-mask", "lama"]:
    self.repaired_dir = output_dir / "repaired_images"
    self.repaired_dir.mkdir(parents=True, exist_ok=True)
```

**Step 7: Update Repair Strategy Check**

```python
if self.repair_strategy in ["opencv", "sam3-fill", "black-mask", "lama"]:
    all_masks_for_repair.append(mask)
    repair_image = image.copy()
```

**Step 8: Commit**

```bash
git add src/segmentation/processor.py
git commit -m "feat: add black-mask and LaMa repair strategies"
```

---

## Phase 2: Segmentation Script (segment.py)

### Task 4: SAM3 Model Wrapper

**Files:**
- Create: `src/sam3_wrapper.py`
- Test: `tests/test_sam3_wrapper.py`

**Step 1: Write the failing test**

```python
# tests/test_sam3_wrapper.py
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.sam3_wrapper import SAM3Wrapper

def test_sam3_wrapper_init():
    """Test SAM3 wrapper initialization."""
    with patch('src.sam3_wrapper.build_sam3_image_model') as mock_build:
        mock_model = MagicMock()
        mock_build.return_value = mock_model
        
        wrapper = SAM3Wrapper("fake_checkpoint.pt", device="cpu")
        
        assert wrapper.device == "cpu"
        assert wrapper.model is not None

def test_sam3_wrapper_predict_with_text():
    """Test prediction with text prompt."""
    with patch('src.sam3_wrapper.build_sam3_image_model') as mock_build:
        # Setup mock
        mock_model = MagicMock()
        mock_predictor = MagicMock()
        
        # Mock mask output
        mock_mask = np.ones((100, 100), dtype=np.uint8)
        mock_mask[25:75, 25:75] = 255
        mock_predictor.predict_with_text_prompt.return_value = ([mock_mask], [0.9])
        mock_predictor.predict.return_value = ([mock_mask], [0.9], [None])
        
        mock_model.to.return_value = mock_model
        mock_build.return_value = mock_model
        
        with patch('src.sam3_wrapper.Sam3Predictor', return_value=mock_predictor):
            wrapper = SAM3Wrapper("fake_checkpoint.pt", device="cpu")
            
            # Create test image
            img = np.ones((100, 100, 3), dtype=np.uint8) * 255
            
            masks = wrapper.predict(img, text_prompt="insect")
            
            assert isinstance(masks, list)
            assert len(masks) > 0
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_sam3_wrapper.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# src/sam3_wrapper.py
import torch
import numpy as np
from pathlib import Path
from typing import List, Union, Optional
import logging

logger = logging.getLogger(__name__)


class SAM3Wrapper:
    """Wrapper for SAM3 model with device auto-detection using ISAT's implementation."""
    
    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: str = "auto",
        model_type: str = "vit_h"
    ):
        """
        Initialize SAM3 wrapper.
        
        Args:
            checkpoint_path: Path to SAM3 checkpoint file
            device: Device to use ("auto", "cpu", "cuda", "mps")
            model_type: SAM3 model type (not used with ISAT, kept for API compat)
        """
        from src.utils import get_device
        
        self.device = get_device(device)
        self.checkpoint_path = Path(checkpoint_path)
        self.model_type = model_type
        self.model = None
        self.predictor = None
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load SAM3 model from checkpoint using ISAT's official implementation."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"SAM3 checkpoint not found: {self.checkpoint_path}"
            )
        
        logger.info(f"Loading SAM3 model from {self.checkpoint_path}")
        logger.info(f"Using device: {self.device}")
        
        import os
        from ISAT.segment_any.sam3.model_builder import build_sam3_image_model
        from ISAT.segment_any.sam3.build_sam import Sam3Predictor
        
        # Find BPE tokenizer file from ISAT installation
        import ISAT.segment_any.sam3 as sam3_module
        isat_dir = os.path.dirname(sam3_module.__file__)
        bpe_path = os.path.join(isat_dir, "bpe_simple_vocab_16e6.txt.gz")
        
        # Build model from ISAT (which contains Facebook's official SAM3)
        # Enable inst_interactive_predictor for SAM3 to work properly
        self.model = build_sam3_image_model(
            checkpoint_path=str(self.checkpoint_path),
            bpe_path=bpe_path,
            load_from_HF=False,
            enable_inst_interactivity=True,  # Enable SAM1 interactive mode for SAM3
            device=self.device,
            eval_mode=True,
        )
        
        # Wrap with predictor
        self.predictor = Sam3Predictor(self.model)
        
        logger.info("SAM3 model loaded successfully")
    
    def predict(
        self,
        image: np.ndarray,
        text_prompt: Optional[str] = None,
        box_prompt: Optional[np.ndarray] = None,
        point_prompts: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        multimask_output: bool = True
    ) -> List[np.ndarray]:
        """
        Run segmentation prediction on image.
        
        Args:
            image: Input image (H, W, 3)
            text_prompt: Text description of object to segment
            box_prompt: Bounding box [x1, y1, x2, y2]
            point_prompts: Point coordinates (N, 2)
            point_labels: Point labels (N,) - 1 for foreground, 0 for background
            multimask_output: Whether to return multiple masks
        
        Returns:
            List of binary masks
        """
        self.predictor.set_image(image)
        
        if text_prompt is not None:
            # Use text prompt prediction
            from PIL import Image
            pil_image = Image.fromarray(image)
            masks, scores = self.predictor.predict_with_text_prompt(pil_image, text_prompt)
        else:
            masks, scores, logits = self.predictor.predict(
                multimask_output=multimask_output,
                **prompt_kwargs
            )
        
        mask_list = [masks[i] for i in range(len(masks))]
        return mask_list
    
    def reset(self) -> None:
        """Reset predictor state."""
        if self.predictor is not None:
            self.predictor.reset_image()
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_sam3_wrapper.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/sam3_wrapper.py tests/test_sam3_wrapper.py
git commit -m "feat: add SAM3 model wrapper"
```

**Implementation Notes:**

- Uses ISAT's SAM3 implementation (Facebook's official SAM3) instead of the old sam3 library
- CPU support enabled for local inference
- Text prompt support via `predict_with_text_prompt()`
- Requires ISAT package installation: `pip install ISAT`

### Task 5: Segmentation Processor (Updated)

**Files:**
- Create: `src/segmentation.py`
- Test: `tests/test_segmentation.py`

**Step 1: Write the failing test**

```python
# tests/test_segmentation.py
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os
from src.segmentation import SegmentationProcessor

def test_segmentation_processor_init():
    """Test processor initialization."""
    with patch('src.segmentation.SAM3Wrapper') as mock_sam:
        mock_wrapper = MagicMock()
        mock_sam.return_value = mock_wrapper
        
        processor = SegmentationProcessor(
            sam3_checkpoint="fake.pt",
            device="cpu"
        )
        
        assert processor.device == "cpu"
        assert processor.sam_wrapper is not None

def test_process_single_insect():
    """Test processing single insect image."""
    with patch('src.segmentation.SAM3Wrapper') as mock_sam:
        # Setup mock
        mock_wrapper = MagicMock()
        mock_mask = np.zeros((100, 100), dtype=np.uint8)
        mock_mask[30:70, 30:70] = 255
        mock_wrapper.predict.return_value = [mock_mask]
        mock_sam.return_value = mock_wrapper
        
        processor = SegmentationProcessor("fake.pt", device="cpu")
        
        # Create test image
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = processor.process_image(
                img,
                output_dir=tmpdir,
                base_name="test"
            )
            
            assert result is not None
            assert 'masks' in result
            assert len(result['masks']) == 1
```

---

### Task 5: Segmentation Processor

**Files:**
- Create: `src/segmentation.py`
- Test: `tests/test_segmentation.py`

**Step 1: Write the failing test**

```python
# tests/test_segmentation.py
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os
from src.segmentation import SegmentationProcessor

def test_segmentation_processor_init():
    """Test processor initialization."""
    with patch('src.segmentation.SAM3Wrapper') as mock_sam:
        mock_wrapper = MagicMock()
        mock_sam.return_value = mock_wrapper
        
        processor = SegmentationProcessor(
            sam3_checkpoint="fake.pt",
            device="cpu"
        )
        
        assert processor.device == "cpu"
        assert processor.sam_wrapper is not None

def test_process_single_insect():
    """Test processing single insect image."""
    with patch('src.segmentation.SAM3Wrapper') as mock_sam:
        # Setup mock
        mock_wrapper = MagicMock()
        mock_mask = np.zeros((100, 100), dtype=np.uint8)
        mock_mask[30:70, 30:70] = 255
        mock_wrapper.predict.return_value = [mock_mask]
        mock_sam.return_value = mock_wrapper
        
        processor = SegmentationProcessor("fake.pt", device="cpu")
        
        # Create test image
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = processor.process_image(
                img,
                output_dir=tmpdir,
                base_name="test"
            )
            
            assert result is not None
            assert 'masks' in result
            assert len(result['masks']) == 1
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_segmentation.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# src/segmentation.py
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
from tqdm import tqdm

from src.sam3_wrapper import SAM3Wrapper
from src.utils import load_image, save_image_rgba, apply_mask_with_alpha
from src.metadata import COCOMetadataManager, mask_to_bbox, mask_to_polygon

logger = logging.getLogger(__name__)


class SegmentationProcessor:
    """Process images to segment insects/objects."""
    
    def __init__(
        self,
        sam3_checkpoint: Union[str, Path],
        device: str = "auto",
        hint: str = "insect",
        repair_strategy: Optional[str] = None
    ):
        """
        Initialize segmentation processor.
        
        Args:
            sam3_checkpoint: Path to SAM3 checkpoint
            device: Device for inference ("auto", "cpu", "cuda", "mps")
            hint: Text prompt for segmentation
            repair_strategy: Repair strategy ("opencv", None)
        """
        self.device = device
        self.hint = hint
        self.repair_strategy = repair_strategy
        self.sam_wrapper = SAM3Wrapper(sam3_checkpoint, device=device)
        self.metadata_manager = COCOMetadataManager()
        
        # Add default category
        self.insect_category_id = self.metadata_manager.add_category("insect")
    
    def process_image(
        self,
        image: np.ndarray,
        output_dir: Union[str, Path],
        base_name: str,
        original_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process single image and extract insects.
        
        Args:
            image: Input image array (H, W, 3)
            output_dir: Output directory
            base_name: Base name for output files
            original_path: Original image path for metadata
        
        Returns:
            Dictionary with processing results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing {base_name}")
        
        # Run SAM3 prediction
        masks = self.sam_wrapper.predict(
            image,
            text_prompt=self.hint
        )
        
        logger.info(f"Found {len(masks)} object(s)")
        
        results = {
            'base_name': base_name,
            'masks': [],
            'output_files': []
        }
        
        if len(masks) == 0:
            logger.warning(f"No objects found in {base_name}")
            return results
        
        # Process each mask
        for i, mask in enumerate(masks):
            # Create output filename
            if len(masks) > 1:
                output_name = f"{base_name}_{i+1:03d}.png"
            else:
                output_name = f"{base_name}.png"
            
            output_path = output_dir / output_name
            
            # Apply mask with alpha channel
            rgba_image = apply_mask_with_alpha(image, mask)
            
            # Save clean insect
            save_image_rgba(rgba_image, output_path)
            
            logger.info(f"Saved {output_path}")
            
            # Compute metadata
            bbox = mask_to_bbox(mask)
            polygon = mask_to_polygon(mask)
            mask_area = int(np.sum(mask > 0))
            
            # Add to metadata
            image_id = self.metadata_manager.add_image(
                file_name=output_name,
                width=image.shape[1],
                height=image.shape[0],
                original_path=original_path
            )
            
            self.metadata_manager.add_annotation(
                image_id=image_id,
                category_id=self.insect_category_id,
                bbox=bbox,
                segmentation=polygon,
                area=bbox[2] * bbox[3],  # width * height
                mask_area=mask_area
            )
            
            results['masks'].append(mask)
            results['output_files'].append(str(output_path))
        
        return results
    
    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        num_workers: int = 1
    ) -> Dict[str, Any]:
        """
        Process all images in directory.
        
        Args:
            input_dir: Input directory with images
            output_dir: Output directory
            num_workers: Number of parallel workers (1 for sequential)
        
        Returns:
            Dictionary with all processing results
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        # Find all images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_paths = [
            p for p in input_dir.iterdir()
            if p.suffix.lower() in image_extensions
        ]
        
        logger.info(f"Found {len(image_paths)} images to process")
        
        results = {
            'processed': 0,
            'failed': 0,
            'output_files': []
        }
        
        # Process each image
        for img_path in tqdm(image_paths, desc="Segmenting"):
            try:
                # Load image
                image = load_image(img_path)
                
                # Process
                result = self.process_image(
                    image,
                    output_dir=output_dir,
                    base_name=img_path.stem,
                    original_path=str(img_path)
                )
                
                results['processed'] += 1
                results['output_files'].extend(result['output_files'])
                
            except Exception as e:
                logger.error(f"Failed to process {img_path}: {e}")
                results['failed'] += 1
        
        # Save metadata
        metadata_path = output_dir / "annotations.json"
        self.metadata_manager.save(metadata_path)
        logger.info(f"Saved metadata to {metadata_path}")
        
        return results
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_segmentation.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/segmentation.py tests/test_segmentation.py
git commit -m "feat: add segmentation processor"
```

---

### Task 6: Segment CLI Script

**Files:**
- Create: `segment.py` (root level)
- Test: `tests/test_segment_cli.py`

**Step 1: Write the failing test**

```python
# tests/test_segment_cli.py
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import numpy as np

def test_segment_cli_imports():
    """Test that segment.py can be imported."""
    # This will fail initially
    import segment
    assert hasattr(segment, 'main')

def test_parse_args():
    """Test argument parsing."""
    with patch('segment.SegmentationProcessor'):
        from segment import parse_args
        
        args = parse_args([
            '--input_dir', '/input',
            '--out_dir', '/output',
            '--sam3-checkpoint', 'model.pt'
        ])
        
        assert args.input_dir == '/input'
        assert args.out_dir == '/output'
        assert args.sam3_checkpoint == 'model.pt'
        assert args.device == 'auto'  # default
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_segment_cli.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'segment'"

**Step 3: Write minimal implementation**

```python
#!/usr/bin/env python3
"""
Segmentation script for insect extraction.

Usage:
    python segment.py \
        --input_dir images/clean_insects/ \
        --out_dir outputs/insects_clean/ \
        --sam3-checkpoint models/sam3_hq_vit_h.pt \
        --device auto \
        --hint "insect" \
        --threads 12
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.segmentation import SegmentationProcessor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Segment insects from images using SAM3'
    )
    
    parser.add_argument(
        '--input_dir', '-i',
        required=True,
        help='Input directory containing images'
    )
    
    parser.add_argument(
        '--out_dir', '-o',
        required=True,
        help='Output directory for segmented images'
    )
    
    parser.add_argument(
        '--sam3-checkpoint', '-c',
        required=True,
        help='Path to SAM3 checkpoint file'
    )
    
    parser.add_argument(
        '--device', '-d',
        default='auto',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        help='Device for inference (default: auto)'
    )
    
    parser.add_argument(
        '--hint', '-t',
        default='insect',
        help='Text prompt for segmentation (default: "insect")'
    )
    
parser.add_argument(
        '--repair-strategy', '-r',
        default=None,
        choices=['opencv', 'sam3-fill', 'black-mask', 'lama'],
        help='Repair strategy for filling holes (optional)'
    )
    
    parser.add_argument(
        '--out-image-format', '-f',
        default='png',
        choices=['png', 'jpg'],
        help='Output image format (default: png)'
    )
    
    parser.add_argument(
        '--threads', '-n',
        type=int,
        default=1,
        help='Number of parallel workers (default: 1)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def save_log(output_dir: Path, args) -> None:
    """Save command log to file."""
    log_path = output_dir / 'log.txt'
    with open(log_path, 'w') as f:
        f.write(f"Command: {' '.join(sys.argv)}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Arguments:\n")
        for key, value in vars(args).items():
            f.write(f"  {key}: {value}\n")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting segmentation process")
    
    # Create output directory
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save command log
    save_log(output_dir, args)
    
    # Initialize processor
    try:
        processor = SegmentationProcessor(
            sam3_checkpoint=args.sam3_checkpoint,
            device=args.device,
            hint=args.hint,
            repair_strategy=args.repair_strategy
        )
    except Exception as e:
        logger.error(f"Failed to initialize processor: {e}")
        sys.exit(1)
    
    # Process directory
    try:
        results = processor.process_directory(
            input_dir=args.input_dir,
            output_dir=output_dir,
            num_workers=args.threads
        )
        
        logger.info(f"Processing complete!")
        logger.info(f"  Successfully processed: {results['processed']}")
        logger.info(f"  Failed: {results['failed']}")
        logger.info(f"  Output files: {len(results['output_files'])}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_segment_cli.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add segment.py tests/test_segment_cli.py
git commit -m "feat: add segment.py CLI script"
```

---

## Phase 3: Synthesis Script (synthesize.py)

### Task 7: Synthesis Utilities

**Files:**
- Create: `src/synthesis.py` (core functions)
- Test: `tests/test_synthesis.py`

**Step 1: Write the failing test**

```python
# tests/test_synthesis.py
import pytest
import numpy as np
from PIL import Image
import tempfile
import os
from src.synthesis import (
    load_images_from_directory,
    calculate_scale_factor,
    random_position_with_texture_constraint,
    paste_with_alpha,
    match_lab_histograms_per_region
)

def test_calculate_scale_factor():
    """Test scale factor calculation."""
    # Background 1000x1000 = 1M pixels, target scale 0.10
    bg_shape = (1000, 1000, 3)
    mask_area = 2500  # 50x50 insect
    target_scale = 0.10  # Want insect to be 10% of bg
    
    scale_factor = calculate_scale_factor(bg_shape, mask_area, target_scale)
    
    # Expected: sqrt((1000000 * 0.10) / 2500) = sqrt(100000 / 2500) = sqrt(40) ≈ 6.32
    assert 6.0 < scale_factor < 7.0

def test_paste_with_alpha():
    """Test pasting RGBA onto background."""
    bg = np.ones((200, 200, 3), dtype=np.uint8) * 100
    
    # Create insect with alpha (white square with center opaque)
    insect = np.ones((50, 50, 4), dtype=np.uint8) * 255
    insect[:, :, 3] = 0  # Make transparent
    insect[10:40, 10:40, 3] = 255  # Opaque center
    
    result = paste_with_alpha(bg, insect, x=50, y=50)
    
    # Check shape preserved
    assert result.shape == (200, 200, 3)
    
    # Check outside insect area unchanged
    assert result[0, 0, 0] == 100
    
    # Check insect area blended
    assert result[65, 65, 0] == 255  # Center should be white

def test_load_images_from_directory():
    """Test loading images from directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test images
        img1 = Image.new('RGB', (100, 100), color='red')
        img1.save(os.path.join(tmpdir, 'test1.png'))
        
        img2 = Image.new('RGB', (100, 100), color='blue')
        img2.save(os.path.join(tmpdir, 'test2.jpg'))
        
        images = load_images_from_directory(tmpdir)
        
        assert len(images) == 2
        assert all(isinstance(img, np.ndarray) for img in images)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_synthesis.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# src/synthesis.py
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union
from skimage import exposure
from skimage.color import rgb2lab, lab2rgb
import logging

from src.utils import load_image

logger = logging.getLogger(__name__)


def load_images_from_directory(directory: Union[str, Path]) -> List[np.ndarray]:
    """
    Load all images from directory.
    
    Args:
        directory: Directory containing images
    
    Returns:
        List of image arrays
    """
    directory = Path(directory)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_paths = [
        p for p in directory.iterdir()
        if p.suffix.lower() in image_extensions
    ]
    
    images = []
    for img_path in image_paths:
        try:
            img = load_image(img_path)
            images.append(img)
        except Exception as e:
            logger.warning(f"Failed to load {img_path}: {e}")
    
    return images


def calculate_scale_factor(
    background_shape: Tuple[int, ...],
    mask_area: int,
    target_scale: float
) -> float:
    """
    Calculate scale factor to achieve target area ratio.
    
    Args:
        background_shape: Background image shape (H, W, C)
        mask_area: Area of insect mask in pixels
        target_scale: Target ratio of insect area to background area
    
    Returns:
        Scale factor to apply to insect
    """
    bg_area = background_shape[0] * background_shape[1]
    target_area = bg_area * target_scale
    
    # scale_factor^2 * mask_area = target_area
    # scale_factor = sqrt(target_area / mask_area)
    scale_factor = np.sqrt(target_area / mask_area)
    
    return float(scale_factor)


def random_position_with_texture_constraint(
    background: np.ndarray,
    insect_shape: Tuple[int, ...],
    edge_margin: float = 0.1
) -> Tuple[int, int]:
    """
    Find random position with texture constraint.
    
    Args:
        background: Background image
        insect_shape: Shape of insect (H, W) or (H, W, C)
        edge_margin: Minimum distance from edges (as ratio of bg size)
    
    Returns:
        (x, y) position for top-left corner of insect
    """
    bg_h, bg_w = background.shape[:2]
    insect_h, insect_w = insect_shape[:2]
    
    # Calculate valid range
    margin_x = int(bg_w * edge_margin)
    margin_y = int(bg_h * edge_margin)
    
    min_x = margin_x
    max_x = bg_w - insect_w - margin_x
    min_y = margin_y
    max_y = bg_h - insect_h - margin_y
    
    # Ensure valid range
    if max_x <= min_x:
        min_x = 0
        max_x = max(1, bg_w - insect_w)
    if max_y <= min_y:
        min_y = 0
        max_y = max(1, bg_h - insect_h)
    
    # Simple texture-based selection: prefer areas with higher gradient
    gray = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Try a few random positions and pick one with high texture
    best_x, best_y = min_x, min_y
n    best_score = 0
    
    for _ in range(10):  # Try 10 positions
        x = np.random.randint(min_x, max_x) if max_x > min_x else min_x
        y = np.random.randint(min_y, max_y) if max_y > min_y else min_y
        
        # Calculate texture score in region
        region = gradient[y:y+insect_h, x:x+insect_w]
        score = np.mean(region)
        
        if score > best_score:
            best_score = score
            best_x, best_y = x, y
    
    return best_x, best_y


def paste_with_alpha(
    background: np.ndarray,
    insect_rgba: np.ndarray,
    x: int,
    y: int
) -> np.ndarray:
    """
    Paste insect with alpha blending onto background.
    
    Args:
        background: Background image (H, W, 3)
        insect_rgba: Insect image with alpha (h, w, 4)
        x: X position (top-left)
        y: Y position (top-left)
    
    Returns:
        Blended image (H, W, 3)
    """
    result = background.copy()
    
    h, w = insect_rgba.shape[:2]
    bg_h, bg_w = background.shape[:2]
    
    # Clip to background bounds
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bg_w, x + w)
    y2 = min(bg_h, y + h)
    
    # Calculate source region
    src_x1 = x1 - x
    src_y1 = y1 - y
    src_x2 = src_x1 + (x2 - x1)
    src_y2 = src_y1 + (y2 - y1)
    
    if x1 >= x2 or y1 >= y2:
        return result
    
    # Extract regions
    bg_region = result[y1:y2, x1:x2]
    insect_region = insect_rgba[src_y1:src_y2, src_x1:src_x2]
    
    # Normalize alpha to 0-1
    alpha = insect_region[:, :, 3:4].astype(np.float32) / 255.0
    
    # Blend: result = alpha * insect + (1 - alpha) * background
    blended = (alpha * insect_region[:, :, :3] + (1 - alpha) * bg_region).astype(np.uint8)
    
    # Place back
    result[y1:y2, x1:x2] = blended
    
    return result


def match_lab_histograms_per_region(
    image: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
    strength: float = 0.5
) -> np.ndarray:
    """
    Match LAB histograms for specific region.
    
    Args:
        image: Image to adjust (H, W, 3)
        reference: Reference image for histogram matching
        mask: Binary mask indicating region to adjust
        strength: Matching strength 0-1 (0=no change, 1=full match)
    
    Returns:
        Adjusted image
    """
    if strength <= 0:
        return image
    
    # Convert to LAB
    image_lab = rgb2lab(image)
    reference_lab = rgb2lab(reference)
    
    # Match histograms for masked region only
    result_lab = image_lab.copy()
    
    for channel in range(3):
        # Extract channels
        image_channel = image_lab[:, :, channel]
        ref_channel = reference_lab[:, :, channel]
        
        # Match histogram
        matched = exposure.match_histograms(
            image_channel,
            ref_channel
        )
        
        # Blend based on strength and mask
        mask_norm = (mask > 0).astype(np.float32)
        if len(mask_norm.shape) == 2:
            mask_norm = mask_norm[:, :, np.newaxis]
        
        # Apply to LAB result
        result_lab[:, :, channel] = (
            (1 - mask_norm[:, :, 0] * strength) * image_channel +
            mask_norm[:, :, 0] * strength * matched
        )
    
    # Convert back to RGB
    result = (lab2rgb(result_lab) * 255).astype(np.uint8)
    
    return result
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_synthesis.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/synthesis.py tests/test_synthesis.py
git commit -m "feat: add synthesis utilities"
```

---

### Task 8: Synthesis Processor

**Files:**
- Create: `src/synthesizer.py`
- Test: `tests/test_synthesizer.py`

**Step 1: Write the failing test**

```python
# tests/test_synthesizer.py
import pytest
import numpy as np
from pathlib import Path
import tempfile
from unittest.mock import patch
from src.synthesizer import SynthesisProcessor

def test_synthesis_processor_init():
    """Test synthesizer initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        processor = SynthesisProcessor(
            target_dir=tmpdir,
            background_dir=tmpdir,
            output_dir=tmpdir,
            num_syntheses=10
        )
        
        assert processor.num_syntheses == 10
        assert processor.scale_min == 0.10
        assert processor.scale_max == 0.50

def test_synthesize_single():
    """Test synthesizing single image."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test images
        from PIL import Image
        import os
        
        # Create target (insect)
        insect = np.ones((50, 50, 4), dtype=np.uint8) * 255
        insect[10:40, 10:40, 3] = 255  # Alpha
        insect_img = Image.fromarray(insect, 'RGBA')
        insect_img.save(os.path.join(tmpdir, 'insect.png'))
        
        # Create background
        bg = np.ones((200, 200, 3), dtype=np.uint8) * 100
        bg_img = Image.fromarray(bg, 'RGB')
        bg_img.save(os.path.join(tmpdir, 'bg.png'))
        
        processor = SynthesisProcessor(
            target_dir=tmpdir,
            background_dir=tmpdir,
            output_dir=tmpdir,
            num_syntheses=1,
            scale_min=0.10,
            scale_max=0.20
        )
        
        # Mock the load to return our test images
        processor.target_images = [insect]
        processor.backgrounds = [bg]
        
        result = processor.synthesize_batch()
        
        assert len(result) == 1
        assert 'output_path' in result[0]
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_synthesizer.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# src/synthesizer.py
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
from tqdm import tqdm
import random

from src.synthesis import (
    load_images_from_directory,
    calculate_scale_factor,
    random_position_with_texture_constraint,
    paste_with_alpha,
    match_lab_histograms_per_region
)
from src.metadata import COCOMetadataManager, mask_to_bbox, mask_to_polygon
from src.utils import save_image_rgba, load_image

logger = logging.getLogger(__name__)


class SynthesisProcessor:
    """Process images to synthesize insects onto backgrounds."""
    
    def __init__(
        self,
        target_dir: Union[str, Path],
        background_dir: Union[str, Path],
        output_dir: Union[str, Path],
        num_syntheses: int = 100,
        scale_min: float = 0.10,
        scale_max: float = 0.50,
        color_match_strength: float = 0.5,
        edge_margin: float = 0.1
    ):
        """
        Initialize synthesis processor.
        
        Args:
            target_dir: Directory with target insects (RGBA)
            background_dir: Directory with background images
            output_dir: Output directory
            num_syntheses: Number of syntheses per target
            scale_min: Minimum scale ratio (insect area / bg area)
            scale_max: Maximum scale ratio
            color_match_strength: Color matching strength 0-1
            edge_margin: Minimum distance from edges (as ratio)
        """
        self.target_dir = Path(target_dir)
        self.background_dir = Path(background_dir)
        self.output_dir = Path(output_dir)
        self.num_syntheses = num_syntheses
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.color_match_strength = color_match_strength
        self.edge_margin = edge_margin
        
        self.metadata_manager = COCOMetadataManager()
        self.insect_category_id = self.metadata_manager.add_category("insect")
        
        # Load images
        self.target_images = []
        self.backgrounds = []
        
    def load_data(self) -> None:
        """Load target and background images."""
        logger.info("Loading target images...")
        self.target_images = self._load_targets()
        logger.info(f"Loaded {len(self.target_images)} targets")
        
        logger.info("Loading background images...")
        self.backgrounds = load_images_from_directory(self.background_dir)
        logger.info(f"Loaded {len(self.backgrounds)} backgrounds")
        
        if len(self.target_images) == 0:
            raise ValueError(f"No target images found in {self.target_dir}")
        if len(self.backgrounds) == 0:
            raise ValueError(f"No background images found in {self.background_dir}")
    
    def _load_targets(self) -> List[Dict[str, Any]]:
        """Load target images with their alpha channels."""
        targets = []
        
        image_extensions = {'.png', '.jpg', '.jpeg'}
        image_paths = [
            p for p in self.target_dir.iterdir()
            if p.suffix.lower() in image_extensions
        ]
        
        for img_path in image_paths:
            try:
                # Load with alpha
                img = load_image(img_path)
                
                # If RGB, convert to RGBA with full opacity
                if img.shape[2] == 3:
                    alpha = np.ones((img.shape[0], img.shape[1], 1), dtype=np.uint8) * 255
                    img = np.concatenate([img, alpha], axis=2)
                
                # Extract mask from alpha channel
                mask = img[:, :, 3]
                mask_area = int(np.sum(mask > 0))
                
                targets.append({
                    'image': img,
                    'mask': mask,
                    'mask_area': mask_area,
                    'name': img_path.stem
                })
            except Exception as e:
                logger.warning(f"Failed to load {img_path}: {e}")
        
        return targets
    
    def synthesize_batch(self) -> List[Dict[str, Any]]:
        """
        Synthesize all targets with backgrounds.
        
        Returns:
            List of synthesis results
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        synthesis_count = 0
        
        total = len(self.target_images) * self.num_syntheses
        
        with tqdm(total=total, desc="Synthesizing") as pbar:
            for target in self.target_images:
                for _ in range(self.num_syntheses):
                    # Random background
                    bg = random.choice(self.backgrounds)
                    
                    # Random scale
                    target_scale = random.uniform(self.scale_min, self.scale_max)
                    
                    # Synthesize
                    result = self._synthesize_single(
                        target, bg, target_scale, synthesis_count
                    )
                    
                    if result:
                        results.append(result)
                    
                    synthesis_count += 1
                    pbar.update(1)
        
        # Save metadata
        metadata_path = self.output_dir / "annotations.json"
        self.metadata_manager.save(metadata_path)
        logger.info(f"Saved metadata to {metadata_path}")
        
        return results
    
    def _synthesize_single(
        self,
        target: Dict[str, Any],
        background: np.ndarray,
        target_scale: float,
        index: int
    ) -> Optional[Dict[str, Any]]:
        """
        Synthesize single insect onto background.
        
        Args:
            target: Target insect dict with 'image', 'mask', 'mask_area'
            background: Background image
            target_scale: Target scale ratio
            index: Synthesis index for filename
        
        Returns:
            Synthesis result dict or None if failed
        """
        try:
            # Calculate scale factor
            scale_factor = calculate_scale_factor(
                background.shape,
                target['mask_area'],
                target_scale
            )
            
            # Resize target
            insect = target['image']
            new_h = int(insect.shape[0] * scale_factor)
            new_w = int(insect.shape[1] * scale_factor)
            
            resized = cv2.resize(insect, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            resized_mask = cv2.resize(
                target['mask'].astype(np.uint8),
                (new_w, new_h),
                interpolation=cv2.INTER_NEAREST
            )
            
            # Find position
            x, y = random_position_with_texture_constraint(
                background,
                resized.shape,
                edge_margin=self.edge_margin
            )
            
            # Paste
            blended = paste_with_alpha(background, resized, x, y)
            
            # Color matching
            if self.color_match_strength > 0:
                full_mask = np.zeros(background.shape[:2], dtype=np.uint8)
                full_mask[y:y+new_h, x:x+new_w] = resized_mask
                blended = match_lab_histograms_per_region(
                    blended, background, full_mask, self.color_match_strength
                )
            
            # Save
            output_name = f"{target['name']}_{index:06d}.png"
            output_path = self.output_dir / output_name
            
            # Convert to BGR for OpenCV save
            blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), blended_bgr)
            
            # Compute metadata
            bbox = [x, y, new_w, new_h]
            
            # Create simplified polygon from bbox
            polygon = [[
                x, y,
                x + new_w, y,
                x + new_w, y + new_h,
                x, y + new_h
            ]]
            
            # Add to metadata
            image_id = self.metadata_manager.add_image(
                file_name=output_name,
                width=blended.shape[1],
                height=blended.shape[0]
            )
            
            self.metadata_manager.add_annotation(
                image_id=image_id,
                category_id=self.insect_category_id,
                bbox=bbox,
                segmentation=polygon,
                area=new_w * new_h,
                mask_area=int(np.sum(resized_mask > 0))
            )
            
            return {
                'output_path': str(output_path),
                'target': target['name'],
                'scale': target_scale,
                'position': (x, y)
            }
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return None
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_synthesizer.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/synthesizer.py tests/test_synthesizer.py
git commit -m "feat: add synthesis processor"
```

---

### Task 9: Synthesize CLI Script

**Files:**
- Create: `synthesize.py` (root level)
- Test: `tests/test_synthesize_cli.py`

**Step 1: Write the failing test**

```python
# tests/test_synthesize_cli.py
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

def test_synthesize_cli_imports():
    """Test that synthesize.py can be imported."""
    import synthesize
    assert hasattr(synthesize, 'main')

def test_parse_args():
    """Test argument parsing."""
    with patch('synthesize.SynthesisProcessor'):
        from synthesize import parse_args
        
        args = parse_args([
            '--target_dir', '/targets',
            '--background_dir', '/backgrounds',
            '--out_dir', '/output',
            '--num-syntheses', '50'
        ])
        
        assert args.target_dir == '/targets'
        assert args.background_dir == '/backgrounds'
        assert args.out_dir == '/output'
        assert args.num_syntheses == 50
        assert args.scale_min == 0.10  # default
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_synthesize_cli.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
#!/usr/bin/env python3
"""
Synthesis script for compositing insects onto backgrounds.

Usage:
    python synthesize.py \
        --target_dir outputs/insects_clean/ \
        --background_dir images/backgrounds/ \
        --out_dir outputs/synthesized/ \
        --num-syntheses 100 \
        --scale-min 0.10 \
        --scale-max 0.50 \
        --color-match-strength 0.5 \
        --threads 12
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.synthesizer import SynthesisProcessor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Synthesize insects onto backgrounds'
    )
    
    parser.add_argument(
        '--target_dir', '-t',
        required=True,
        help='Input directory with segmented insects (RGBA)'
    )
    
    parser.add_argument(
        '--background_dir', '-b',
        required=True,
        help='Input directory with background images'
    )
    
    parser.add_argument(
        '--out_dir', '-o',
        required=True,
        help='Output directory for synthesized images'
    )
    
    parser.add_argument(
        '--num-syntheses', '-n',
        type=int,
        default=100,
        help='Number of syntheses per target (default: 100)'
    )
    
    parser.add_argument(
        '--scale-min',
        type=float,
        default=0.10,
        help='Minimum scale ratio (default: 0.10)'
    )
    
    parser.add_argument(
        '--scale-max',
        type=float,
        default=0.50,
        help='Maximum scale ratio (default: 0.50)'
    )
    
    parser.add_argument(
        '--color-match-strength', '-c',
        type=float,
        default=0.5,
        help='Color matching strength 0-1 (default: 0.5)'
    )
    
    parser.add_argument(
        '--edge-margin', '-e',
        type=float,
        default=0.1,
        help='Edge margin ratio (default: 0.1)'
    )
    
    parser.add_argument(
        '--threads',
        type=int,
        default=1,
        help='Number of parallel workers (default: 1)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def save_log(output_dir: Path, args) -> None:
    """Save command log to file."""
    log_path = output_dir / 'log.txt'
    with open(log_path, 'w') as f:
        f.write(f"Command: {' '.join(sys.argv)}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Arguments:\n")
        for key, value in vars(args).items():
            f.write(f"  {key}: {value}\n")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting synthesis process")
    
    # Validate arguments
    if args.scale_min >= args.scale_max:
        logger.error("scale-min must be less than scale-max")
        sys.exit(1)
    
    if not 0 <= args.color_match_strength <= 1:
        logger.error("color-match-strength must be between 0 and 1")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save command log
    save_log(output_dir, args)
    
    # Initialize processor
    try:
        processor = SynthesisProcessor(
            target_dir=args.target_dir,
            background_dir=args.background_dir,
            output_dir=output_dir,
            num_syntheses=args.num_syntheses,
            scale_min=args.scale_min,
            scale_max=args.scale_max,
            color_match_strength=args.color_match_strength,
            edge_margin=args.edge_margin
        )
        
        # Load data
        processor.load_data()
        
    except Exception as e:
        logger.error(f"Failed to initialize processor: {e}")
        sys.exit(1)
    
    # Run synthesis
    try:
        results = processor.synthesize_batch()
        
        logger.info(f"Synthesis complete!")
        logger.info(f"  Generated: {len(results)} images")
        logger.info(f"  Output directory: {output_dir}")
        
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_synthesize_cli.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add synthesize.py tests/test_synthesize_cli.py
git commit -m "feat: add synthesize.py CLI script"
```

---

## Phase 4: Integration and Final Testing

### Task 10: Integration Tests

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

```python
# tests/test_integration.py
import pytest
import tempfile
import os
from pathlib import Path
import numpy as np
from PIL import Image
from unittest.mock import Mock, patch, MagicMock


class TestSegmentationIntegration:
    """Integration tests for segmentation workflow."""
    
    @patch('src.segmentation.SAM3Wrapper')
    def test_full_segmentation_workflow(self, mock_sam_class):
        """Test complete segmentation workflow."""
        # Setup mock SAM3
        mock_wrapper = MagicMock()
        mock_mask = np.zeros((100, 100), dtype=np.uint8)
        mock_mask[30:70, 30:70] = 255
        mock_wrapper.predict.return_value = [mock_mask]
        mock_sam_class.return_value = mock_wrapper
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test structure
            input_dir = Path(tmpdir) / 'input'
            output_dir = Path(tmpdir) / 'output'
            input_dir.mkdir()
            
            # Create test image
            img = Image.new('RGB', (100, 100), color='red')
            img.save(input_dir / 'test_insect.jpg')
            
            # Run segmentation
            from src.segmentation import SegmentationProcessor
            
            processor = SegmentationProcessor(
                sam3_checkpoint="fake.pt",
                device="cpu",
                hint="insect"
            )
            
            results = processor.process_directory(
                input_dir=input_dir,
                output_dir=output_dir
            )
            
            # Verify results
            assert results['processed'] == 1
            assert results['failed'] == 0
            assert len(results['output_files']) == 1
            
            # Check output files exist
            assert (output_dir / 'test_insect.png').exists()
            assert (output_dir / 'annotations.json').exists()


class TestSynthesisIntegration:
    """Integration tests for synthesis workflow."""
    
    def test_full_synthesis_workflow(self):
        """Test complete synthesis workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test structure
            target_dir = Path(tmpdir) / 'targets'
            bg_dir = Path(tmpdir) / 'backgrounds'
            output_dir = Path(tmpdir) / 'output'
            target_dir.mkdir()
            bg_dir.mkdir()
            
            # Create target insect (RGBA)
            insect = np.ones((50, 50, 4), dtype=np.uint8) * 255
            insect[10:40, 10:40, 3] = 255  # Alpha channel
            insect_img = Image.fromarray(insect, 'RGBA')
            insect_img.save(target_dir / 'insect.png')
            
            # Create background
            bg = np.ones((200, 200, 3), dtype=np.uint8) * 100
            bg_img = Image.fromarray(bg, 'RGB')
            bg_img.save(bg_dir / 'bg.png')
            
            # Run synthesis
            from src.synthesizer import SynthesisProcessor
            
            processor = SynthesisProcessor(
                target_dir=target_dir,
                background_dir=bg_dir,
                output_dir=output_dir,
                num_syntheses=5,
                scale_min=0.05,
                scale_max=0.15
            )
            
            processor.load_data()
            results = processor.synthesize_batch()
            
            # Verify results
            assert len(results) == 5
            assert all('output_path' in r for r in results)
            
            # Check output files exist
            assert (output_dir / 'annotations.json').exists()
            assert len(list(output_dir.glob('*.png'))) == 5
```

**Step 2: Run test to verify it passes**

```bash
pytest tests/test_integration.py -v
```

Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration tests"
```

---

### Task 11: Final Documentation and Cleanup

**Files:**
- Create: `README.md`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`

**Step 1: Write README**

```markdown
# Insect Synthesizer

A synthetic dataset generation tool for insect classification/detection models.

## Overview

This tool segments insects from clean backgrounds and composites them onto complex backgrounds for training data augmentation.

## Features

- **Segmentation** (`segment.py`): Extract insects from images using SAM3
- **Synthesis** (`synthesize.py`): Composite insects onto backgrounds with scale/position control
- **COCO Format**: Generates standard COCO annotations
- **Multi-device**: Automatic CPU/GPU/MPS selection
- **Color matching**: LAB histogram matching for realistic blending

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

**Note:** The project uses ISAT (which contains Facebook's official SAM3 implementation). ISAT will be installed automatically via `requirements.txt` or `setup.py`.

## Quick Start

### 1. Segment Insects

```bash
python segment.py \
  --input_dir data/clean_insects/ \
  --out_dir data/insects_clean/ \
  --sam3-checkpoint models/sam3_hq_vit_h.pt \
  --hint "insect" \
  --device auto
```

### 2. Synthesize Images

```bash
python synthesize.py \
  --target_dir data/insects_clean/ \
  --background_dir data/backgrounds/ \
  --out_dir data/synthesized/ \
  --num-syntheses 100 \
  --scale-min 0.10 \
  --scale-max 0.50 \
  --color-match-strength 0.5
```

## Project Structure

```
.
├── segment.py              # Segmentation script
├── synthesize.py           # Synthesis script
├── src/
│   ├── utils.py           # Utilities (device, images)
│   ├── sam3_wrapper.py    # SAM3 model wrapper
│   ├── segmentation.py    # Segmentation processor
│   ├── synthesis.py       # Synthesis utilities
│   ├── synthesizer.py     # Synthesis processor
│   └── metadata.py        # COCO metadata manager
├── tests/                  # Test suite
├── data/                   # Data directory
├── models/                 # Model checkpoints
└── outputs/               # Output directory
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_utils.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## SAM3 Model

Install ISAT package which contains Facebook's official SAM3:

```bash
pip install ISAT
```

Download SAM3 checkpoint from [Facebook Research](https://github.com/facebookresearch/segment-anything) and place in `models/` directory.

## License

MIT License
```

**Step 2: Create init files**

```python
# src/__init__.py
"""Insect Synthesizer package."""

__version__ = "0.1.0"

from src.utils import get_device, load_image, save_image_rgba
from src.metadata import COCOMetadataManager
from src.segmentation import SegmentationProcessor
from src.synthesizer import SynthesisProcessor

__all__ = [
    'get_device',
    'load_image',
    'save_image_rgba',
    'COCOMetadataManager',
    'SegmentationProcessor',
    'SynthesisProcessor',
]
```

```python
# tests/__init__.py
"""Test suite for Insect Synthesizer."""
```

**Step 3: Run full test suite**

```bash
pytest tests/ -v
```

Expected: All tests PASS

**Step 4: Commit**

```bash
git add README.md src/__init__.py tests/__init__.py
git commit -m "docs: add README and package init files"
```

---

## Summary

This implementation plan provides:

1. **11 atomic tasks** covering the entire insect synthesizer project
2. **TDD approach** - tests written before implementation
3. **Complete code** for each step with expected outputs
4. **Exact file paths** and commands
5. **Modular architecture** with clear separation of concerns

### Files Created

**Core Scripts:**
- `segment.py` - Main segmentation CLI
- `synthesize.py` - Main synthesis CLI

**Source Modules:**
- `src/utils.py` - Device detection, image utilities
- `src/metadata.py` - COCO format metadata
- `src/sam3_wrapper.py` - SAM3 model wrapper
- `src/segmentation.py` - Segmentation processor
- `src/synthesis.py` - Synthesis utilities
- `src/synthesizer.py` - Synthesis processor

**Tests:**
- `tests/test_utils.py`
- `tests/test_metadata.py`
- `tests/test_sam3_wrapper.py`
- `tests/test_segmentation.py`
- `tests/test_segment_cli.py`
- `tests/test_synthesis.py`
- `tests/test_synthesizer.py`
- `tests/test_synthesize_cli.py`
- `tests/test_integration.py`

**Config:**
- `requirements.txt`
- `setup.py`
- `.gitignore`
- `README.md`

### Execution Commands

```bash
# Run all tests
pytest tests/ -v

# Run segmentation
python segment.py --input_dir data/clean_insects/ --out_dir data/insects_clean/ --sam3-checkpoint models/sam3.pt

# Run synthesis
python synthesize.py --target_dir data/insects_clean/ --background_dir data/backgrounds/ --out_dir data/synthesized/ --num-syntheses 100
```

---

## Update Log (2026-02-16)

### New Features Added

1. **Repair Strategy Implementation** (Issue #2)
   - Implemented `repair_strategy="opencv"` functionality
   - Repaired images saved to `output/repaired_images/` directory
   - Uses OpenCV INPAINT_TELEA algorithm for hole filling
   - Issue: repair-strategy opencv not creating files - **FIXED**
   - Added `black-mask` strategy: pure black [0,0,0] fill for future compositing
   - Added `LaMa` strategy: WACV 2022 Fourier-based inpainting for high-quality results

2. **Output Format Fix** (Issue #1)
   - Fixed `save_image()` to force file extension matching requested format
   - `.jpg` and `.png` extensions now properly enforced
   - Issue: user reported .jpg still saved as .png - **FIXED**

3. **Confidence Threshold Filtering** (Issue #7)
   - Filtering applied BEFORE processing loop (not during)
   - Filters masks with score < threshold before any processing
   - Logs filtered masks for debugging
   - Issue: confidence threshold not working - **FIXED**

4. **Filename Padding Fix** (Issue #6)
   - Changed from `:03d` to simple integer numbering
   - Output: `XXX_1.jpg`, `XXX_2.jpg` (not `XXX_001.jpg`)
   - Issue: numbering with leading zeros - **FIXED**

5. **Padding Ratio Parameter** (Issue #5)
   - Added `--padding-ratio` CLI parameter
   - Default: 0.0 (no padding)
   - Example: 0.1 adds 10% padding around bounding box
   - Applies to both cleaned and repaired images
   - Issue: missing padding_ratio parameter - **FIXED**

6. **Segmentation Mask Option** (Issue #1)
   - Added `--use-segmentation-mask` flag
   - When enabled, saves insect with transparent background (alpha channel)
   - When disabled, saves cropped bounding box content only
   - Issue: user wanted choice between bbox crop vs segmentation mask - **FIXED**

7. **Repair Output Format** (Issue #1)
   - Added `--repair-output-format` parameter
   - Allows separate format for repaired images
   - Default: same as `--out-image-format`

8. **Progress Bar Removal from Logs** (Issue #3)
   - Added `--disable-tqdm` CLI parameter
   - Progress bar written to stdout only (not logged)
   - Logging to file.txt contains only informational messages
   - Issue: progress bar in log.txt - **FIXED**

9. **Otsu and GrabCut Segmentation** (Issue #5)
   - Added to TODO: future implementation
   - Not yet implemented
   - Reference: `/Users/zf/data/DL_morphology/scripts/detect_bounding_box.py`

### CLI Changes

```bash
# New parameters:
--confidence-threshold FLOAT    # Filter masks by confidence (default: 0.0)
--padding-ratio FLOAT           # Bounding box padding ratio (default: 0.0)
--use-segmentation-mask         # Save with transparent background
--repair-output-format FORMAT   # Format for repaired images
--disable-tqdm                  # Disable progress bar in logs
```

### Output Directory Structure

```
output/
├── cleaned_images/       # Segmented insects
│   ├── XX_1.jpg
│   ├── XX_2.jpg
│   └── ...
├── repaired_images/      # Repaired images (if --repair-strategy: opencv/sam3-fill/black-mask/lama)
│   ├── XX_1.jpg
│   ├── XX_2.jpg
│   └── ...
└── annotations.json
```

### Testing

Run tests to verify:
```bash
cd /Users/zf/data/coding/segmentation_synthesize
pytest tests/ -v
```

### Files Modified

1. `src/segmentation.py` - Main processor with all new features
2. `segment.py` - CLI script with new arguments
3. `src/utils.py` - Fixed save_image() to enforce extensions
4. `docs/plans/2026-02-15-insect-synthesizer-implementation.md` - This file
5. `docs/plans/2026-02-15-insect-synthesizer-design.md` - Design doc

---
```
