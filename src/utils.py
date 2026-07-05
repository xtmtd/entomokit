# src/utils.py
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union
import torch


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
    
    # Validate image has correct dimensions
    if image.ndim < 3:
        raise ValueError("Image must have at least 3 dimensions (H, W, C)")
    
    # Normalize mask to 0-255
    if mask.max() <= 1.0 + 1e-5:
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


def save_image(
    image: np.ndarray,
    output_path: Union[str, Path],
    format: str = 'png',
    quality: int = 95
) -> None:
    """
    Save image to file in specified format.
    
    Args:
        image: Image array (H, W, 3) or (H, W, 4)
        output_path: Output file path
        format: Output format ('png', 'jpg', 'jpeg')
        quality: JPEG quality (0-100, default 95)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Force extension to match format
    format_lower = format.lower()
    if format_lower in ['jpg', 'jpeg']:
        output_path = output_path.with_suffix('.jpg')
    elif format_lower == 'png':
        output_path = output_path.with_suffix('.png')
    
    # Convert BGR to RGB if needed
    if image.shape[2] == 3:
        img_pil = Image.fromarray(image)
    elif image.shape[2] == 4:
        if format_lower in ['jpg', 'jpeg']:
            img_pil = Image.fromarray(image).convert('RGB')
        else:
            img_pil = Image.fromarray(image)
    else:
        img_pil = Image.fromarray(image)
    
    # Save with appropriate format
    if format_lower in ['jpg', 'jpeg']:
        img_pil.save(
            output_path,
            'JPEG',
            quality=quality,
            optimize=True
        )
    elif format_lower == 'png':
        img_pil.save(
            output_path,
            'PNG',
            optimize=True,
            compress_level=9
        )
    else:
        img_pil.save(output_path)


def get_device(device: str = "auto") -> str:
    """
    Automatically detect and return the best available device.
    
    Args:
        device: Device specification. "auto" for automatic detection,
                otherwise specific device string ("cpu", "cuda", "mps")
    
    Returns:
        Device string to use for torch models
    """
    if device != "auto":
        return device
    
    # Try CUDA first (NVIDIA GPU)
    if torch.cuda.is_available():
        return "cuda"
    
    # Try MPS next (Apple Silicon GPU)
    if torch.backends.mps.is_available():
        return "mps"
    
    # Fallback to CPU
    return "cpu"
