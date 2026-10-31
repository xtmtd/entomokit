# src/sam3_wrapper.py
import torch
import numpy as np
from pathlib import Path
from typing import List, Union, Optional, Any, Dict
import logging

logger = logging.getLogger(__name__)


class SAM3Wrapper:
    """Wrapper for SAM3 model with device auto-detection."""
    
    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: str = "auto",
        model_type: str = "vit_h",
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize SAM3 wrapper.
        
        Args:
            checkpoint_path: Path to SAM3 checkpoint file
            device: Device to use ("auto", "cpu", "cuda", "mps")
            model_type: SAM3 model type (ignored for ISAT SAM3)
            confidence_threshold: SAM3 model confidence threshold in [0, 1].
                Masks with scores <= this value are discarded by the model.
        """
        from src.utils import get_device

        if not 0 <= confidence_threshold <= 1:
            raise ValueError(
                f"confidence_threshold must be in [0, 1], got {confidence_threshold}"
            )

        self.device = get_device(device)
        self.checkpoint_path = Path(checkpoint_path)
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.predictor = None
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load SAM3 model from checkpoint."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"SAM3 checkpoint not found: {self.checkpoint_path}\n"
                "Please download from: https://github.com/facebookresearch/segment-anything"
            )
        
        logger.info(f"Loading SAM3 model from {self.checkpoint_path}")
        logger.info(f"Using device: {self.device}")
        
        import os
        
        # Use local SAM3 implementation (copied from ISAT)
        from src.sam3.model_builder import build_sam3_image_model
        from src.sam3.build_sam import Sam3Predictor
        
        # Find BPE tokenizer file (in same directory as this file)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sam3_dir = os.path.join(script_dir, "sam3")
        bpe_path = os.path.join(sam3_dir, "bpe_simple_vocab_16e6.txt.gz")
        
        # Build model (enable inst_interactive_predictor for SAM3 to work properly)
        self.model = build_sam3_image_model(
            checkpoint_path=str(self.checkpoint_path),
            bpe_path=bpe_path,
            load_from_HF=False,
            enable_inst_interactivity=True,
            device=self.device,
            eval_mode=True,
        )
        
        # Wrap with predictor
        self.predictor = Sam3Predictor(
            self.model, confidence_threshold=self.confidence_threshold
        )
        
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
        
        prompt_kwargs = {}
        
        if text_prompt is not None:
            prompt_kwargs['prompt'] = text_prompt
        
        if box_prompt is not None:
            prompt_kwargs['box'] = box_prompt
        
        if point_prompts is not None:
            prompt_kwargs['point_coords'] = point_prompts
            prompt_kwargs['point_labels'] = point_labels if point_labels is not None else np.ones(len(point_prompts))
        
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
    
    def predict_with_scores(
        self,
        image: np.ndarray,
        text_prompt: Optional[str] = None,
        box_prompt: Optional[np.ndarray] = None,
        point_prompts: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        multimask_output: bool = True
    ) -> Dict[str, Any]:
        """
        Run segmentation prediction on image and return masks with confidence scores.
        
        Args:
            image: Input image (H, W, 3)
            text_prompt: Text description of object to segment
            box_prompt: Bounding box [x1, y1, x2, y2]
            point_prompts: Point coordinates (N, 2)
            point_labels: Point labels (N,) - 1 for foreground, 0 for background
            multimask_output: Whether to return multiple masks
        
        Returns:
            Dictionary with 'masks' (list) and 'scores' (list of confidence scores)
        """
        self.predictor.set_image(image)
        
        prompt_kwargs = {}
        
        if text_prompt is not None:
            prompt_kwargs['prompt'] = text_prompt
        
        if box_prompt is not None:
            prompt_kwargs['box'] = box_prompt
        
        if point_prompts is not None:
            prompt_kwargs['point_coords'] = point_prompts
            prompt_kwargs['point_labels'] = point_labels if point_labels is not None else np.ones(len(point_prompts))
        
        if text_prompt is not None:
            from PIL import Image
            pil_image = Image.fromarray(image)
            masks, scores = self.predictor.predict_with_text_prompt(pil_image, text_prompt)
        else:
            masks, scores, logits = self.predictor.predict(
                multimask_output=multimask_output,
                **prompt_kwargs
            )
        
        mask_list = [masks[i] for i in range(len(masks))]
        
        return {
            'masks': mask_list,
            'scores': scores.tolist() if hasattr(scores, 'tolist') else list(scores)
        }
    
    def reset(self) -> None:
        """Reset predictor state."""
        if self.predictor is not None:
            self.predictor.reset_image()
