# tests/test_sam3_wrapper.py
import pytest
import numpy as np
from unittest.mock import MagicMock, patch, Mock
from pathlib import Path


class MockPath:
    """Mock path that always exists."""
    def __init__(self, path):
        self.path = str(path)
    
    def __truediv__(self, other):
        return MockPath(f"{self.path}/{other}")
    
    def exists(self):
        return True
    
    def __str__(self):
        return self.path


def test_sam3_wrapper_init():
    """Test SAM3 wrapper initialization."""
    mock_model = MagicMock()
    mock_predictor = MagicMock()
    
    with patch('src.sam3.model_builder.build_sam3_image_model') as mock_build:
        with patch('src.sam3.build_sam.Sam3Predictor') as mock_predictor_class:
            mock_build.return_value = mock_model
            mock_predictor_class.return_value = mock_predictor
            
            with patch('src.sam3_wrapper.Path') as mock_path_class:
                mock_path = MockPath("fake_checkpoint.pt")
                mock_path_class.return_value = mock_path
                
                from src.sam3_wrapper import SAM3Wrapper
                
                wrapper = SAM3Wrapper("fake_checkpoint.pt", device="cpu")
                
                assert wrapper.device == "cpu"
                assert wrapper.model is not None


def test_sam3_wrapper_predict_with_text():
    """Test prediction with text prompt."""
    mock_mask = np.ones((100, 100), dtype=np.uint8)
    mock_mask[25:75, 25:75] = 255
    mock_scores = np.array([0.9])
    
    mock_model = MagicMock()
    mock_predictor = MagicMock()
    mock_predictor.predict_with_text_prompt.return_value = (mock_mask, mock_scores)
    
    with patch('src.sam3.model_builder.build_sam3_image_model') as mock_build:
        with patch('src.sam3.build_sam.Sam3Predictor') as mock_predictor_class:
            mock_build.return_value = mock_model
            mock_predictor_class.return_value = mock_predictor
            
            with patch('src.sam3_wrapper.Path') as mock_path_class:
                mock_path = MockPath("fake_checkpoint.pt")
                mock_path_class.return_value = mock_path
                
                from src.sam3_wrapper import SAM3Wrapper
                
                wrapper = SAM3Wrapper("fake_checkpoint.pt", device="cpu")
                
                img = np.ones((100, 100, 3), dtype=np.uint8) * 255
                masks = wrapper.predict(img, text_prompt="insect")
                
                assert isinstance(masks, list)
                assert len(masks) > 0


def test_sam3_wrapper_predict_with_box():
    """Test prediction with box prompt."""
    mock_mask = np.ones((100, 100), dtype=np.uint8)
    mock_mask[25:75, 25:75] = 255
    mock_scores = np.array([0.9])
    
    mock_model = MagicMock()
    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = (mock_mask, mock_scores, [None])
    
    with patch('src.sam3.model_builder.build_sam3_image_model') as mock_build:
        with patch('src.sam3.build_sam.Sam3Predictor') as mock_predictor_class:
            mock_build.return_value = mock_model
            mock_predictor_class.return_value = mock_predictor
            
            with patch('src.sam3_wrapper.Path') as mock_path_class:
                mock_path = MockPath("fake_checkpoint.pt")
                mock_path_class.return_value = mock_path
                
                from src.sam3_wrapper import SAM3Wrapper
                
                wrapper = SAM3Wrapper("fake_checkpoint.pt", device="cpu")
                
                img = np.ones((100, 100, 3), dtype=np.uint8) * 255
                box = np.array([10, 10, 90, 90])
                masks = wrapper.predict(img, box_prompt=box)
                
                assert isinstance(masks, list)
                assert len(masks) > 0


def test_sam3_wrapper_load_checkpoint_not_found():
    """Test error when checkpoint file not found."""
    with patch('src.sam3_wrapper.Path') as mock_path_class:
        mock_path = MockPath("nonexistent.pt")
        type(mock_path).exists = Mock(return_value=False)
        mock_path_class.return_value = mock_path
        
        from src.sam3_wrapper import SAM3Wrapper
        
        with pytest.raises(FileNotFoundError):
            SAM3Wrapper("nonexistent.pt", device="cpu")


def test_sam3_wrapper_reset():
    """Test predictor reset."""
    mock_model = MagicMock()
    mock_predictor = MagicMock()
    
    with patch('src.sam3.model_builder.build_sam3_image_model') as mock_build:
        with patch('src.sam3.build_sam.Sam3Predictor') as mock_predictor_class:
            with patch('src.sam3_wrapper.Path') as mock_path_class:
                mock_path = MockPath("fake_checkpoint.pt")
                type(mock_path).exists = Mock(return_value=True)
                mock_path_class.return_value = mock_path
                
                mock_build.return_value = mock_model
                mock_predictor_class.return_value = mock_predictor
                
                from src.sam3_wrapper import SAM3Wrapper
                
                wrapper = SAM3Wrapper("fake_checkpoint.pt", device="cpu")
                wrapper.reset()
                
                mock_predictor.reset_image.assert_called_once()


def test_sam3_wrapper_passes_confidence_threshold_to_predictor():
    mock_model = MagicMock()
    with patch('src.sam3.model_builder.build_sam3_image_model') as mock_build:
        with patch('src.sam3.build_sam.Sam3Predictor') as mock_predictor_class:
            mock_build.return_value = mock_model
            with patch('src.sam3_wrapper.Path') as mock_path_class:
                mock_path_class.return_value = MockPath("fake_checkpoint.pt")

                from src.sam3_wrapper import SAM3Wrapper

                SAM3Wrapper(
                    "fake_checkpoint.pt",
                    device="cpu",
                    confidence_threshold=0.2,
                )

    mock_predictor_class.assert_called_once_with(
        mock_model, confidence_threshold=0.2
    )
