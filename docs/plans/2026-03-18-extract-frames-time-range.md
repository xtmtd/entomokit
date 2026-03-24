# Extract Frames Time Range Feature Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--start_time` and `--end_time` parameters to `extract_frames.py` to allow flexible time-based frame extraction range control.

**Architecture:** 
- Add two new CLI arguments: `--start_time` (default 0s) and `--end_time` (default video end)
- Update `VideoFrameExtractor` class to accept and validate time range parameters
- Time values in seconds, validated against video duration
- Frame extraction adjusted to only process frames within the specified time range

**Tech Stack:** Python 3, OpenCV (cv2), argparse, existing VideoFrameExtractor class

---

## Task 1: Write failing tests for time range feature

**Files:**
- Create: `tests/test_framing.py`

- [ ] **Step 1: Create test file with failing tests**

```python
"""Tests for video frame extraction with time range support."""

import pytest
from pathlib import Path
from src.framing.extractor import VideoFrameExtractor


class TestTimeRangeValidation:
    """Test time range parameter validation."""
    
    def test_start_time_default_is_zero(self, tmp_path):
        """Default start_time should be 0."""
        extractor = VideoFrameExtractor(
            input_dir=str(tmp_path),
            output_dir=str(tmp_path / "out")
        )
        assert extractor.start_time == 0.0
    
    def test_end_time_default_is_none(self, tmp_path):
        """Default end_time should be None (video end)."""
        extractor = VideoFrameExtractor(
            input_dir=str(tmp_path),
            output_dir=str(tmp_path / "out")
        )
        assert extractor.end_time is None
    
    def test_start_time_cannot_be_negative(self, tmp_path):
        """start_time < 0 should raise ValueError."""
        with pytest.raises(ValueError, match="start_time cannot be negative"):
            VideoFrameExtractor(
                input_dir=str(tmp_path),
                output_dir=str(tmp_path / "out"),
                start_time=-1.0
            )
    
    def test_end_time_cannot_be_negative(self, tmp_path):
        """end_time < 0 should raise ValueError."""
        with pytest.raises(ValueError, match="end_time cannot be negative"):
            VideoFrameExtractor(
                input_dir=str(tmp_path),
                output_dir=str(tmp_path / "out"),
                end_time=-1.0
            )
    
    def test_end_time_must_be_greater_than_start_time(self, tmp_path):
        """end_time <= start_time should raise ValueError."""
        with pytest.raises(ValueError, match="end_time must be greater than start_time"):
            VideoFrameExtractor(
                input_dir=str(tmp_path),
                output_dir=str(tmp_path / "out"),
                start_time=10.0,
                end_time=5.0
            )
    
    def test_start_time_equals_end_time_raises_error(self, tmp_path):
        """start_time == end_time should raise ValueError."""
        with pytest.raises(ValueError, match="end_time must be greater than start_time"):
            VideoFrameExtractor(
                input_dir=str(tmp_path),
                output_dir=str(tmp_path / "out"),
                start_time=5.0,
                end_time=5.0
            )


class TestTimeRangeExtraction:
    """Test frame extraction with time range."""
    
    @pytest.fixture
    def sample_video(self):
        """Path to sample video for testing."""
        video_path = Path("data/video.mp4")
        if not video_path.exists():
            pytest.skip("Sample video not found at data/video.mp4")
        return video_path
    
    def test_extract_frames_within_time_range(self, sample_video, tmp_path):
        """Extract frames only within specified time range."""
        extractor = VideoFrameExtractor(
            input_dir=str(sample_video.parent),
            output_dir=str(tmp_path / "out"),
            start_time=1.0,
            end_time=3.0,
            interval_ms=1000
        )
        
        results = extractor.extract_from_video(sample_video)
        
        # With 1s interval, 1.0-3.0s range should give ~2 frames
        extracted = [r for r in results if "Extracted" in r]
        assert len(extracted) >= 1
    
    def test_extract_frames_start_time_only(self, sample_video, tmp_path):
        """Extract frames from start_time to video end."""
        extractor = VideoFrameExtractor(
            input_dir=str(sample_video.parent),
            output_dir=str(tmp_path / "out"),
            start_time=2.0,
            interval_ms=1000
        )
        
        results = extractor.extract_from_video(sample_video)
        
        # Should extract frames from 2s onwards
        extracted = [r for r in results if "Extracted" in r]
        assert len(extracted) >= 0
    
    def test_start_time_exceeds_video_duration(self, sample_video, tmp_path):
        """start_time > video duration should extract no frames."""
        import cv2
        cap = cv2.VideoCapture(str(sample_video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        extractor = VideoFrameExtractor(
            input_dir=str(sample_video.parent),
            output_dir=str(tmp_path / "out"),
            start_time=duration + 100.0,
            interval_ms=1000
        )
        
        results = extractor.extract_from_video(sample_video)
        
        extracted = [r for r in results if "Extracted" in r]
        assert len(extracted) == 0
    
    def test_end_time_exceeds_video_duration(self, sample_video, tmp_path):
        """end_time > video duration should use video duration."""
        extractor = VideoFrameExtractor(
            input_dir=str(sample_video.parent),
            output_dir=str(tmp_path / "out"),
            start_time=0.0,
            end_time=10000.0,
            interval_ms=1000
        )
        
        # Should not raise error, just cap at video duration
        results = extractor.extract_from_video(sample_video)
        # Results should contain frames
        assert isinstance(results, list)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_framing.py -v
```

Expected: Import errors or attribute errors (tests fail because feature not implemented)

- [ ] **Step 3: Commit test file**

```bash
git add tests/test_framing.py
git commit -m "test: add failing tests for extract_frames time range feature"
```

---

## Task 2: Update VideoFrameExtractor class to support time range

**Files:**
- Modify: `src/framing/extractor.py`

- [ ] **Step 1: Update `__init__` method signature and validation**

Find the `__init__` method (lines 30-53) and update:

```python
def __init__(
    self,
    input_dir: str,
    output_dir: str,
    interval_ms: int = 1000,
    image_format: str = 'jpg',
    max_frames: Optional[int] = None,
    threads: int = 8,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None
):
    if cv2 is None:
        raise ImportError("opencv-python is required. Install with: pip install opencv-python")
    
    self.input_dir = Path(input_dir)
    self.output_dir = Path(output_dir)
    self.interval_ms = interval_ms
    self.image_format = image_format.lower()
    self.max_frames = max_frames
    self.threads = threads
    self.start_time = start_time if start_time is not None else 0.0
    self.end_time = end_time
    
    # Validate time range parameters
    if self.start_time < 0:
        raise ValueError("start_time cannot be negative")
    if self.end_time is not None and self.end_time < 0:
        raise ValueError("end_time cannot be negative")
    if self.end_time is not None and self.end_time <= self.start_time:
        raise ValueError("end_time must be greater than start_time")
    
    self.output_dir.mkdir(parents=True, exist_ok=True)
    
    self.total_frames_extracted = 0
    self.total_processing_time = 0.0
    self.errors: List[Tuple[Path, str]] = []
```

- [ ] **Step 2: Update `extract_from_video` method to use time range**

Find the `extract_from_video` method (lines 122-162) and update the frame calculation logic:

```python
def extract_from_video(self, video_path: Path) -> List[str]:
    """Extract frames from a single video file."""
    results = []
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        try:
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_ms = (total_frames / fps) * 1000 if fps > 0 else 0
            duration_sec = duration_ms / 1000.0
            
            # Calculate frame range based on time parameters
            effective_start_time = self.start_time
            effective_end_time = self.end_time if self.end_time is not None else duration_sec
            
            # Clamp end_time to video duration
            if effective_end_time > duration_sec:
                effective_end_time = duration_sec
            
            # Skip if start_time exceeds video duration
            if effective_start_time >= duration_sec:
                return results
            
            # Convert time to frame numbers
            start_frame = int(effective_start_time * fps)
            end_frame = int(effective_end_time * fps)
            
            interval_frames = int(self.interval_ms * fps / 1000)
            if interval_frames < 1:
                interval_frames = 1
            
            frames_to_extract = list(range(start_frame, min(end_frame, total_frames), interval_frames))
            
            if self.max_frames is not None:
                frames_to_extract = frames_to_extract[:self.max_frames]
            
            output_path = self.output_dir / video_path.stem
            output_path.mkdir(parents=True, exist_ok=True)
            
            for seq_num, frame_idx in enumerate(frames_to_extract, 1):
                success, message = self.extract_frame(
                    video_path, frame_idx, output_path, self.image_format, seq_num
                )
                results.append(message)
                
        finally:
            cap.release()
            
    except Exception as e:
        error_msg = f"Error processing {video_path.name}: {str(e)}"
        results.append(error_msg)
        self.errors.append((video_path, error_msg))
    
    return results
```

- [ ] **Step 3: Run tests to verify they pass**

```bash
pytest tests/test_framing.py -v
```

Expected: All tests pass

- [ ] **Step 4: Commit changes**

```bash
git add src/framing/extractor.py
git commit -m "feat: add start_time and end_time parameters to VideoFrameExtractor"
```

---

## Task 3: Update CLI script with new arguments

**Files:**
- Modify: `scripts/extract_frames.py`

- [ ] **Step 1: Add new CLI arguments to `parse_args` function**

Add after the `--max_frames` argument (around line 74):

```python
parser.add_argument(
    '--start_time',
    type=float,
    default=0.0,
    help='Start time for extraction in seconds (default: 0)'
)

parser.add_argument(
    '--end_time',
    type=float,
    default=None,
    help='End time for extraction in seconds (default: video end)'
)
```

- [ ] **Step 2: Pass new parameters to VideoFrameExtractor**

Update the `VideoFrameExtractor` instantiation in `main()` (lines 112-119):

```python
extractor = VideoFrameExtractor(
    input_dir=args.input_dir,
    output_dir=args.out_dir,
    interval_ms=args.interval,
    image_format=args.out_image_format,
    max_frames=args.max_frames,
    threads=args.threads,
    start_time=args.start_time,
    end_time=args.end_time
)
```

- [ ] **Step 3: Update output info display**

Update the print statements (around lines 127-131) to include new parameters:

```python
if not args.quiet:
    print(f"\n{'='*60}")
    print("Video Frame Extraction")
    print(f"{'='*60}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.out_dir}")
    print(f"Extraction interval: {args.interval}ms")
    print(f"Time range: {args.start_time}s - {args.end_time if args.end_time else 'video end'}")
    print(f"Output format: {args.out_image_format}")
    print(f"Threads: {args.threads}")
    print(f"Max frames per video: {args.max_frames or 'all'}")
    print(f"{'='*60}\n")
```

- [ ] **Step 4: Update docstring examples**

Update the epilog in `parse_args` (lines 27-33):

```python
epilog='''
Examples:
  %(prog)s --input_dir ./videos --out_dir ./frames
  %(prog)s --input_dir ./videos --out_dir ./frames --interval 500 --threads 4
  %(prog)s --input_dir ./videos --out_dir ./frames --max_frames 100 --out_image_format png
  %(prog)s --input_dir ./videos --out_dir ./frames --start_time 5.0 --end_time 30.0
  %(prog)s --input_dir ./videos --out_dir ./frames --start_time 10.0 --interval 500
        '''
```

- [ ] **Step 5: Commit changes**

```bash
git add scripts/extract_frames.py
git commit -m "feat: add --start_time and --end_time CLI arguments to extract_frames.py"
```

---

## Task 4: Update README.md documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update parameter table for extract_frames**

Find the extract_frames section (around lines 205-214) and update the parameter table:

```markdown
**Parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--input_dir` | Input directory containing video files | Required |
| `--out_dir` | Output directory for extracted images | Required |
| `--interval` | Extraction interval in milliseconds | 1000 (1s) |
| `--start_time` | Start time for extraction in seconds | 0 |
| `--end_time` | End time for extraction in seconds | video end |
| `--out_image_format` | Output format (jpg/png/tif/pdf) | jpg |
| `--threads` | Number of parallel threads | 8 |
| `--max_frames` | Maximum frames per video | All |
```

- [ ] **Step 2: Add usage example with time range**

Add a new example in the usage examples section (around lines 187-203):

```bash
# Extract frames from specific time range (5s to 30s)
python scripts/extract_frames.py \
    --input_dir videos/ \
    --out_dir frames/ \
    --start_time 5.0 \
    --end_time 30.0

# Extract frames starting from 10 seconds to end
python scripts/extract_frames.py \
    --input_dir videos/ \
    --out_dir frames/ \
    --start_time 10.0
```

- [ ] **Step 3: Commit documentation update**

```bash
git add README.md
git commit -m "docs: update README with start_time and end_time parameters"
```

---

## Task 5: Manual integration test

**Files:**
- Run: Manual test with sample video

- [ ] **Step 1: Test basic time range extraction**

```bash
python scripts/extract_frames.py \
    --input_dir data/ \
    --out_dir outputs/test_frames/ \
    --start_time 1.0 \
    --end_time 3.0 \
    --interval 500
```

Expected: Frames extracted from 1s to 3s at 500ms intervals

- [ ] **Step 2: Test start_time only**

```bash
python scripts/extract_frames.py \
    --input_dir data/ \
    --out_dir outputs/test_frames2/ \
    --start_time 2.0 \
    --interval 1000
```

Expected: Frames extracted from 2s to video end at 1s intervals

- [ ] **Step 3: Verify output directory structure**

```bash
ls -la outputs/test_frames/video/
```

Expected: Frame files named `video_01.jpg`, `video_02.jpg`, etc.

- [ ] **Step 4: Clean up test outputs**

```bash
rm -rf outputs/test_frames outputs/test_frames2
```

---

## Task 6: Run full test suite

- [ ] **Step 1: Run all tests**

```bash
pytest tests/ -v
```

Expected: All tests pass

- [ ] **Step 2: Final commit if needed**

```bash
git status
# If any uncommitted changes:
git add -A
git commit -m "fix: ensure all tests pass with time range feature"
```

---

## Summary

**Files Modified:**
- `src/framing/extractor.py` - Add start_time/end_time parameters and frame range calculation
- `scripts/extract_frames.py` - Add CLI arguments and pass to extractor
- `README.md` - Update documentation with new parameters

**Files Created:**
- `tests/test_framing.py` - New test file for framing module

**Key Features:**
1. `--start_time`: Start extraction from specified time (default: 0s)
2. `--end_time`: End extraction at specified time (default: video end)
3. Time values in seconds
4. Validation: start_time >= 0, end_time > start_time
5. Auto-clamp end_time to video duration if exceeded
6. Graceful handling when start_time exceeds video duration