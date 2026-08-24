"""Video frame extraction utilities."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None


class VideoFrameExtractor:
    """Extracts frames from video files with multithreading support."""

    SUPPORTED_VIDEO_FORMATS = {
        "mp4",
        "mov",
        "avi",
        "mkv",
        "webm",
        "flv",
        "m4v",
        "mpeg",
        "mpg",
        "wmv",
        "3gp",
        "ts",
    }

    SUPPORTED_IMAGE_FORMATS = {"jpg", "jpeg", "png", "tif", "tiff", "pdf"}

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        interval_ms: int = 1000,
        image_format: str = "jpg",
        max_frames: Optional[int] = None,
        threads: int = 8,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        recursive: bool = False,
    ):
        if cv2 is None:
            raise ImportError(
                "opencv-python is required. Install with: pip install opencv-python"
            )

        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.interval_ms = interval_ms
        self.image_format = image_format.lower()
        self.max_frames = max_frames
        self.threads = threads
        self.start_time = start_time if start_time is not None else 0.0
        self.end_time = end_time
        self.recursive = recursive

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

    def get_video_files(self) -> List[Path]:
        """Return video files to process."""
        single_filter = getattr(self, "_single_file_filter", None)
        files = []
        for ext in self.SUPPORTED_VIDEO_FORMATS:
            if self.recursive:
                files.extend(self.input_dir.rglob(f"*.{ext}"))
                files.extend(self.input_dir.rglob(f"*.{ext.upper()}"))
            else:
                files.extend(self.input_dir.glob(f"*.{ext}"))
                files.extend(self.input_dir.glob(f"*.{ext.upper()}"))
        if single_filter:
            files = [f for f in files if f.name == single_filter]
        return sorted(set(files))

    def get_video_duration_ms(self, video_path: Path) -> float:
        """Get video duration in milliseconds."""
        cap = cv2.VideoCapture(str(video_path))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_ms = (frame_count / fps) * 1000 if fps > 0 else 0
            return duration_ms
        finally:
            cap.release()

    def extract_frame(
        self,
        video_path: Path,
        frame_number: int,
        output_path: Path,
        image_format: str,
        seq_num: Optional[int] = None,
    ) -> Tuple[bool, str]:
        """Extract a single frame from video."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()

                if not ret:
                    return (
                        False,
                        f"Failed to read frame {frame_number} from {video_path.name}",
                    )

                if seq_num is not None:
                    output_name = (
                        f"{video_path.stem}_{seq_num:02d}.{image_format.lower()}"
                    )
                else:
                    output_name = (
                        f"{video_path.stem}_{frame_number:02d}.{image_format.lower()}"
                    )

                output_file = output_path / output_name

                if output_file.exists():
                    return True, f"Skipped existing frame: {output_file.name}"

                if image_format.lower() in ("tif", "tiff"):
                    cv2.imwrite(str(output_file), frame)
                else:
                    cv2.imwrite(str(output_file), frame)

                return True, f"Extracted: {output_file.name}"

            finally:
                cap.release()

        except Exception as e:
            return False, f"Error extracting frame from {video_path.name}: {str(e)}"

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

                effective_start_time = self.start_time
                effective_end_time = (
                    self.end_time if self.end_time is not None else duration_sec
                )

                if effective_end_time > duration_sec:
                    effective_end_time = duration_sec

                if effective_start_time >= duration_sec:
                    return results

                start_frame = int(effective_start_time * fps)
                end_frame = int(effective_end_time * fps)

                interval_frames = int(self.interval_ms * fps / 1000)
                if interval_frames < 1:
                    interval_frames = 1

                frames_to_extract = list(
                    range(start_frame, min(end_frame, total_frames), interval_frames)
                )

                if self.max_frames is not None:
                    frames_to_extract = frames_to_extract[: self.max_frames]

                if self.recursive:
                    rel = video_path.parent.relative_to(self.input_dir)
                    output_path = self.output_dir / rel / video_path.stem
                else:
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

    def extract_all(self, show_progress: bool = True) -> dict:
        """Extract frames from all video files using multithreading."""
        video_files = self.get_video_files()

        if not video_files:
            return {
                "total_videos": 0,
                "total_frames": 0,
                "errors": 0,
                "processing_time": 0,
            }

        if TQDM_AVAILABLE and show_progress:
            from tqdm import tqdm
        else:
            tqdm = None

        start_time = __import__("time").time()
        all_results = []

        with ThreadPoolExecutor(max_workers=self.threads) as executor:
            futures = {
                executor.submit(self.extract_from_video, video): video
                for video in video_files
            }

            if TQDM_AVAILABLE and show_progress and tqdm is not None:
                progress = tqdm(
                    total=len(futures), desc="Processing videos", unit="video"
                )
                for future in as_completed(futures):
                    video_file = futures[future]
                    try:
                        results = future.result()
                        all_results.extend(results)
                        progress.update(1)
                        progress.set_postfix({"video": video_file.name})
                    except Exception as e:
                        self.errors.append((video_file, str(e)))
                        progress.update(1)
                progress.close()
            else:
                for future in as_completed(futures):
                    video_file = futures[future]
                    try:
                        results = future.result()
                        all_results.extend(results)
                    except Exception as e:
                        self.errors.append((video_file, str(e)))

        self.total_processing_time = __import__("time").time() - start_time
        self.total_frames_extracted = len(all_results)

        extracted_count = sum(1 for r in all_results if "Extracted" in r)
        skipped_count = sum(1 for r in all_results if "Skipped" in r)
        error_count = len(self.errors)

        return {
            "total_videos": len(video_files),
            "total_frames": extracted_count,
            "skipped_frames": skipped_count,
            "errors": error_count,
            "processing_time": self.total_processing_time,
            "results": all_results,
        }
