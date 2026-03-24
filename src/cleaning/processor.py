"""Image cleaning and deduplication utilities."""

import hashlib
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Set, Tuple

try:
    import numpy as np
except ImportError:
    np = None

try:
    from PIL import Image, UnidentifiedImageError
except ImportError:
    Image = None
    UnidentifiedImageError = None

try:
    import imagehash
except ImportError:
    imagehash = None

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None


INVALID_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]+")


def clean_filename(name: str) -> str:
    """Clean filename by removing invalid characters."""
    name = unicodedata.normalize("NFC", name)
    name = INVALID_CHARS_RE.sub("_", name)
    name = name.strip("._")
    if not name:
        name = "untitled"
    return name


def ensure_unique_prefix(base: str, used: Set[str], lock: Lock) -> str:
    """Ensure filename is unique by adding suffix if needed."""
    with lock:
        name = base
        count = 1
        while name.lower() in used:
            name = f"{base}_{count}"
            count += 1
        used.add(name.lower())
        return name


def compute_md5(img: "Image.Image") -> str:
    """Compute MD5 hash of image."""
    if np is None:
        raise ImportError("numpy is required. Install with: pip install numpy")

    buf = img.tobytes()
    return hashlib.md5(buf).hexdigest()


def compute_phash(img: "Image.Image") -> Optional["imagehash.ImageHash"]:
    """Compute perceptual hash of image."""
    if imagehash is None:
        raise ImportError("imagehash is required. Install with: pip install imagehash")

    try:
        return imagehash.phash(img)
    except Exception:
        return None


def phash_to_int(phash_obj) -> Optional[int]:
    """Convert ImageHash object to integer."""
    if phash_obj is None:
        return None

    try:
        hash_array = phash_obj.hash
        if hash_array is None:
            return None

        flat = np.asarray(hash_array).flatten()

        if len(flat) == 0:
            return None

        value = 0
        for bit in flat:
            value = (value << 1) | int(bit)
        return value
    except (AttributeError, TypeError, ValueError):
        return None


def hamming_distance(phash_a: int, phash_b: int) -> int:
    """Compute Hamming distance between two phash integers."""
    xor = phash_a ^ phash_b
    return bin(xor).count("1")


def resize_short_edge(img: "Image.Image", short_size: int) -> "Image.Image":
    """Resize image if short_size > 0; return original if short_size == -1."""
    if short_size == -1:
        return img

    w, h = img.size
    short = min(w, h)
    if short <= short_size:
        return img
    scale = short_size / short
    new_size = (round(w * scale), round(h * scale))
    return img.resize(new_size, Image.Resampling.LANCZOS)


class ImageCleaner:
    """Clean and deduplicate images."""

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        out_short_size: int = 512,
        out_image_format: str = "jpg",
        dedup_mode: str = "md5",
        phash_threshold: int = 5,
        threads: int = 12,
        keep_exif: bool = False,
    ):
        if Image is None:
            raise ImportError("Pillow is required. Install with: pip install Pillow")

        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.out_short_size = out_short_size
        self.out_image_format = out_image_format.lower()
        self.dedup_mode = dedup_mode
        self.phash_threshold = phash_threshold
        self.threads = threads
        self.keep_exif = keep_exif

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.used_names: Set[str] = set()
        self.dedup_hashes_md5: Dict[str, str] = {}
        self.dedup_hashes_phash: List[Tuple[int, str]] = []

        self.name_lock = Lock()
        self.md5_lock = Lock()
        self.phash_lock = Lock()

        self._prepopulate_hashes()

    def _prepopulate_hashes(self) -> None:
        """Pre-populate hashes from existing files in output directory."""
        if not self.output_dir.exists():
            return

        for f in self.output_dir.iterdir():
            if f.is_file():
                self.used_names.add(f.stem.lower())

                try:
                    img = Image.open(f)

                    if self.dedup_mode == "md5":
                        md5 = compute_md5(img)
                        self.dedup_hashes_md5[md5] = f.name

                    elif self.dedup_mode == "phash":
                        ph = compute_phash(img)
                        ph_value = phash_to_int(ph)
                        if ph_value is not None:
                            self.dedup_hashes_phash.append((ph_value, f.name))

                    img.close()
                except Exception:
                    pass

    def process_one(self, src: Path, log_file, log_lock: Lock) -> None:
        """Process a single image file."""
        try:
            suffix = src.suffix.lower()
            ph_value = None

            try:
                img = Image.open(src)
            except UnidentifiedImageError:
                with log_lock:
                    log_file.write(f"Unidentified image: {src}\n")
                    log_file.flush()
                return

            if self.dedup_mode == "md5":
                md5 = compute_md5(img)
                with self.md5_lock:
                    if md5 in self.dedup_hashes_md5:
                        existing_file = self.dedup_hashes_md5[md5]
                        with log_lock:
                            log_file.write(
                                f"Duplicate(md5): {src} == {existing_file} (same MD5: {md5[:8]}...)\n"
                            )
                            log_file.flush()
                        return
                    self.dedup_hashes_md5[md5] = src.name

            elif self.dedup_mode == "phash":
                ph = compute_phash(img)
                ph_value = phash_to_int(ph)

                if ph_value is None:
                    with log_lock:
                        log_file.write(f"Cannot compute valid phash: {src}\n")
                        log_file.flush()
                    return

                with self.phash_lock:
                    for old_value, old_name in self.dedup_hashes_phash:
                        dist = hamming_distance(ph_value, old_value)
                        if dist <= self.phash_threshold:
                            with log_lock:
                                log_file.write(
                                    f"Duplicate(phash): {src} == {old_name} (hamming distance={dist})\n"
                                )
                                log_file.flush()
                            return

            img = resize_short_edge(img, self.out_short_size)

            base = clean_filename(src.stem)
            base = ensure_unique_prefix(base, self.used_names, self.name_lock)
            out_ext = "." + self.out_image_format.lower()
            dst = self.output_dir / f"{base}{out_ext}"

            if out_ext == ".jpg" and img.mode not in ("RGB", "L"):
                img = img.convert("RGB")

            save_params = {}
            if not self.keep_exif:
                save_params["exif"] = b""
            else:
                if "exif" in img.info:
                    save_params["exif"] = img.info["exif"]

            img.save(dst, **save_params)

            if self.dedup_mode == "md5":
                with self.md5_lock:
                    self.dedup_hashes_md5[md5] = dst.name
            elif self.dedup_mode == "phash" and ph_value is not None:
                with self.phash_lock:
                    self.dedup_hashes_phash.append((ph_value, dst.name))

        except Exception as e:
            with log_lock:
                log_file.write(f"Error processing {src}: {e}\n")
                log_file.flush()

    def process_directory(
        self,
        input_dir: Optional[str] = None,
        log_path: Optional[str] = "log.txt",
        recursive: bool = False,
    ) -> dict:
        """Process all images in directory."""
        input_dir = Path(input_dir or self.input_dir)

        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

        IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
        if recursive:
            files = [
                p
                for p in input_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ]
        else:
            files = [
                p
                for p in input_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ]

        if not files:
            return {"total": 0, "processed": 0, "skipped": 0, "errors": 0}

        log_file_path = Path(log_path)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        results = {"total": len(files), "processed": 0, "skipped": 0, "errors": 0}

        log_lock = Lock()

        with log_file_path.open("a", encoding="utf8") as log_file:
            log_file.write(f"Processing {len(files)} files from {input_dir}\n")
            log_file.write(f"Output dir: {self.output_dir}\n")
            log_file.write(f"Dedup mode: {self.dedup_mode}\n")
            log_file.write("-" * 60 + "\n")

            if TQDM_AVAILABLE and tqdm:
                with ThreadPoolExecutor(max_workers=self.threads) as pool:
                    tasks = []
                    for f in files:
                        tasks.append(
                            pool.submit(self.process_one, f, log_file, log_lock)
                        )

                    for task in tqdm(tasks, desc="Processing images"):
                        try:
                            task.result()
                            results["processed"] += 1
                        except Exception:
                            results["errors"] += 1

            else:
                with ThreadPoolExecutor(max_workers=self.threads) as pool:
                    futures = []
                    for f in files:
                        futures.append(
                            pool.submit(self.process_one, f, log_file, log_lock)
                        )

                    for future in futures:
                        try:
                            future.result()
                            results["processed"] += 1
                        except Exception:
                            results["errors"] += 1

        with log_file_path.open("a", encoding="utf8") as log_file:
            log_file.write("-" * 60 + "\n")
            log_file.write(
                f"Results: processed={results['processed']}, errors={results['errors']}\n"
            )

        return results
