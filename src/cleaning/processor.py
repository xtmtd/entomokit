"""Image cleaning and deduplication utilities."""

import csv
import hashlib
import json
import logging
import os
import re
import tempfile
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Sequence, Set, Tuple

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


from src.common.files import IMAGE_EXTENSIONS, iter_files

INVALID_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]+")
logger = logging.getLogger(__name__)

FILTER_DECISION_FIELDS = [
    "input_path",
    "status",
    "reason",
    "kept_path",
    "kept_scope",
    "matched_path",
    "matched_scope",
    "distance",
    "group_id",
    "group_size",
    "md5",
    "phash",
    "output_path",
]
RETAINED_FIELDS = ["input_path", "status", "output_path"]
DROPPED_FIELDS = [
    "input_path",
    "reason",
    "matched_path",
    "matched_scope",
    "distance",
    "group_id",
]


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
    """Compute the MD5 of decoded pixel bytes (not the file bytes)."""
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


def _border_median_rgb(img: "Image.Image") -> tuple:
    """Median RGB over the four outermost rows/columns of the image."""
    import statistics

    w, h = img.size
    pixels = img.convert("RGB").load()
    border = []
    for x in range(w):
        border.append(pixels[x, 0])
        if h > 1:
            border.append(pixels[x, h - 1])
    for y in range(1, h - 1):
        border.append(pixels[0, y])
        if w > 1:
            border.append(pixels[w - 1, y])
    return tuple(round(statistics.median(p[c] for p in border)) for c in range(3))


def pad_to_square(img: "Image.Image", color: str = "none") -> "Image.Image":
    """Return image unchanged for none, otherwise center it on a square canvas."""
    if color == "none":
        return img
    if color == "median":
        fill = _border_median_rgb(img)
    elif color == "black":
        fill = (0, 0, 0)
    elif color == "white":
        fill = (255, 255, 255)
    else:
        raise ValueError(f"Unknown pad color: {color}")

    w, h = img.size
    side = max(w, h)

    if img.mode in ("RGBA", "LA", "P"):
        rgba = img.convert("RGBA")
        source = Image.new("RGB", rgba.size, fill)
        source.paste(rgba, mask=rgba.split()[-1])
    else:
        source = img.convert("RGB")

    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(source, ((side - w) // 2, (side - h) // 2))
    return canvas


def split_rank(relative_path: str) -> int:
    """Return held-out priority: test=0, val=1, train=2, none=3."""
    parts = {part.lower() for part in Path(relative_path).parts}
    if "test" in parts:
        return 0
    if parts & {"val", "valid", "validation"}:
        return 1
    if "train" in parts:
        return 2
    return 3


def _format_phash(value: Optional[int]) -> str:
    return "" if value is None else f"{value:016x}"


def _parse_phash(value) -> Optional[int]:
    if value in (None, ""):
        return None
    if isinstance(value, int):
        parsed = value
    else:
        try:
            parsed = int(str(value), 16)
        except (TypeError, ValueError):
            return None
    if parsed < 0 or parsed >= (1 << 64):
        return None
    return parsed


def _valid_md5(value) -> str:
    text = str(value or "").strip().lower()
    if len(text) == 32 and all(char in "0123456789abcdef" for char in text):
        return text
    return ""


def _validated_output(
    report_dir: Path, output_path
) -> Optional[Tuple[str, Path]]:
    """Resolve a manifest output path and reject unsafe/non-image paths."""
    if not output_path:
        return None
    raw = Path(str(output_path))
    if raw.is_absolute():
        return None
    root = report_dir.resolve()
    try:
        resolved = (root / raw).resolve()
        rel_path = resolved.relative_to(root).as_posix()
    except (OSError, ValueError):
        return None
    if not resolved.is_file():
        return None
    if resolved.suffix.lower() not in IMAGE_EXTENSIONS:
        return None
    if Image is not None:
        try:
            with Image.open(resolved) as img:
                img.load()
        except Exception:
            return None
    return rel_path, resolved


@dataclass
class CleanDecision:
    input_path: str
    status: str
    reason: str = ""
    kept_path: str = ""
    kept_scope: str = ""
    matched_path: str = ""
    matched_scope: str = ""
    distance: Optional[int] = None
    group_id: str = ""
    group_size: int = 0
    md5: str = ""
    phash: str = ""
    output_path: str = ""


@dataclass
class ImageRecord:
    path: Path
    rel_path: str
    md5: str = ""
    phash: Optional[int] = None
    error: str = ""


@dataclass
class ExistingOutput:
    rel_path: str
    path: Path
    input_path: str = ""
    source_md5: str = ""
    source_phash: Optional[int] = None
    output_md5: str = ""
    output_phash: Optional[int] = None
    from_index: bool = False


@dataclass
class _Node:
    key: Tuple[str, str]
    rel_path: str
    md5: str = ""
    phash: Optional[int] = None
    is_existing: bool = False
    existing: Optional[ExistingOutput] = None
    record: Optional[ImageRecord] = None


class _UnionFind:
    def __init__(self, nodes: Sequence[_Node]):
        self.parent = {node.key: node.key for node in nodes}
        self.nodes = {node.key: node for node in nodes}

    def find(self, key: Tuple[str, str]) -> Tuple[str, str]:
        parent = self.parent[key]
        while parent != key:
            self.parent[key] = self.parent[parent]
            key = parent
            parent = self.parent[key]
        return key

    def union(self, left: Tuple[str, str], right: Tuple[str, str]) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if root_left < root_right:
            self.parent[root_right] = root_left
        else:
            self.parent[root_left] = root_right

    def groups(self) -> List[List[_Node]]:
        grouped: Dict[Tuple[str, str], List[_Node]] = {}
        for key in self.parent:
            grouped.setdefault(self.find(key), []).append(self.nodes[key])
        return list(grouped.values())


def _atomic_write(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        writer(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict]) -> None:
    def writer(tmp: Path) -> None:
        with tmp.open("w", newline="", encoding="utf-8") as handle:
            out = csv.DictWriter(
                handle,
                fieldnames=list(fieldnames),
                lineterminator="\n",
                extrasaction="ignore",
            )
            out.writeheader()
            for row in rows:
                out.writerow(
                    {
                        key: "" if row.get(key) is None else row.get(key)
                        for key in fieldnames
                    }
                )

    _atomic_write(path, writer)


def _write_json(path: Path, payload: dict) -> None:
    def writer(tmp: Path) -> None:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    _atomic_write(path, writer)


def write_clean_reports(
    decisions: Sequence[CleanDecision],
    report_dir: Path,
    log_path: Optional[Path],
    summary: dict,
    dry_run: bool,
    index_rows: Optional[Sequence[dict]] = None,
) -> dict:
    """Write the clean report artifacts and append the log lines/summary."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(decisions, key=lambda decision: decision.input_path)

    rows = [asdict(decision) for decision in ordered]
    retained_rows = [
        row for row in rows if row["status"] in {"retained", "existing"}
    ]
    dropped_rows = [row for row in rows if row["status"] == "dropped"]

    report_paths = {
        "retained_images": report_dir / "retained_images.csv",
        "dropped_images": report_dir / "dropped_images.csv",
        "filter_decisions": report_dir / "filter_decisions.csv",
        "clean_manifest": report_dir / "clean_manifest.json",
    }

    _write_csv(report_paths["retained_images"], RETAINED_FIELDS, retained_rows)
    _write_csv(report_paths["dropped_images"], DROPPED_FIELDS, dropped_rows)
    _write_csv(report_paths["filter_decisions"], FILTER_DECISION_FIELDS, rows)

    n_input = len(rows)
    n_retained = sum(1 for row in rows if row["status"] == "retained")
    n_dropped = sum(1 for row in rows if row["status"] == "dropped")
    n_existing = sum(1 for row in rows if row["status"] == "existing")
    n_errors = sum(1 for row in rows if row["status"] == "error")

    manifest = {
        "schema_version": 1,
        "command": "clean",
        "dedup_mode": summary.get("dedup_mode", ""),
        "phash_threshold": summary.get("phash_threshold", 0),
        "threads": summary.get("threads", 0),
        "dry_run": bool(dry_run),
        "n_input": n_input,
        "n_retained": n_retained,
        "n_dropped": n_dropped,
        "n_existing": n_existing,
        "n_errors": n_errors,
        "output_files": {
            "retained_images": "retained_images.csv",
            "dropped_images": "dropped_images.csv",
            "filter_decisions": "filter_decisions.csv",
            "cleaned_images": None
            if dry_run
            else summary.get("cleaned_images", "cleaned_images/"),
        },
        "files": rows,
        "index": list(index_rows or []),
    }
    _write_json(report_paths["clean_manifest"], manifest)

    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            for decision in ordered:
                if decision.status == "retained":
                    handle.write(
                        f"retained: {decision.input_path} -> {decision.output_path}\n"
                    )
                elif decision.status == "existing":
                    handle.write(
                        f"existing: {decision.input_path} -> {decision.output_path}\n"
                    )
                elif decision.status == "dropped":
                    distance = (
                        "-" if decision.distance is None else decision.distance
                    )
                    handle.write(
                        f"dropped: {decision.input_path} -> "
                        f"{decision.matched_path} ({decision.reason}, "
                        f"distance={distance})\n"
                    )
                else:
                    handle.write(
                        f"error: {decision.input_path} ({decision.reason})\n"
                    )
            handle.write("-" * 60 + "\n")
            handle.write(
                "Results: input={}, retained={}, dropped={}, existing={}, errors={}\n".format(
                    n_input, n_retained, n_dropped, n_existing, n_errors
                )
            )
            for name, path in report_paths.items():
                handle.write(f"Report {name}: {path}\n")

    return {
        "retained_images": str(report_paths["retained_images"]),
        "dropped_images": str(report_paths["dropped_images"]),
        "filter_decisions": str(report_paths["filter_decisions"]),
        "clean_manifest": str(report_paths["clean_manifest"]),
    }


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
        pad_color: str = "none",
        dry_run: bool = False,
        resume: bool = False,
    ):
        if Image is None:
            raise ImportError("Pillow is required. Install with: pip install Pillow")

        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.out_short_size = out_short_size
        self.out_image_format = out_image_format.lower()
        self.dedup_mode = dedup_mode
        self.phash_threshold = phash_threshold
        self.threads = max(1, int(threads))
        self.keep_exif = keep_exif
        self.pad_color = pad_color
        self.dry_run = dry_run
        self.resume = resume

        if not self.dry_run:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.used_names_by_dir: Dict[Path, Set[str]] = {}
        self.name_lock = Lock()

    def _validated_existing_output(
        self, report_dir: Path, output_path
    ) -> Optional[Tuple[str, Path]]:
        validated = _validated_output(report_dir, output_path)
        if validated is None:
            return None
        rel_path, resolved = validated
        try:
            resolved.relative_to(self.output_dir.resolve())
        except ValueError:
            return None
        return rel_path, resolved

    def _load_existing_outputs(
        self, report_dir: Path
    ) -> Tuple[List[ExistingOutput], List[dict]]:
        existing: List[ExistingOutput] = []
        index_rows: List[dict] = []
        manifest_path = report_dir / "clean_manifest.json"
        manifest_loaded = False
        covered: Set[str] = set()
        seen_index: Set[Tuple[str, str, str, str]] = set()

        if manifest_path.is_file():
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Ignoring unreadable %s: %s", manifest_path, exc)
                payload = None
            if isinstance(payload, dict) and payload.get("schema_version") == 1:
                manifest_loaded = True
                dry_run_manifest = bool(payload.get("dry_run"))
                raw_rows = payload.get("index")
                legacy = raw_rows is None
                if dry_run_manifest and raw_rows is None:
                    # A dry-run manifest may only contribute its copied index.
                    raw_rows = []
                elif raw_rows is None:
                    raw_rows = payload.get("files", [])
                if not isinstance(raw_rows, list):
                    logger.warning(
                        "Ignoring %s: index/files is not a list", manifest_path
                    )
                    raw_rows = []

                allowed = {"retained", "existing"}
                if not legacy:
                    allowed = {"retained", "existing", "dropped"}

                for row in raw_rows:
                    if not isinstance(row, dict):
                        continue
                    status = str(row.get("status") or "")
                    if status not in allowed:
                        continue
                    validated = self._validated_existing_output(
                        report_dir, row.get("output_path")
                    )
                    if validated is None:
                        logger.warning(
                            "Ignoring missing output (or unsafe path) referenced by %s: %s",
                            manifest_path,
                            row.get("output_path"),
                        )
                        continue
                    rel_path, output = validated
                    input_path = str(row.get("input_path") or "")
                    if not input_path:
                        logger.warning(
                            "Ignoring %s index row without input_path: %s",
                            manifest_path,
                            rel_path,
                        )
                        continue
                    md5 = _valid_md5(row.get("md5"))
                    phash = _parse_phash(row.get("phash"))
                    key = (
                        rel_path,
                        input_path,
                        md5,
                        _format_phash(phash),
                    )
                    if key in seen_index:
                        continue
                    seen_index.add(key)
                    covered.add(rel_path)
                    index_rows.append(
                        {
                            "input_path": input_path,
                            "output_path": rel_path,
                            "status": status,
                            "md5": md5,
                            "phash": _format_phash(phash),
                        }
                    )
                    existing.append(
                        ExistingOutput(
                            rel_path=rel_path,
                            path=output,
                            input_path=input_path,
                            source_md5=md5,
                            source_phash=phash,
                            from_index=True,
                        )
                    )
            elif payload is not None:
                logger.warning(
                    "Ignoring %s with unsupported schema_version=%r",
                    manifest_path,
                    payload.get("schema_version")
                    if isinstance(payload, dict)
                    else None,
                )

        if self.output_dir.is_dir():
            root = report_dir.resolve()
            for path in sorted(
                self.output_dir.rglob("*"), key=lambda item: item.as_posix()
            ):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                try:
                    rel_path = path.resolve().relative_to(root).as_posix()
                except ValueError:
                    continue
                if rel_path in covered:
                    continue
                covered.add(rel_path)
                existing.append(
                    ExistingOutput(rel_path=rel_path, path=path.resolve())
                )

        existing.sort(key=lambda item: (item.rel_path, item.input_path))
        index_rows.sort(key=lambda row: (row["input_path"], row["output_path"]))
        index_has_md5 = any(_valid_md5(row.get("md5")) for row in index_rows)
        if (
            existing
            and self.dedup_mode in {"md5", "md5+phash"}
            and (not manifest_loaded or not index_rows or not index_has_md5)
        ):
            logger.warning(
                "--resume found output files but no usable clean_manifest.json "
                "MD5 index; output-hash matching may miss resized/converted "
                "outputs. Use a fresh run or --overwrite when exact matching is "
                "required."
            )
        return existing, index_rows

    def _ensure_output_hashes(
        self, existing: Sequence[ExistingOutput]
    ) -> None:
        """Decode existing outputs once and cache their decoded-pixel hashes."""
        need_md5 = self.dedup_mode in {"md5", "md5+phash"}
        need_phash = self.dedup_mode in {"phash", "md5+phash"}
        if not need_md5 and not need_phash:
            return

        for item in existing:
            if (need_md5 and not item.output_md5) or (
                need_phash and item.output_phash is None
            ):
                try:
                    with Image.open(item.path) as img:
                        img.load()
                        if need_md5 and not item.output_md5:
                            item.output_md5 = compute_md5(img)
                        if need_phash and item.output_phash is None:
                            item.output_phash = phash_to_int(compute_phash(img))
                except Exception as exc:
                    logger.warning(
                        "Cannot hash existing output %s: %s", item.path, exc
                    )

    def _hash_images(
        self, files: Sequence[Path], input_dir: Path
    ) -> List[ImageRecord]:
        def work(src: Path) -> ImageRecord:
            rel_path = src.relative_to(input_dir).as_posix()
            record = ImageRecord(path=src, rel_path=rel_path)
            try:
                with Image.open(src) as img:
                    if self.dedup_mode in {"md5", "md5+phash"}:
                        record.md5 = compute_md5(img)
                    if self.dedup_mode in {"phash", "md5+phash"}:
                        record.phash = phash_to_int(compute_phash(img))
                        if record.phash is None and self.dedup_mode == "phash":
                            record.error = "hash_error:phash"
            except UnidentifiedImageError:
                record.error = "unidentified_image"
            except Exception as exc:
                record.error = f"hash_error:{exc}"
            return record

        with ThreadPoolExecutor(max_workers=self.threads) as pool:
            records = list(pool.map(work, files))
        records.sort(key=lambda record: record.rel_path)
        return records

    def _nearest(
        self, node: _Node, group: Sequence[_Node], rep: _Node
    ) -> Tuple[_Node, str, Optional[int]]:
        same_md5 = [
            member
            for member in group
            if member is not node and node.md5 and member.md5 == node.md5
        ]
        if same_md5:
            match = rep if rep in same_md5 else min(same_md5, key=lambda m: m.rel_path)
            return match, "duplicate_md5", None

        candidates = [
            member
            for member in group
            if member is not node
            and node.phash is not None
            and member.phash is not None
        ]
        if not candidates:
            return rep, "duplicate_phash", None
        match = min(
            candidates,
            key=lambda member: (
                hamming_distance(node.phash, member.phash),
                member.rel_path,
            ),
        )
        return (
            match,
            "duplicate_phash",
            hamming_distance(node.phash, match.phash),
        )

    def _match_existing_output(
        self, record: ImageRecord, existing_outputs: Sequence[ExistingOutput]
    ) -> Tuple[Optional[ExistingOutput], Optional[int]]:
        """Match one input against source index hashes, then output hashes.

        The cumulative index is tried first. Every existing output, including
        indexed ones, is then eligible for decoded-output hash fallback.
        """

        def match(
            items: Sequence[ExistingOutput], *, use_source: bool
        ) -> Tuple[Optional[ExistingOutput], Optional[int]]:
            if self.dedup_mode in {"md5", "md5+phash"} and record.md5:
                md5_matches = []
                for item in items:
                    md5 = item.source_md5 if use_source else item.output_md5
                    if md5 and md5 == record.md5:
                        md5_matches.append(item)
                if md5_matches:
                    return (
                        min(md5_matches, key=lambda item: item.rel_path),
                        None,
                    )

            if (
                self.dedup_mode in {"phash", "md5+phash"}
                and record.phash is not None
            ):
                candidates: List[Tuple[int, str, ExistingOutput]] = []
                for item in items:
                    phash = (
                        item.source_phash
                        if use_source
                        else item.output_phash
                    )
                    if phash is None:
                        continue
                    distance = hamming_distance(record.phash, phash)
                    if distance <= self.phash_threshold:
                        candidates.append((distance, item.rel_path, item))
                if candidates:
                    distance, _rel_path, item = min(
                        candidates, key=lambda entry: (entry[0], entry[1])
                    )
                    return item, distance

            return None, None

        found, distance = match(
            [item for item in existing_outputs if item.from_index],
            use_source=True,
        )
        if found is not None:
            return found, distance

        self._ensure_output_hashes(existing_outputs)
        return match(existing_outputs, use_source=False)

    def _decide_groups(
        self, records: Sequence[ImageRecord], existing_outputs: Sequence[ExistingOutput]
    ) -> List[CleanDecision]:
        nodes: List[_Node] = []
        decisions: List[CleanDecision] = []
        group_decisions: List[Tuple[str, List[CleanDecision]]] = []

        errors = [record for record in records if record.error]
        valid_records = [record for record in records if not record.error]

        matched: Dict[str, List[Tuple[ImageRecord, Optional[int]]]] = {}
        unmatched: List[ImageRecord] = []
        for record in valid_records:
            match, distance = self._match_existing_output(
                record, existing_outputs
            )
            if match is None:
                unmatched.append(record)
            else:
                matched.setdefault(match.rel_path, []).append((record, distance))

        existing_by_rel: Dict[str, List[ExistingOutput]] = {}
        for item in existing_outputs:
            existing_by_rel.setdefault(item.rel_path, []).append(item)

        for rel_path in sorted(matched):
            canonical = min(
                existing_by_rel[rel_path], key=lambda item: item.rel_path
            )
            group_size = 1 + len(matched[rel_path])
            rows: List[CleanDecision] = []
            for record, distance in matched[rel_path]:
                rows.append(
                    CleanDecision(
                        input_path=record.rel_path,
                        status="existing",
                        reason="duplicate_existing_output",
                        kept_path=canonical.rel_path,
                        kept_scope="existing_output",
                        matched_path=canonical.rel_path,
                        matched_scope="existing_output",
                        distance=distance,
                        group_size=group_size,
                        md5=record.md5,
                        phash=_format_phash(record.phash),
                        output_path=canonical.rel_path,
                    )
                )
            decisions.extend(rows)
            group_decisions.append((canonical.rel_path, rows))

        for record in unmatched:
            nodes.append(
                _Node(
                    key=("input", record.rel_path),
                    rel_path=record.rel_path,
                    md5=record.md5,
                    phash=record.phash,
                    record=record,
                )
            )

        union = _UnionFind(nodes)
        if self.dedup_mode in {"md5", "md5+phash"}:
            buckets: Dict[str, List[_Node]] = {}
            for node in nodes:
                if node.md5:
                    buckets.setdefault(node.md5, []).append(node)
            for bucket in buckets.values():
                for node in bucket[1:]:
                    union.union(bucket[0].key, node.key)
            if self.dedup_mode == "md5+phash":
                reps: List[_Node] = []
                for bucket in buckets.values():
                    with_phash = [
                        node for node in bucket if node.phash is not None
                    ]
                    if with_phash:
                        reps.append(
                            min(
                                with_phash,
                                key=lambda node: (
                                    split_rank(node.rel_path),
                                    node.rel_path,
                                ),
                            )
                        )
                for index, left in enumerate(reps):
                    for right in reps[index + 1 :]:
                        if (
                            hamming_distance(left.phash, right.phash)
                            <= self.phash_threshold
                        ):
                            union.union(left.key, right.key)
        elif self.dedup_mode == "phash":
            phash_nodes = [node for node in nodes if node.phash is not None]
            for index, left in enumerate(phash_nodes):
                for right in phash_nodes[index + 1 :]:
                    if (
                        hamming_distance(left.phash, right.phash)
                        <= self.phash_threshold
                    ):
                        union.union(left.key, right.key)

        for group in union.groups():
            inputs = [node for node in group if not node.is_existing]
            if not inputs:
                continue
            group_size = len(group)
            rep = min(
                inputs,
                key=lambda node: (split_rank(node.rel_path), node.rel_path),
            )
            rows = []
            for node in inputs:
                if node is rep:
                    row = CleanDecision(
                        input_path=node.rel_path,
                        status="retained",
                        kept_path=node.rel_path,
                        kept_scope="input",
                        group_size=group_size,
                        md5=node.md5,
                        phash=_format_phash(node.phash),
                    )
                else:
                    match, reason, distance = self._nearest(node, group, rep)
                    row = CleanDecision(
                        input_path=node.rel_path,
                        status="dropped",
                        reason=reason,
                        kept_path=rep.rel_path,
                        kept_scope="input",
                        matched_path=match.rel_path,
                        matched_scope="input",
                        distance=distance,
                        group_size=group_size,
                        md5=node.md5,
                        phash=_format_phash(node.phash),
                    )
                rows.append(row)
            decisions.extend(rows)
            group_decisions.append((rep.rel_path, rows))

        for record in errors:
            decisions.append(
                CleanDecision(
                    input_path=record.rel_path,
                    status="error",
                    reason=record.error,
                    md5=record.md5,
                    phash=_format_phash(record.phash),
                )
            )

        group_decisions.sort(key=lambda item: (split_rank(item[0]), item[0]))
        for index, (_rep_path, rows) in enumerate(group_decisions, start=1):
            group_id = f"g{index:04d}"
            for row in rows:
                row.group_id = group_id

        decisions.sort(key=lambda decision: decision.input_path)
        return decisions

    def _decide_none(
        self, records: Sequence[ImageRecord], existing_outputs: Sequence[ExistingOutput]
    ) -> List[CleanDecision]:
        existing_by_input: Dict[str, ExistingOutput] = {}
        for item in existing_outputs:
            if not item.input_path:
                continue
            current = existing_by_input.get(item.input_path)
            if current is None or item.rel_path < current.rel_path:
                existing_by_input[item.input_path] = item

        groups: List[Tuple[str, List[CleanDecision]]] = []
        decisions: List[CleanDecision] = []
        matched: Dict[str, Tuple[ExistingOutput, List[ImageRecord]]] = {}
        warned_unmatched = False

        for record in records:
            if record.error:
                decisions.append(
                    CleanDecision(
                        input_path=record.rel_path,
                        status="error",
                        reason=record.error,
                    )
                )
                continue
            existing = existing_by_input.get(record.rel_path)
            if existing is not None:
                matched.setdefault(
                    existing.rel_path, (existing, [])
                )[1].append(record)
            else:
                if self.resume and not warned_unmatched:
                    logger.warning(
                        "--resume with --dedup-mode none could not match %s to an "
                        "existing output; it will be written as a new file.",
                        record.rel_path,
                    )
                    warned_unmatched = True
                row = CleanDecision(
                    input_path=record.rel_path,
                    status="retained",
                    kept_path=record.rel_path,
                    kept_scope="input",
                    group_size=1,
                )
                decisions.append(row)
                groups.append((record.rel_path, [row]))

        for existing, group_records in matched.values():
            rows = []
            group_size = 1 + len(group_records)
            for record in group_records:
                row = CleanDecision(
                    input_path=record.rel_path,
                    status="existing",
                    reason="duplicate_existing_output",
                    kept_path=existing.rel_path,
                    kept_scope="existing_output",
                    matched_path=existing.rel_path,
                    matched_scope="existing_output",
                    group_size=group_size,
                    output_path=existing.rel_path,
                )
                rows.append(row)
                decisions.append(row)
            groups.append((existing.rel_path, rows))

        groups.sort(key=lambda item: (split_rank(item[0]), item[0]))
        for index, (_rep_path, rows) in enumerate(groups, start=1):
            group_id = f"g{index:04d}"
            for row in rows:
                row.group_id = group_id
        decisions.sort(key=lambda decision: decision.input_path)
        return decisions

    def _assign_single_output_path(
        self, decision: CleanDecision, report_dir: Path
    ) -> str:
        rel = Path(decision.input_path)
        dst_dir = self.output_dir / rel.parent
        key = dst_dir.resolve()
        used = self.used_names_by_dir.setdefault(key, set())
        base = clean_filename(rel.stem)
        name = ensure_unique_prefix(base, used, self.name_lock)
        dst = dst_dir / f"{name}.{self.out_image_format}"
        return Path(os.path.relpath(dst, report_dir)).as_posix()

    def _assign_output_paths(
        self, decisions: Sequence[CleanDecision], report_dir: Path
    ) -> None:
        for decision in sorted(decisions, key=lambda item: item.input_path):
            if decision.status == "retained":
                decision.output_path = self._assign_single_output_path(
                    decision, report_dir
                )

    def _write_one(
        self,
        decision: CleanDecision,
        input_dir: Path,
        report_dir: Path,
    ) -> Optional[str]:
        src = input_dir / decision.input_path
        dst = report_dir / decision.output_path
        try:
            with Image.open(src) as img:
                img = resize_short_edge(img, self.out_short_size)
                img = pad_to_square(img, self.pad_color)

                if self.dry_run:
                    return None

                dst.parent.mkdir(parents=True, exist_ok=True)
                if self.out_image_format == "jpg" and img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")

                save_params = {}
                if not self.keep_exif:
                    save_params["exif"] = b""
                elif "exif" in img.info:
                    save_params["exif"] = img.info["exif"]

                img.save(dst, **save_params)
            return None
        except Exception as exc:
            if not self.dry_run:
                try:
                    dst.unlink(missing_ok=True)
                except OSError:
                    pass
            return str(exc)

    def _decision_distance(
        self, left: CleanDecision, right: CleanDecision
    ) -> Optional[int]:
        left_phash = _parse_phash(left.phash)
        right_phash = _parse_phash(right.phash)
        if left_phash is None or right_phash is None:
            return None
        return hamming_distance(left_phash, right_phash)

    def _nearest_decision(
        self,
        row: CleanDecision,
        candidates: Sequence[CleanDecision],
        rep: CleanDecision,
    ) -> Tuple[CleanDecision, str, Optional[int]]:
        same_md5 = [
            candidate
            for candidate in candidates
            if candidate is not row
            and row.md5
            and candidate.md5 == row.md5
        ]
        if same_md5:
            match = rep if rep in same_md5 else min(
                same_md5, key=lambda candidate: candidate.input_path
            )
            return match, "duplicate_md5", None

        phash_candidates: List[Tuple[int, str, CleanDecision]] = []
        for candidate in candidates:
            if candidate is row:
                continue
            distance = self._decision_distance(row, candidate)
            if distance is not None:
                phash_candidates.append(
                    (distance, candidate.input_path, candidate)
                )
        if not phash_candidates:
            return rep, "duplicate_phash", None
        distance, _path, match = min(
            phash_candidates, key=lambda entry: (entry[0], entry[1])
        )
        return match, "duplicate_phash", distance

    def _write_retained(
        self,
        decisions: Sequence[CleanDecision],
        input_dir: Path,
        report_dir: Path,
    ) -> None:
        initial = [row for row in decisions if row.status == "retained"]
        errors: Dict[int, Optional[str]] = {}
        if initial:
            with ThreadPoolExecutor(max_workers=self.threads) as pool:
                futures = [
                    pool.submit(self._write_one, row, input_dir, report_dir)
                    for row in initial
                ]
                for row, future in zip(initial, futures):
                    errors[id(row)] = future.result()

        group_members: Dict[str, List[CleanDecision]] = {}
        for row in decisions:
            if row.group_id and row.status != "existing":
                group_members.setdefault(row.group_id, []).append(row)
        for members in group_members.values():
            members.sort(key=lambda row: (split_rank(row.input_path), row.input_path))

        for group_id in sorted(group_members):
            members = group_members[group_id]
            rep = next((row for row in members if row.status == "retained"), None)
            if rep is None:
                continue
            rep_error = errors.get(id(rep))
            if rep_error is None:
                continue

            rep.status = "error"
            rep.reason = f"write_error:{rep_error}"
            rep.output_path = ""

            survivor: Optional[CleanDecision] = None
            for member in members:
                if member is rep or member.status != "dropped":
                    continue
                member.output_path = self._assign_single_output_path(
                    member, report_dir
                )
                error = self._write_one(member, input_dir, report_dir)
                if error:
                    member.status = "error"
                    member.reason = f"write_error:{error}"
                    member.output_path = ""
                    continue
                member.status = "retained"
                member.reason = ""
                member.kept_path = member.input_path
                member.kept_scope = "input"
                member.matched_path = ""
                member.matched_scope = ""
                member.distance = None
                survivor = member
                break

            if survivor is not None:
                valid_members = [survivor] + [
                    member
                    for member in members
                    if member is not survivor and member.status == "dropped"
                ]
                for member in members:
                    if member is survivor or member.status == "existing":
                        continue
                    member.kept_path = survivor.input_path
                    member.kept_scope = "input"
                    if member.status == "error":
                        member.matched_path = ""
                        member.matched_scope = ""
                        member.distance = None
                        continue
                    match, reason, distance = self._nearest_decision(
                        member, valid_members, survivor
                    )
                    member.matched_path = match.input_path
                    member.matched_scope = "input"
                    member.reason = reason
                    member.distance = distance
                continue

            for member in members:
                if member.status == "existing":
                    continue
                if member is rep:
                    member.status = "error"
                    member.reason = f"write_error:{rep_error}"
                else:
                    member.status = "error"
                    member.reason = "group_write_failed"
                member.kept_path = ""
                member.kept_scope = ""
                member.matched_path = ""
                member.matched_scope = ""
                member.distance = None
                member.output_path = ""

    def _merge_index(
        self,
        previous: Sequence[dict],
        decisions: Sequence[CleanDecision],
        report_dir: Path,
    ) -> List[dict]:
        merged: Dict[str, dict] = {}

        for row in previous:
            if not isinstance(row, dict):
                continue
            input_path = str(row.get("input_path") or "")
            if not input_path:
                continue
            validated = self._validated_existing_output(
                report_dir, row.get("output_path")
            )
            if validated is None:
                logger.warning(
                    "Dropping cumulative index row with missing/invalid output: %s",
                    row.get("output_path"),
                )
                continue
            rel_path, _output = validated
            status = str(row.get("status") or "")
            if status not in {"retained", "existing", "dropped"}:
                continue
            merged[input_path] = {
                "input_path": input_path,
                "output_path": rel_path,
                "status": status,
                "md5": _valid_md5(row.get("md5")),
                "phash": _format_phash(_parse_phash(row.get("phash"))),
            }

        current_inputs = {row.input_path for row in decisions}
        for input_path in current_inputs:
            merged.pop(input_path, None)

        group_outputs: Dict[str, str] = {}
        for row in decisions:
            if (
                row.group_id
                and row.status in {"retained", "existing"}
                and row.output_path
            ):
                group_outputs.setdefault(row.group_id, row.output_path)

        for row in decisions:
            if row.status in {"retained", "existing"}:
                output_path = row.output_path
                status = row.status
            elif row.group_id and row.group_id in group_outputs:
                output_path = group_outputs[row.group_id]
                status = "dropped"
            else:
                continue
            validated = self._validated_existing_output(report_dir, output_path)
            if validated is None:
                logger.warning(
                    "Not indexing %s: output is missing or unsafe: %s",
                    row.input_path,
                    output_path,
                )
                continue
            rel_path, _output = validated
            merged[row.input_path] = {
                "input_path": row.input_path,
                "output_path": rel_path,
                "status": status,
                "md5": row.md5,
                "phash": row.phash,
            }

        return [merged[key] for key in sorted(merged)]

    def process_directory(
        self,
        input_dir: Optional[str] = None,
        log_path: Optional[str] = "log.txt",
    ) -> dict:
        """Process all images recursively, mirroring input-relative paths."""
        input_dir = Path(input_dir or self.input_dir)

        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
        if not input_dir.is_dir():
            raise NotADirectoryError(f"Input directory is not a directory: {input_dir}")

        files = iter_files(input_dir, IMAGE_EXTENSIONS)
        report_dir = Path(log_path).parent if log_path else self.output_dir.parent
        report_dir.mkdir(parents=True, exist_ok=True)

        if log_path:
            log_file_path = Path(log_path)
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
            with log_file_path.open("a", encoding="utf-8") as handle:
                handle.write(f"Processing {len(files)} files from {input_dir}\n")
                handle.write(f"Output dir: {self.output_dir}\n")
                handle.write(f"Dedup mode: {self.dedup_mode}\n")
                handle.write("-" * 60 + "\n")

        if self.resume:
            existing_outputs, previous_index = self._load_existing_outputs(
                report_dir
            )
        else:
            existing_outputs, previous_index = [], []
        self.used_names_by_dir = {}
        for item in existing_outputs:
            self.used_names_by_dir.setdefault(item.path.parent.resolve(), set()).add(
                item.path.stem.lower()
            )

        records = self._hash_images(files, input_dir)
        if self.dedup_mode == "none":
            decisions = self._decide_none(records, existing_outputs)
        else:
            decisions = self._decide_groups(records, existing_outputs)

        self._assign_output_paths(decisions, report_dir)
        self._write_retained(decisions, input_dir, report_dir)

        summary = {
            "dedup_mode": self.dedup_mode,
            "phash_threshold": self.phash_threshold,
            "threads": self.threads,
            "cleaned_images": None
            if self.dry_run
            else f"{self.output_dir.name}/",
        }
        index_rows = (
            previous_index
            if self.dry_run
            else self._merge_index(previous_index, decisions, report_dir)
        )
        reports = write_clean_reports(
            decisions,
            report_dir,
            Path(log_path) if log_path else None,
            summary,
            self.dry_run,
            index_rows,
        )

        n_retained = sum(1 for row in decisions if row.status == "retained")
        n_dropped = sum(1 for row in decisions if row.status == "dropped")
        n_existing = sum(1 for row in decisions if row.status == "existing")
        n_errors = sum(1 for row in decisions if row.status == "error")
        return {
            "total": len(files),
            "processed": n_retained,
            "skipped": n_dropped,
            "errors": n_errors,
            "retained": n_retained,
            "dropped": n_dropped,
            "existing": n_existing,
            "reports": reports,
        }
