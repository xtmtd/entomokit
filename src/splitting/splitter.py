"""Dataset splitting utilities."""

import os
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class CopyImageDestinationCollisionError(ValueError):
    """Raised when distinct sources resolve to the same copied image path."""


def _collision_relative_path(image_path: str) -> Path:
    """Map a collision-group path beneath a split output directory."""
    source_path = Path(image_path)
    parts = [
        part
        for part in source_path.parts
        if part not in {source_path.anchor, ".", ".."} and part
    ]
    return Path(*parts) if parts else Path(source_path.name)


class DatasetSplitter:
    """Split datasets into train/test/unknown classes."""

    def __init__(self, raw_image_csv: str, out_dir: str = "datasets", seed: int = 42):
        self.raw_image_csv = raw_image_csv
        self.out_dir = Path(out_dir)
        self.seed = seed

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.class_count_dir = self.out_dir / "class_count"
        self.class_count_dir.mkdir(exist_ok=True)

        self.all_data: Optional[pd.DataFrame] = None
        self.class_counts: Optional[pd.DataFrame] = None
        self.total_samples = 0

    def _write_class_count(self, data: pd.DataFrame, filename: str) -> None:
        """Write class count CSV with explicit label/count columns."""
        counts = (
            data["label"].value_counts().rename_axis("label").reset_index(name="count")
        )
        counts.to_csv(self.class_count_dir / filename, index=False)

    def load_data(self) -> None:
        """Load and validate input CSV."""
        if not os.path.exists(self.raw_image_csv):
            raise FileNotFoundError(f"Input file not found: {self.raw_image_csv}")

        self.all_data = pd.read_csv(self.raw_image_csv)
        self.total_samples = len(self.all_data)

        class_counts = self.all_data["label"].value_counts().reset_index()
        class_counts.columns = ["label", "count"]
        self.class_counts = class_counts

        self._write_class_count(self.all_data, "class.count")

        print(f"Loaded {self.total_samples} samples, {len(class_counts)} classes.")

    def split_ratio_mode(
        self,
        unknown_test_sample_ratio: float = 0.0,
        known_test_sample_ratio: float = 0.1,
        val_ratio: float = 0.0,
    ) -> dict:
        """Split dataset using ratio-based mode."""
        if self.all_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        known_data = self.all_data.copy()
        test_unknown_data = pd.DataFrame()
        test_known_data = pd.DataFrame()
        train_data = pd.DataFrame()

        if unknown_test_sample_ratio > 0:
            target_unknown = self.total_samples * unknown_test_sample_ratio
            shuffled_classes = self.class_counts.sample(frac=1, random_state=self.seed)
            accumulated = 0
            unknown_labels = []

            for _, row in shuffled_classes.iterrows():
                unknown_labels.append(row["label"])
                accumulated += row["count"]
                if accumulated >= target_unknown:
                    break

            test_unknown_data = self.all_data[
                self.all_data.label.isin(unknown_labels)
            ].reset_index(drop=True)
            known_data = self.all_data[
                ~self.all_data.label.isin(unknown_labels)
            ].reset_index(drop=True)

            test_unknown_data.to_csv(self.out_dir / "test.unknown.csv", index=False)
            self._write_class_count(test_unknown_data, "class.test.unknown.count")
        else:
            print("Skip unknown ratio split, ratio=0")

        if len(known_data) > 0:
            train_idx = (
                known_data.groupby("label", group_keys=False)
                .sample(frac=1 - known_test_sample_ratio, random_state=self.seed)
                .index
            )
            train_data = known_data.loc[train_idx].reset_index(drop=True)
            test_known_data = known_data.drop(train_idx).reset_index(drop=True)

            train_data.to_csv(self.out_dir / "train.csv", index=False)
            test_known_data.to_csv(self.out_dir / "test.known.csv", index=False)

            self._write_class_count(train_data, "class.train.count")
            self._write_class_count(test_known_data, "class.test.known.count")
        else:
            print("No known data after unknown split.")

        val_data = pd.DataFrame()
        if val_ratio > 0 and len(train_data) > 0:
            val_idx = (
                train_data.groupby("label", group_keys=False)
                .sample(frac=val_ratio, random_state=self.seed)
                .index
            )
            val_data = train_data.loc[val_idx].reset_index(drop=True)
            train_data = train_data.drop(val_idx).reset_index(drop=True)
            train_data.to_csv(self.out_dir / "train.csv", index=False)
            self._write_class_count(train_data, "class.train.count")
            val_data.to_csv(self.out_dir / "val.csv", index=False)
            self._write_class_count(val_data, "class.val.count")

        return {
            "test_unknown": len(test_unknown_data),
            "test_known": len(test_known_data),
            "train": len(train_data),
            "val": len(val_data),
        }

    def split_count_mode(
        self,
        unknown_test_sample_count: int = 0,
        known_test_sample_count: int = 0,
        min_count_per_class: int = 0,
        max_count_per_class: Optional[int] = None,
        val_count: int = 0,
    ) -> dict:
        """Split dataset using count-based mode."""
        if self.all_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        df = self.all_data.copy()
        np.random.seed(self.seed)

        test_unknown_data = pd.DataFrame()
        test_known_data = pd.DataFrame()
        train_data = pd.DataFrame()

        if unknown_test_sample_count > 0:
            target_unknown = unknown_test_sample_count
            shuffled_classes = self.class_counts.sample(frac=1, random_state=self.seed)
            accumulated = 0
            unknown_labels = []

            for _, row in shuffled_classes.iterrows():
                unknown_labels.append(row["label"])
                accumulated += row["count"]
                if accumulated >= target_unknown:
                    break

            test_unknown_data = df[df.label.isin(unknown_labels)].reset_index(drop=True)
            df = df[~df.label.isin(unknown_labels)].reset_index(drop=True)

            test_unknown_data.to_csv(self.out_dir / "test.unknown.csv", index=False)
            self._write_class_count(test_unknown_data, "class.test.unknown.count")

        if known_test_sample_count > 0:
            target_known = known_test_sample_count
            known_test_samples = []
            accumulated = 0

            for lbl, group in df.groupby("label"):
                group = group.sample(frac=1, random_state=self.seed)
                for idx, row in group.iterrows():
                    known_test_samples.append(row)
                    accumulated += 1
                    if accumulated >= target_known:
                        break
                if accumulated >= target_known:
                    break

            test_known_data = pd.DataFrame(known_test_samples)
            remain_df = df.drop(test_known_data.index, errors="ignore").reset_index(
                drop=True
            )
        else:
            test_known_data = pd.DataFrame()
            remain_df = df.copy()

        if len(test_known_data) > 0:
            test_known_data.to_csv(self.out_dir / "test.known.csv", index=False)
            self._write_class_count(test_known_data, "class.test.known.count")

        train_rows = []
        for lbl, group in remain_df.groupby("label"):
            n = len(group)
            if n < min_count_per_class:
                continue
            if max_count_per_class is not None:
                take = min(n, max_count_per_class)
            else:
                take = n
            sampled = group.sample(n=take, random_state=self.seed)
            train_rows.append(sampled)

        if len(train_rows) > 0:
            train_data = pd.concat(train_rows).reset_index(drop=True)
            train_data.to_csv(self.out_dir / "train.csv", index=False)
            self._write_class_count(train_data, "class.train.count")

        val_data = pd.DataFrame()
        if val_count > 0 and len(train_data) > 0:
            val_rows = []
            val_indices = []
            accumulated = 0
            for lbl, group in train_data.groupby("label"):
                group = group.sample(frac=1, random_state=self.seed)
                for idx, row in group.iterrows():
                    val_rows.append(row)
                    val_indices.append(idx)
                    accumulated += 1
                    if accumulated >= val_count:
                        break
                if accumulated >= val_count:
                    break
            if val_rows:
                val_data = pd.DataFrame(val_rows).reset_index(drop=True)
                train_data = train_data.drop(val_indices, errors="ignore").reset_index(
                    drop=True
                )
                train_data.to_csv(self.out_dir / "train.csv", index=False)
                self._write_class_count(train_data, "class.train.count")
                val_data.to_csv(self.out_dir / "val.csv", index=False)
                self._write_class_count(val_data, "class.val.count")

        return {
            "test_unknown": len(test_unknown_data),
            "test_known": len(test_known_data),
            "train": len(train_data),
            "val": len(val_data),
        }

    def _copy_images(self, images_dir: Path, splits: dict) -> None:
        """Copy images, retaining parents only for colliding basenames."""
        images_root = self.out_dir / "images"
        colliding_by_split: dict[str, set[str]] = {}

        # Pass 1 counts distinct source paths per basename without storing every row.
        for split_name, df in splits.items():
            if df is None or len(df) == 0:
                continue
            basename_sources: dict[str, tuple[str, str | None]] = {}
            for img_path in df["image"]:
                src = images_dir / img_path
                if not src.exists():
                    continue
                source_value = Path(img_path)
                basename_key = os.path.normcase(source_value.name)
                source_path = str(img_path)
                previous = basename_sources.get(basename_key)
                if previous is None:
                    basename_sources[basename_key] = (source_path, None)
                elif source_path != previous[0] and source_path != previous[1]:
                    basename_sources[basename_key] = (previous[0], source_path)
            colliding_by_split[split_name] = {
                basename
                for basename, (_first, second) in basename_sources.items()
                if second is not None
            }

        # Pass 2 validates every destination before the first copy is written.
        for split_name, df in splits.items():
            if df is None or len(df) == 0:
                continue
            split_dir = images_root / split_name
            colliding_basenames = colliding_by_split[split_name]
            destinations: dict[str, str] = {}
            for img_path in df["image"]:
                src = images_dir / img_path
                if not src.exists():
                    continue
                source_value = Path(img_path)
                basename_key = os.path.normcase(source_value.name)
                target_relative = (
                    _collision_relative_path(str(img_path))
                    if basename_key in colliding_basenames
                    else Path(source_value.name)
                )
                dst = split_dir / target_relative
                key = os.path.normcase(os.path.normpath(str(dst.resolve())))
                previous = destinations.get(key)
                if previous is not None and previous != str(img_path):
                    raise CopyImageDestinationCollisionError(
                        f"{previous} and {img_path} resolve to the same copied "
                        f"image path {dst} in split {split_name}"
                    )
                destinations[key] = str(img_path)

            ordered_destinations = sorted(destinations.items())
            for index, (key, source) in enumerate(ordered_destinations[:-1]):
                next_key, next_source = ordered_destinations[index + 1]
                if next_key.startswith(key + os.sep):
                    raise CopyImageDestinationCollisionError(
                        f"{source} and {next_source} have conflicting copied image "
                        f"paths {key} and {next_key} in split {split_name}"
                    )

        # Pass 3 copies directly from the DataFrames; no per-image plan is cached.
        created_dirs: set[Path] = set()
        for split_name, df in splits.items():
            if df is None or len(df) == 0:
                continue
            split_dir = images_root / split_name
            colliding_basenames = colliding_by_split[split_name]
            for img_path in df["image"]:
                src = images_dir / img_path
                if not src.exists():
                    continue
                source_value = Path(img_path)
                basename_key = os.path.normcase(source_value.name)
                target_relative = (
                    _collision_relative_path(str(img_path))
                    if basename_key in colliding_basenames
                    else Path(source_value.name)
                )
                dst = split_dir / target_relative
                if dst.parent not in created_dirs:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    created_dirs.add(dst.parent)
                shutil.copy2(src, dst)

    def split(
        self,
        mode: str = "ratio",
        unknown_test_ratio: float = 0.0,
        known_test_ratio: float = 0.1,
        unknown_test_count: int = 0,
        known_test_count: int = 0,
        min_count_per_class: int = 0,
        max_count_per_class: Optional[int] = None,
        val_ratio: float = 0.0,
        val_count: int = 0,
        copy_images: bool = False,
        images_dir: Optional[Path] = None,
    ) -> dict:
        """Split dataset into train/test sets.

        Args:
            mode: Split mode - 'ratio' or 'count'
            unknown_test_ratio: Ratio of samples for unknown test classes
            known_test_ratio: Ratio of known class samples for test
            unknown_test_count: Number of samples for unknown test classes
            known_test_count: Number of samples for known test classes
            min_count_per_class: Minimum samples per class for train
            max_count_per_class: Maximum samples per class for train
            val_ratio: Val split ratio (from train). 0 = no val split.
            val_count: Val split count (from train). 0 = no val split.
            copy_images: If True, copy images into out_dir/images/{split}/ subdirs.
            images_dir: Source image directory (required when copy_images=True).

        Returns:
            Dictionary with split statistics
        """
        self.load_data()

        if mode == "ratio":
            results = self.split_ratio_mode(
                unknown_test_sample_ratio=unknown_test_ratio,
                known_test_sample_ratio=known_test_ratio,
                val_ratio=val_ratio,
            )
        elif mode == "count":
            results = self.split_count_mode(
                unknown_test_sample_count=unknown_test_count,
                known_test_sample_count=known_test_count,
                min_count_per_class=min_count_per_class,
                max_count_per_class=max_count_per_class,
                val_count=val_count,
            )
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'ratio' or 'count'.")

        if copy_images and images_dir is not None:
            # Build splits dict from saved CSVs
            splits = {}
            for split_name, csv_name in [
                ("train", "train.csv"),
                ("val", "val.csv"),
                ("test_known", "test.known.csv"),
                ("test_unknown", "test.unknown.csv"),
            ]:
                csv_file = self.out_dir / csv_name
                if csv_file.exists():
                    splits[split_name] = pd.read_csv(csv_file)
            self._copy_images(images_dir, splits)

        return results
