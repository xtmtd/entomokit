"""Dataset splitting utilities."""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


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

    def load_data(self) -> None:
        """Load and validate input CSV."""
        if not os.path.exists(self.raw_image_csv):
            raise FileNotFoundError(f"Input file not found: {self.raw_image_csv}")

        self.all_data = pd.read_csv(self.raw_image_csv)
        self.total_samples = len(self.all_data)

        class_counts = self.all_data["label"].value_counts().reset_index()
        class_counts.columns = ["label", "count"]
        self.class_counts = class_counts

        class_counts.to_csv(self.class_count_dir / "class.count", index=False)

        print(f"Loaded {self.total_samples} samples, {len(class_counts)} classes.")

    def split_ratio_mode(
        self,
        unknown_test_classes_ratio: float = 0.0,
        known_test_classes_ratio: float = 0.1,
        val_ratio: float = 0.0,
    ) -> dict:
        """Split dataset using ratio-based mode."""
        if self.all_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        known_data = self.all_data.copy()
        test_unknown_data = pd.DataFrame()
        test_known_data = pd.DataFrame()
        train_data = pd.DataFrame()

        if unknown_test_classes_ratio > 0:
            target_unknown = self.total_samples * unknown_test_classes_ratio
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
            test_unknown_data.label.value_counts().to_csv(
                self.class_count_dir / "class.test.unknown.count", index=False
            )
        else:
            print("Skip unknown ratio split, ratio=0")

        if len(known_data) > 0:
            train_idx = (
                known_data.groupby("label", group_keys=False)
                .sample(frac=1 - known_test_classes_ratio, random_state=self.seed)
                .index
            )
            train_data = known_data.loc[train_idx].reset_index(drop=True)
            test_known_data = known_data.drop(train_idx).reset_index(drop=True)

            train_data.to_csv(self.out_dir / "train.csv", index=False)
            test_known_data.to_csv(self.out_dir / "test.known.csv", index=False)

            train_data.label.value_counts().to_csv(
                self.class_count_dir / "class.train.count", index=False
            )
            test_known_data.label.value_counts().to_csv(
                self.class_count_dir / "class.test.known.count", index=False
            )
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
            train_data.label.value_counts().to_csv(
                self.class_count_dir / "class.train.count", index=False
            )
            val_data.to_csv(self.out_dir / "val.csv", index=False)
            val_data.label.value_counts().to_csv(
                self.class_count_dir / "class.val.count", index=False
            )

        return {
            "test_unknown": len(test_unknown_data),
            "test_known": len(test_known_data),
            "train": len(train_data),
            "val": len(val_data),
        }

    def split_count_mode(
        self,
        unknown_test_classes_count: int = 0,
        known_test_classes_count: int = 0,
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

        if unknown_test_classes_count > 0:
            target_unknown = unknown_test_classes_count
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
            test_unknown_data.label.value_counts().to_csv(
                self.class_count_dir / "class.test.unknown.count", index=False
            )

        if known_test_classes_count > 0:
            target_known = known_test_classes_count
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
            test_known_data.label.value_counts().to_csv(
                self.class_count_dir / "class.test.known.count", index=False
            )

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
            train_data.label.value_counts().to_csv(
                self.class_count_dir / "class.train.count", index=False
            )

        val_data = pd.DataFrame()
        if val_count > 0 and len(train_data) > 0:
            val_rows = []
            accumulated = 0
            for lbl, group in train_data.groupby("label"):
                group = group.sample(frac=1, random_state=self.seed)
                for idx, row in group.iterrows():
                    val_rows.append(row)
                    accumulated += 1
                    if accumulated >= val_count:
                        break
                if accumulated >= val_count:
                    break
            if val_rows:
                val_data = pd.DataFrame(val_rows).reset_index(drop=True)
                train_data = train_data.drop(
                    val_data.index, errors="ignore"
                ).reset_index(drop=True)
                train_data.to_csv(self.out_dir / "train.csv", index=False)
                train_data.label.value_counts().to_csv(
                    self.class_count_dir / "class.train.count", index=False
                )
                val_data.to_csv(self.out_dir / "val.csv", index=False)
                val_data.label.value_counts().to_csv(
                    self.class_count_dir / "class.val.count", index=False
                )

        return {
            "test_unknown": len(test_unknown_data),
            "test_known": len(test_known_data),
            "train": len(train_data),
            "val": len(val_data),
        }

    def _copy_images(self, images_dir: Path, splits: dict) -> None:
        """Copy images into out_dir/images/{split}/ subdirs.

        Args:
            images_dir: Source image directory.
            splits: Dict mapping split name -> DataFrame with 'image' column.
        """
        import shutil

        images_root = self.out_dir / "images"
        for split_name, df in splits.items():
            if df is None or len(df) == 0:
                continue
            split_dir = images_root / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            for img_path in df["image"]:
                src = images_dir / img_path
                dst = split_dir / Path(img_path).name
                if src.exists():
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
                unknown_test_classes_ratio=unknown_test_ratio,
                known_test_classes_ratio=known_test_ratio,
                val_ratio=val_ratio,
            )
        elif mode == "count":
            results = self.split_count_mode(
                unknown_test_classes_count=unknown_test_count,
                known_test_classes_count=known_test_count,
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
