"""Tests for classify embed CLI outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def test_classify_embed_run_writes_metrics_to_out_dir_and_reports_paths(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    from entomokit.classify import embed as embed_cli

    out_dir = tmp_path / "embed_out"
    label_csv = tmp_path / "labels.csv"
    pd.DataFrame({"image": ["a.jpg", "b.jpg"], "label": ["x", "y"]}).to_csv(
        label_csv, index=False
    )

    monkeypatch.setattr("src.common.cli.save_log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "src.classification.utils.select_device",
        lambda _device: type("_D", (), {"type": "cpu"})(),
    )
    monkeypatch.setattr("src.classification.utils.set_num_threads", lambda *_args: None)
    monkeypatch.setattr(
        "src.classification.embedder.extract_embeddings_timm",
        lambda **_kwargs: pd.DataFrame(
            {
                "image": ["a.jpg", "b.jpg"],
                "feat_0": [0.1, 0.2],
                "feat_1": [0.2, 0.1],
            }
        ),
    )
    monkeypatch.setattr(
        "src.classification.embedder.compute_embedding_metrics",
        lambda *_args, **_kwargs: {"NMI": 0.5, "ARI": 0.6},
    )

    umap_calls: list[Path] = []

    def _fake_visualize_umap(*_args, out_path: Path, **_kwargs) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"pdf")
        umap_calls.append(out_path)

    monkeypatch.setattr(
        "src.classification.embedder.visualize_umap", _fake_visualize_umap
    )

    args = argparse.Namespace(
        images_dir=str(tmp_path / "images"),
        out_dir=str(out_dir),
        base_model="convnextv2_femto",
        model_dir=None,
        label_csv=str(label_csv),
        visualize=True,
        umap_n_neighbors=15,
        umap_min_dist=0.1,
        umap_metric="euclidean",
        umap_seed=42,
        batch_size=2,
        num_workers=0,
        num_threads=0,
        device="auto",
        overwrite=False,
        metrics_sample_size=100,
        recursive=False,
    )

    embed_cli.run(args)

    assert (out_dir / "embeddings.csv").exists()
    assert (out_dir / "metrics.csv").exists()
    assert not (out_dir / "logs" / "metrics.csv").exists()
    assert umap_calls == [out_dir / "umap.pdf"]

    out_text = capsys.readouterr().out
    assert f"UMAP saved to: {out_dir / 'umap.pdf'}" in out_text
    assert f"Metrics saved to: {out_dir / 'metrics.csv'}" in out_text
