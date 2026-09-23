"""Tests for classify embed CLI outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pytest


def _make_images_dir(tmp_path: Path) -> Path:
    """Create the image directory the CLI now requires before extraction."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    for name in ("a.jpg", "b.jpg"):
        (images_dir / name).write_bytes(b"")
    return images_dir


def _stub_runtime(
    monkeypatch,
    extracted: list | None = None,
    metrics: dict | None = None,
) -> None:
    """Replace device/logging/model/metrics so only the CLI wiring runs."""
    extracted = extracted if extracted is not None else []

    def _fake_extract(**kwargs):
        extracted.append(kwargs)
        return pd.DataFrame(
            {
                "image": ["a.jpg", "b.jpg"],
                "feat_0": [0.1, 0.2],
                "feat_1": [0.2, 0.1],
            }
        )

    monkeypatch.setattr("src.common.cli.save_log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "src.classification.utils.select_device",
        lambda _device: type("_D", (), {"type": "cpu"})(),
    )
    monkeypatch.setattr("src.classification.utils.set_num_threads", lambda *_args: None)
    monkeypatch.setattr(
        "src.classification.embedder.extract_embeddings_timm", _fake_extract
    )
    monkeypatch.setattr(
        "src.classification.embedder.compute_embedding_metrics",
        lambda *_args, **_kwargs: (
            metrics if metrics is not None else {"NMI": 0.5, "ARI": 0.6}
        ),
    )


def _embed_args(tmp_path: Path, out_dir: Path, **kw) -> argparse.Namespace:
    args = dict(
        images_dir=str(tmp_path / "images"),
        out_dir=str(out_dir),
        base_model="convnextv2_femto",
        model_dir=None,
        label_csv=None,
        visualize=False,
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
    )
    args.update(kw)
    return argparse.Namespace(**args)


def test_classify_embed_run_writes_metrics_to_out_dir_and_reports_paths(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    from entomokit.classify import embed as embed_cli

    out_dir = tmp_path / "embed_out"
    images_dir = _make_images_dir(tmp_path)
    label_csv = tmp_path / "labels.csv"
    pd.DataFrame({"image": ["a.jpg", "b.jpg"], "label": ["x", "y"]}).to_csv(
        label_csv, index=False
    )

    _stub_runtime(monkeypatch)

    umap_calls: list[Path] = []

    def _fake_visualize_umap(*_args, out_path: Path, **_kwargs) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"pdf")
        umap_calls.append(out_path)

    monkeypatch.setattr(
        "src.classification.embedder.visualize_umap", _fake_visualize_umap
    )

    args = _embed_args(
        tmp_path,
        out_dir,
        images_dir=str(images_dir),
        label_csv=str(label_csv),
        visualize=True,
    )

    embed_cli.run(args)

    assert (out_dir / "embeddings.csv").exists()
    assert (out_dir / "metrics.csv").exists()
    assert not (out_dir / "logs" / "metrics.csv").exists()
    assert umap_calls == [out_dir / "umap.pdf"]

    out_text = capsys.readouterr().out
    assert f"UMAP saved to: {out_dir / 'umap.pdf'}" in out_text
    assert f"Metrics saved to: {out_dir / 'metrics.csv'}" in out_text


def test_classify_embed_rejects_duplicate_label_images(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from entomokit.classify import embed as embed_cli

    out_dir = tmp_path / "embed_out"
    images_dir = _make_images_dir(tmp_path)
    label_csv = tmp_path / "labels.csv"
    pd.DataFrame(
        {"image": ["a.jpg", "a.jpg", "b.jpg"], "label": ["x", "x", "y"]}
    ).to_csv(label_csv, index=False)
    _stub_runtime(monkeypatch)

    args = _embed_args(
        tmp_path, out_dir, images_dir=str(images_dir), label_csv=str(label_csv)
    )

    with pytest.raises(ValueError, match="--label-csv"):
        embed_cli.run(args)

    assert not (out_dir / "embeddings.csv").exists()


def test_classify_embed_rejects_label_csv_without_matching_images(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from entomokit.classify import embed as embed_cli

    out_dir = tmp_path / "embed_out"
    images_dir = _make_images_dir(tmp_path)
    label_csv = tmp_path / "labels.csv"
    pd.DataFrame({"image": ["zzz.jpg"], "label": ["x"]}).to_csv(
        label_csv, index=False
    )
    extracted: list = []
    _stub_runtime(monkeypatch, extracted=extracted)

    args = _embed_args(
        tmp_path, out_dir, images_dir=str(images_dir), label_csv=str(label_csv)
    )

    with pytest.raises(ValueError, match="matching"):
        embed_cli.run(args)

    assert not out_dir.exists()
    assert extracted == []


def test_classify_embed_ignores_directories_named_like_images(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A directory named ``*.jpg`` must not satisfy the image-name overlap check."""
    from entomokit.classify import embed as embed_cli

    out_dir = tmp_path / "embed_out"
    images_dir = _make_images_dir(tmp_path)
    (images_dir / "dir.jpg").mkdir()
    label_csv = tmp_path / "labels.csv"
    pd.DataFrame({"image": ["dir.jpg"], "label": ["x"]}).to_csv(
        label_csv, index=False
    )
    extracted: list = []
    _stub_runtime(monkeypatch, extracted=extracted)

    args = _embed_args(
        tmp_path, out_dir, images_dir=str(images_dir), label_csv=str(label_csv)
    )

    with pytest.raises(ValueError, match="matching"):
        embed_cli.run(args)

    assert not out_dir.exists()
    assert extracted == []


def test_classify_embed_prints_na_for_unavailable_metrics(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    from entomokit.classify import embed as embed_cli

    out_dir = tmp_path / "embed_out"
    images_dir = _make_images_dir(tmp_path)
    label_csv = tmp_path / "labels.csv"
    pd.DataFrame({"image": ["a.jpg", "b.jpg"], "label": ["x", "y"]}).to_csv(
        label_csv, index=False
    )
    _stub_runtime(
        monkeypatch, metrics={"NMI": 0.5, "Linear_Probing_Balanced_Acc": None}
    )

    args = _embed_args(
        tmp_path, out_dir, images_dir=str(images_dir), label_csv=str(label_csv)
    )

    embed_cli.run(args)

    out_text = capsys.readouterr().out
    assert "Linear_Probing_Balanced_Acc: N/A" in out_text
    written = pd.read_csv(out_dir / "metrics.csv")
    assert pd.isna(written.loc[0, "Linear_Probing_Balanced_Acc"])


def test_extract_embeddings_timm_uses_relative_image_names(tmp_path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")

    import timm

    from src.classification import embedder

    images_dir = tmp_path / "images"
    (images_dir / "beetles").mkdir(parents=True)
    from PIL import Image as _PIL

    _PIL.new("RGB", (8, 8)).save(images_dir / "beetles" / "a.jpg")
    _PIL.new("RGB", (8, 8)).save(images_dir / "top.jpg")

    class _FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._p = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return torch.zeros(x.shape[0], 3)

    monkeypatch.setattr(timm, "create_model", lambda *_a, **_k: _FakeModel())
    monkeypatch.setattr(
        timm.data, "resolve_model_data_config", lambda *_a, **_k: {}
    )
    monkeypatch.setattr(
        timm.data.transforms_factory,
        "create_transform",
        lambda *_a, **_k: (lambda _img: torch.zeros(3, 8, 8)),
    )

    df = embedder.extract_embeddings_timm(
        images_dir=images_dir,
        base_model="fake",
        batch_size=2,
        num_workers=0,
        device=torch.device("cpu"),
    )

    assert sorted(df["image"].tolist()) == ["beetles/a.jpg", "top.jpg"]


def test_classify_embed_rejects_basename_for_nested_image(tmp_path, monkeypatch):
    from entomokit.classify import embed as embed_cli

    out_dir = tmp_path / "embed_out"
    images_dir = tmp_path / "images"
    (images_dir / "beetles").mkdir(parents=True)
    (images_dir / "beetles" / "a.jpg").write_bytes(b"")
    label_csv = tmp_path / "labels.csv"
    pd.DataFrame({"image": ["a.jpg"], "label": ["x"]}).to_csv(label_csv, index=False)
    extracted: list = []
    _stub_runtime(monkeypatch, extracted=extracted)

    args = _embed_args(
        tmp_path, out_dir, images_dir=str(images_dir), label_csv=str(label_csv)
    )

    with pytest.raises(ValueError, match="matching"):
        embed_cli.run(args)
    assert extracted == []


def test_classify_embed_errors_when_merge_matches_no_rows(tmp_path, monkeypatch):
    from entomokit.classify import embed as embed_cli

    out_dir = tmp_path / "embed_out"
    images_dir = tmp_path / "images"
    (images_dir / "beetles").mkdir(parents=True)
    (images_dir / "beetles" / "a.jpg").write_bytes(b"")
    label_csv = tmp_path / "labels.csv"
    pd.DataFrame({"image": ["beetles/a.jpg"], "label": ["x"]}).to_csv(
        label_csv, index=False
    )
    # The stub returns top-level "a.jpg"/"b.jpg" embeddings, which pass the
    # pre-check but cannot merge with the nested label path.
    _stub_runtime(monkeypatch)

    args = _embed_args(
        tmp_path, out_dir, images_dir=str(images_dir), label_csv=str(label_csv)
    )

    with pytest.raises(ValueError, match="match no embeddings"):
        embed_cli.run(args)
    assert (out_dir / "embeddings.csv").exists()
