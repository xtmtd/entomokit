# Completion And Evaluate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace runtime argcomplete shell injection with static completion commands, and extend `entomokit classify evaluate` to emit per-class diagnostic artifacts alongside overall metrics.

**Architecture:** Keep the existing `argparse` CLI and add one focused `entomokit/completion.py` module for static Bash/Zsh/Fish script generation plus user-level install helpers. Extend `src/classification/evaluator.py` to produce a single structured evaluation result that the CLI writes to CSV/PDF outputs for both AutoGluon and ONNX paths.

**Tech Stack:** Python standard library, argparse, pandas, numpy, scikit-learn, matplotlib

## Global Constraints

- Do not migrate the CLI to `Typer` or `Click`.
- Do not keep `register-python-argcomplete` in shell startup files.
- Prefer standard library only for completion; do not preserve `argcomplete` unless implementation proves unavoidable.
- Preserve existing `evaluations.csv` output and terminal metric summary.
- Add `confusion_matrix.csv`, `confusion_matrix_normalized.csv`, `per_class_metrics.csv`, and `confusion_matrix.pdf`.
- Generate `confusion_matrix.pdf` only when `num_classes <= 50`; otherwise skip it and print a short reason.
- Use one shared class order across all per-class outputs.
- Prefer model/sidecar class order when available; otherwise use stable sorted `y_true ∪ y_pred` order.

---

## File Structure

### Modify

- `entomokit/main.py` — remove argcomplete activation/install path, register the new `completion` command group, keep the temporary `--install-completion` flag routed through static install logic.
- `entomokit/classify/evaluate.py` — switch from plain metric dict handling to structured evaluation artifact writing.
- `src/classification/evaluator.py` — compute shared evaluation outputs for both backends and expose helpers to write CSV/PDF artifacts.
- `tests/test_main_cli.py` — replace argcomplete-specific assertions with static completion command coverage.
- `tests/test_classify_evaluate_cli.py` — cover new evaluation artifact files, class ordering, and PDF skip behavior.

### Create

- `entomokit/completion.py` — static script generation and per-shell install helpers.

---

### Task 1: Static Completion Command And Install Flow

**Files:**
- Create: `entomokit/completion.py`
- Modify: `entomokit/main.py`
- Test: `tests/test_main_cli.py`

**Interfaces:**
- Consumes: `argparse._SubParsersAction` from `entomokit.main._build_parser()`.
- Produces: `entomokit.completion.register(subparsers: argparse._SubParsersAction) -> None`
- Produces: `entomokit.completion.install_for_shell(shell: str | None = None) -> int`
- Produces: `entomokit.completion.render_completion_script(shell: str) -> str`

- [ ] **Step 1: Write the failing completion tests**

Add these tests to `tests/test_main_cli.py`:

```python
def test_completion_group_help_lists_shells(capsys: pytest.CaptureFixture[str]) -> None:
    from entomokit.main import main

    with pytest.raises(SystemExit) as exc:
        main(["completion", "--help"])

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "bash" in out
    assert "zsh" in out
    assert "fish" in out


def test_completion_zsh_outputs_static_script(capsys: pytest.CaptureFixture[str]) -> None:
    from entomokit.main import main

    main(["completion", "zsh"])
    out = capsys.readouterr().out
    assert "entomokit" in out
    assert "compdef" in out or "complete" in out


def test_install_completion_routes_to_static_installer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from entomokit import main as cli_main

    called = {"value": False}

    def fake_install() -> int:
        called["value"] = True
        return 0

    monkeypatch.setattr(cli_main, "_install_completion", fake_install)
    with pytest.raises(SystemExit) as exc:
        cli_main.main(["--install-completion"])

    assert exc.value.code == 0
    assert called["value"] is True
```

- [ ] **Step 2: Run the completion tests to see them fail**

Run: `pytest tests/test_main_cli.py::test_completion_group_help_lists_shells tests/test_main_cli.py::test_completion_zsh_outputs_static_script tests/test_main_cli.py::test_install_completion_routes_to_static_installer -v`

Expected: FAIL because `completion` is not a registered command and the static script path does not exist yet.

- [ ] **Step 3: Implement minimal static completion module**

Create `entomokit/completion.py` with this shape:

```python
"""Static shell completion helpers for entomokit."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def render_completion_script(shell: str) -> str:
    if shell == "bash":
        return (
            "_entomokit_completion() {\n"
            "    COMPREPLY=( $( compgen -W 'extract-frames segment measure synthesize clean augment split-csv classify doctor completion' -- \"${COMP_WORDS[COMP_CWORD]}\" ) )\n"
            "}\n"
            "complete -F _entomokit_completion entomokit\n"
        )
    if shell == "zsh":
        return (
            "#compdef entomokit\n"
            "_entomokit() {\n"
            "  local -a commands\n"
            "  commands=(extract-frames segment measure synthesize clean augment split-csv classify doctor completion)\n"
            "  _describe 'command' commands\n"
            "}\n"
            "compdef _entomokit entomokit\n"
        )
    if shell == "fish":
        return (
            "complete -c entomokit -f\n"
            "complete -c entomokit -n '__fish_use_subcommand' -a 'extract-frames segment measure synthesize clean augment split-csv classify doctor completion'\n"
        )
    raise ValueError(f"Unsupported shell: {shell}")


def detect_shell() -> str:
    shell = Path(os.environ.get("SHELL", "")).name
    return shell if shell in {"bash", "zsh", "fish"} else "bash"


def install_for_shell(shell: str | None = None) -> int:
    shell = shell or detect_shell()
    script = render_completion_script(shell)
    if shell == "zsh":
        target = Path.home() / ".zfunc" / "_entomokit"
    elif shell == "fish":
        target = Path.home() / ".config" / "fish" / "completions" / "entomokit.fish"
    else:
        target = Path.home() / ".local" / "share" / "bash-completion" / "completions" / "entomokit"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(script, encoding="utf-8")
    print(f"Installed {shell} completion at: {target}")
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("completion", help="Generate static shell completion scripts.")
    shell_subparsers = parser.add_subparsers(dest="shell", required=True)
    for shell_name in ("bash", "zsh", "fish"):
        shell_parser = shell_subparsers.add_parser(shell_name, help=f"Print a {shell_name} completion script.")
        shell_parser.add_argument("--install", action="store_true", help="Write the script to the default user location.")
        shell_parser.set_defaults(func=run, completion_shell=shell_name)


def run(args: argparse.Namespace) -> None:
    if args.install:
        raise SystemExit(install_for_shell(args.completion_shell))
    sys.stdout.write(render_completion_script(args.completion_shell))
```

- [ ] **Step 4: Wire the new command into `entomokit/main.py`**

Update `entomokit/main.py` to remove argcomplete-only code and register the new command:

```python
from entomokit import completion as _completion


def _install_completion() -> int:
    from entomokit.completion import install_for_shell

    return install_for_shell()


def _build_parser() -> argparse.ArgumentParser:
    ...
    _register_classify(subparsers)
    _doctor.register(subparsers)
    _completion.register(subparsers)
    return parser


def main(argv: list[str] | None = None) -> None:
    _ensure_project_root_on_path()
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.install_completion:
        raise SystemExit(_install_completion())
    ...
```

Delete these functions entirely:

```python
def _completion_snippet(shell: str) -> str: ...
def _activate_argcomplete(parser: argparse.ArgumentParser) -> None: ...
```

- [ ] **Step 5: Run the completion tests again**

Run: `pytest tests/test_main_cli.py::test_completion_group_help_lists_shells tests/test_main_cli.py::test_completion_zsh_outputs_static_script tests/test_main_cli.py::test_install_completion_routes_to_static_installer -v`

Expected: PASS.

- [ ] **Step 6: Run the broader CLI regression slice**

Run: `pytest tests/test_main_cli.py -v`

Expected: PASS and no test references to `argcomplete` remain.
Also verify `entomokit completion bash --install`, `entomokit completion zsh --install`, and `entomokit completion fish --install` each target their own shell-specific install path rather than the current `$SHELL` path.

- [ ] **Step 7: Commit**

```bash
git add entomokit/completion.py entomokit/main.py tests/test_main_cli.py
git commit -m "feat(cli): add static shell completion command"
```

### Task 2: Shared Evaluation Artifact Computation

**Files:**
- Modify: `src/classification/evaluator.py`
- Test: `tests/test_classify_evaluate_cli.py`

**Interfaces:**
- Consumes: `labels`, `predictions`, `proba`, `class_labels`
- Produces: `build_evaluation_result(labels, predictions, proba=None, class_labels=None) -> dict[str, object]`
- Produces: `write_evaluation_outputs(result: dict[str, object], out_dir: Path, pdf_class_limit: int = 50) -> dict[str, Path]`
- Produces: `evaluate(...) -> dict[str, object]`
- Produces: `evaluate_onnx(...) -> dict[str, object]`

- [ ] **Step 1: Write failing tests for per-class outputs and class ordering**

Add these tests to `tests/test_classify_evaluate_cli.py`:

```python
def test_build_evaluation_result_returns_confusion_and_per_class_tables() -> None:
    from src.classification.evaluator import build_evaluation_result

    result = build_evaluation_result(
        labels=["b", "a", "a", "b"],
        predictions=["b", "b", "a", "a"],
        class_labels=["b", "a"],
    )

    confusion = result["confusion_matrix"]
    per_class = result["per_class_metrics"]
    assert list(confusion.index) == ["b", "a"]
    assert list(confusion.columns) == ["b", "a"]
    assert list(per_class["label"]) == ["b", "a"]


def test_write_evaluation_outputs_skips_pdf_when_class_count_exceeds_limit(tmp_path) -> None:
    from src.classification.evaluator import build_evaluation_result, write_evaluation_outputs

    labels = [f"cls_{i}" for i in range(51)]
    result = build_evaluation_result(labels=labels, predictions=labels)

    paths = write_evaluation_outputs(result, tmp_path, pdf_class_limit=50)

    assert (tmp_path / "confusion_matrix.csv").exists()
    assert (tmp_path / "confusion_matrix_normalized.csv").exists()
    assert (tmp_path / "per_class_metrics.csv").exists()
    assert "confusion_matrix_pdf" not in paths
    assert not (tmp_path / "confusion_matrix.pdf").exists()


def test_evaluate_onnx_preserves_sidecar_label_order(tmp_path, monkeypatch) -> None:
    from src.classification.evaluator import evaluate_onnx

    test_csv = tmp_path / "test.csv"
    pd.DataFrame({"image": ["a.jpg", "b.jpg"], "label": ["beta", "alpha"]}).to_csv(test_csv, index=False)
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")
    (tmp_path / "label_classes.json").write_text(json.dumps({"class_labels": ["beta", "alpha"]}), encoding="utf-8")

    monkeypatch.setattr(
        "src.classification.predictor.predict_onnx",
        lambda *_args, **_kwargs: pd.DataFrame(
            {
                "image": ["a.jpg", "b.jpg"],
                "prediction": ["beta", "alpha"],
                "prediction_index": [0, 1],
                "proba_0": [0.9, 0.1],
                "proba_1": [0.1, 0.9],
            }
        ),
    )

    result = evaluate_onnx(test_csv=test_csv, images_dir=tmp_path, onnx_path=onnx_path, batch_size=2, num_threads=0)

    assert list(result["per_class_metrics"]["label"]) == ["beta", "alpha"]
```

- [ ] **Step 2: Run the new evaluator tests to verify failure**

Run: `pytest tests/test_classify_evaluate_cli.py::test_build_evaluation_result_returns_confusion_and_per_class_tables tests/test_classify_evaluate_cli.py::test_write_evaluation_outputs_skips_pdf_when_class_count_exceeds_limit tests/test_classify_evaluate_cli.py::test_evaluate_onnx_preserves_sidecar_label_order -v`

Expected: FAIL because the helper functions and structured return value do not exist yet.

- [ ] **Step 3: Implement shared evaluation result building in `src/classification/evaluator.py`**

Add these helpers near the top of the file:

```python
def _resolve_class_labels(labels, predictions, class_labels=None) -> list:
    if class_labels is not None:
        return list(class_labels)
    return sorted(set(labels) | set(predictions))


def build_evaluation_result(labels, predictions, proba=None, class_labels=None) -> dict[str, object]:
    from sklearn.metrics import classification_report, confusion_matrix

    ordered_labels = _resolve_class_labels(labels, predictions, class_labels=class_labels)
    metrics = compute_classification_metrics(labels=labels, predictions=predictions, proba=proba)

    confusion = confusion_matrix(labels, predictions, labels=ordered_labels)
    confusion_df = pd.DataFrame(confusion, index=ordered_labels, columns=ordered_labels)
    normalized = confusion.astype(float)
    row_sums = normalized.sum(axis=1, keepdims=True)
    normalized = np.divide(normalized, row_sums, out=np.zeros_like(normalized), where=row_sums != 0)
    normalized_df = pd.DataFrame(normalized, index=ordered_labels, columns=ordered_labels)

    report = classification_report(
        labels,
        predictions,
        labels=ordered_labels,
        output_dict=True,
        zero_division=0,
    )
    per_class_df = pd.DataFrame(
        [
            {
                "label": label,
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1-score": report[label]["f1-score"],
                "support": report[label]["support"],
            }
            for label in ordered_labels
        ]
    )

    return {
        "metrics": metrics,
        "class_labels": ordered_labels,
        "confusion_matrix": confusion_df,
        "confusion_matrix_normalized": normalized_df,
        "per_class_metrics": per_class_df,
    }
```

- [ ] **Step 4: Implement artifact writing and switch backend returns**

Extend `src/classification/evaluator.py` with the file-writing helper and update both backend functions:

```python
def write_evaluation_outputs(result: dict[str, object], out_dir: Path, pdf_class_limit: int = 50) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(
        [{"metric": name, "value": value} for name, value in result["metrics"].items()]
    )
    metrics_csv = out_dir / "evaluations.csv"
    confusion_csv = out_dir / "confusion_matrix.csv"
    normalized_csv = out_dir / "confusion_matrix_normalized.csv"
    per_class_csv = out_dir / "per_class_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    result["confusion_matrix"].to_csv(confusion_csv, index_label="label")
    result["confusion_matrix_normalized"].to_csv(normalized_csv, index_label="label")
    result["per_class_metrics"].to_csv(per_class_csv, index=False)

    written = {
        "evaluations_csv": metrics_csv,
        "confusion_matrix_csv": confusion_csv,
        "confusion_matrix_normalized_csv": normalized_csv,
        "per_class_metrics_csv": per_class_csv,
    }

    class_count = len(result["class_labels"])
    if class_count <= pdf_class_limit:
        fig, ax = plt.subplots(figsize=(max(6, class_count * 0.4), max(5, class_count * 0.4)))
        ax.imshow(result["confusion_matrix_normalized"].to_numpy(), cmap="Blues", aspect="auto")
        ax.set_xticks(range(class_count), result["class_labels"], rotation=90)
        ax.set_yticks(range(class_count), result["class_labels"])
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        fig.tight_layout()
        pdf_path = out_dir / "confusion_matrix.pdf"
        fig.savefig(pdf_path)
        plt.close(fig)
        written["confusion_matrix_pdf"] = pdf_path

    return written


def evaluate(...):
    ...
    return build_evaluation_result(labels=labels, predictions=predictions, proba=proba)


def evaluate_onnx(...):
    ...
    resolved_class_labels = class_labels if class_labels is not None else None
    return build_evaluation_result(
        labels=labels,
        predictions=predictions,
        proba=proba,
        class_labels=resolved_class_labels,
    )
```

- [ ] **Step 5: Run the evaluator tests again**

Run: `pytest tests/test_classify_evaluate_cli.py::test_build_evaluation_result_returns_confusion_and_per_class_tables tests/test_classify_evaluate_cli.py::test_write_evaluation_outputs_skips_pdf_when_class_count_exceeds_limit tests/test_classify_evaluate_cli.py::test_evaluate_onnx_preserves_sidecar_label_order -v`

Expected: PASS.

- [ ] **Step 6: Run the full evaluate test file**

Run: `pytest tests/test_classify_evaluate_cli.py -v`

Expected: PASS, including existing metric tests.

- [ ] **Step 7: Commit**

```bash
git add src/classification/evaluator.py tests/test_classify_evaluate_cli.py
git commit -m "feat(classify): add confusion matrix evaluation outputs"
```

### Task 3: CLI Integration And Output Regression

**Files:**
- Modify: `entomokit/classify/evaluate.py`
- Modify: `tests/test_classify_evaluate_cli.py`

**Interfaces:**
- Consumes: `src.classification.evaluator.evaluate(...) -> dict[str, object]`
- Consumes: `src.classification.evaluator.evaluate_onnx(...) -> dict[str, object]`
- Consumes: `src.classification.evaluator.write_evaluation_outputs(result, out_dir, pdf_class_limit=50) -> dict[str, Path]`
- Produces: CLI output files in `out_dir/`

- [ ] **Step 1: Write the failing CLI integration test**

Replace the current simple CSV-only assertion in `tests/test_classify_evaluate_cli.py` with:

```python
def test_classify_evaluate_run_writes_all_outputs(tmp_path, monkeypatch) -> None:
    from entomokit.classify import evaluate as evaluate_cli
    from types import SimpleNamespace

    out_dir = tmp_path / "eval_out"
    monkeypatch.setattr("src.common.cli.save_log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("src.classification.utils.select_device", lambda _device: SimpleNamespace(type="cpu"))
    monkeypatch.setattr("src.classification.utils.ag_device_map", lambda _d: "cpu")
    monkeypatch.setattr(
        "src.classification.evaluator.evaluate",
        lambda **_kwargs: {
            "metrics": {"accuracy": 0.9, "balanced_accuracy": 0.88, "f1_weighted": 0.91},
            "class_labels": ["a", "b"],
            "confusion_matrix": pd.DataFrame([[2, 1], [0, 3]], index=["a", "b"], columns=["a", "b"]),
            "confusion_matrix_normalized": pd.DataFrame([[2 / 3, 1 / 3], [0.0, 1.0]], index=["a", "b"], columns=["a", "b"]),
            "per_class_metrics": pd.DataFrame(
                [
                    {"label": "a", "precision": 1.0, "recall": 2 / 3, "f1-score": 0.8, "support": 3},
                    {"label": "b", "precision": 0.75, "recall": 1.0, "f1-score": 0.857143, "support": 3},
                ]
            ),
        },
    )

    args = argparse.Namespace(
        test_csv="test.csv",
        images_dir="images",
        model_dir="model_dir",
        onnx_model=None,
        out_dir=str(out_dir),
        batch_size=32,
        num_workers=2,
        num_threads=0,
        device="auto",
    )

    evaluate_cli.run(args)

    assert (out_dir / "evaluations.csv").exists()
    assert (out_dir / "confusion_matrix.csv").exists()
    assert (out_dir / "confusion_matrix_normalized.csv").exists()
    assert (out_dir / "per_class_metrics.csv").exists()
```

- [ ] **Step 2: Run the CLI integration test to verify failure**

Run: `pytest tests/test_classify_evaluate_cli.py::test_classify_evaluate_run_writes_all_outputs -v`

Expected: FAIL because `entomokit/classify/evaluate.py` still assumes a plain metric dict.

- [ ] **Step 3: Update the CLI writer to use the shared artifact helper**

Replace the write block in `entomokit/classify/evaluate.py` with:

```python
from src.classification.evaluator import write_evaluation_outputs


def run(args: argparse.Namespace) -> None:
    import pandas as pd
    from src.classification.utils import select_device, ag_device_map
    from src.common.cli import save_log

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_log(out_dir, args)
    device = select_device(args.device)

    if args.model_dir:
        from src.classification.evaluator import evaluate

        result = evaluate(
            test_csv=Path(args.test_csv),
            images_dir=Path(args.images_dir),
            model_dir=Path(args.model_dir),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_threads=args.num_threads,
            device=ag_device_map(device),
        )
    else:
        from src.classification.evaluator import evaluate_onnx

        result = evaluate_onnx(
            test_csv=Path(args.test_csv),
            images_dir=Path(args.images_dir),
            onnx_path=Path(args.onnx_model),
            batch_size=args.batch_size,
            num_threads=args.num_threads,
        )

    written = write_evaluation_outputs(result, out_dir, pdf_class_limit=50)
    for metric_name, metric_value in result["metrics"].items():
        print(f"{metric_name}: {metric_value:.6f}")
    if "confusion_matrix_pdf" not in written:
        print("Skipped confusion_matrix.pdf because class count exceeds 50.")
    print(f"\nResults saved to: {written['evaluations_csv']}")
```

- [ ] **Step 4: Run the CLI integration test again**

Run: `pytest tests/test_classify_evaluate_cli.py::test_classify_evaluate_run_writes_all_outputs -v`

Expected: PASS.

- [ ] **Step 5: Run the combined regression slice**

Run: `pytest tests/test_main_cli.py tests/test_classify_evaluate_cli.py -v`

Expected: PASS.

- [ ] **Step 6: Run one full targeted command set before claiming done**

Run: `pytest tests/test_main_cli.py tests/test_classify_evaluate_cli.py tests/test_cli_schema.py -v`

Expected: PASS and no schema/help regressions from adding `completion`.

- [ ] **Step 7: Commit**

```bash
git add entomokit/classify/evaluate.py tests/test_classify_evaluate_cli.py
git commit -m "feat(classify): wire evaluation artifact outputs"
```

---

## Self-Review

- Spec coverage: completion command, legacy `--install-completion` compatibility, artifact outputs, class ordering, and PDF threshold each map to Tasks 1-3.
- Placeholder scan: no `TODO`/`TBD` markers remain; each task has explicit files, commands, and code.
- Type consistency: all later tasks use the same `result["metrics"]`, `result["class_labels"]`, `result["confusion_matrix"]`, `result["confusion_matrix_normalized"]`, and `result["per_class_metrics"]` keys defined in Task 2.

Plan complete and saved to `docs/superpowers/plans/2026-07-05-completion-evaluate-implementation.md`. Two execution options:

1. Subagent-Driven (recommended) - I dispatch a fresh subagent per task, review between tasks, fast iteration
2. Inline Execution - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
