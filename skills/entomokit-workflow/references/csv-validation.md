# CSV Validation

## Required Schema

- Columns must be exactly `image` and `label` (case-sensitive).
- File format: UTF-8 CSV (not xlsx).

## Checks Before split-csv

0. Label-rule confirmation: user explicitly confirms how labels are derived (folder name, filename first two words, or mapping rule).
1. Header check: both columns present.
2. Null check: no empty values in either column.
3. Path check: if `--images-dir` is used, sample rows must resolve.
4. Class distribution check: flag classes with low sample counts.

## Common Repair Guidance

- Wrong header names -> rename to `image,label`.
- Empty cells -> fill or remove rows.
- Absolute paths -> normalize to relative paths for portability.
- BOM/encoding issues -> re-save as UTF-8 without BOM.
