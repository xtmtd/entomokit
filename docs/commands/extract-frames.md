# entomokit extract-frames
[English](extract-frames.md) | [中文](extract-frames.cn.md)

## Purpose

Extract still frames from video files. `--input-dir` accepts either a directory of
videos or one video file, so a single clip can be processed without arranging a
temporary directory.

## Usage

Minimal invocation with a video directory:

```bash
entomokit extract-frames --input-dir videos/ --out-dir frames/
```

A single video file uses the same arguments:

```bash
entomokit extract-frames --input-dir clip.mp4 --out-dir frames/
```

## Parameters

### `--input-dir`, `-i`

Required. No default. Video directory, or a single video file path. A directory is
scanned recursively for supported videos; a file path processes only that video.

### `--out-dir`, `-o`

Required. No default. Output directory for extracted frames.

### `--out-image-format`

Optional. Default `jpg`. Choices: `jpg`, `png`, `tif`. Image format of the written
frames.

### `--interval`

Optional. Default `1000`. Extraction interval in milliseconds; `1000` writes one
frame per second.

### `--start-time`

Optional. Default `0.0`. Sampling start time in seconds.

### `--end-time`

Optional. No default; sampling continues to the end of each video. Sampling end
time in seconds.

### `--max-frames`

Optional. No default; every sampled frame in the time range is written. Maximum
number of frames extracted per video.

### `--threads`

Optional. Default `8`. Number of worker threads used for frame extraction.

### `--resume`

Optional. Default off (boolean flag without a value). Skip frames already present
in `--out-dir`, continuing a previous run. Because frame numbering is positional,
the output-affecting parameters (`--interval`, `--out-image-format`, `--max-frames`,
`--start-time`, `--end-time`) are recorded in
`out-dir/.entomokit/extract-frames_params.json` and must match the previous run; a
mismatch exits with an error instead of overwriting frames with different content.

### `--overwrite`

Optional. Default off (boolean flag without a value). Delete the contents of
`--out-dir` and start fresh.

### `--verbose`, `-v`

Optional. Default off (boolean flag without a value). Enable verbose logging.

### `--quiet`, `-q`

Optional. Default off (boolean flag without a value). Suppress non-error output and
progress bars.

## Inputs

- `--input-dir` is either a directory or one video file. Directory input is scanned
  recursively.
- Supported video extensions: `mp4`, `mov`, `avi`, `mkv`, `webm`, `flv`, `m4v`,
  `mpeg`, `mpg`, `wmv`, `3gp`, `ts`.
- Frames are sampled every `--interval` milliseconds between `--start-time` and the
  end of the video or `--end-time`.
- Reading video requires the video extra (OpenCV); see the README's installation
  extras mapping.

## Outputs

- Frames are written under `out-dir/<video's input-relative directory>/<video-stem>/`,
  so same-named videos in different subdirectories produce separate frame trees;
  `--resume` checks that mapped frame directory.
- One image file per sampled frame, in the format selected by `--out-image-format`.
- Nothing is written outside `--out-dir`.

## Examples

Extract a five-to-thirty-second window from one video:

```bash
entomokit extract-frames --input-dir video.mp4 --out-dir frames/ \
    --start-time 5.0 --end-time 30.0
```

Sample twice per second as PNG and cap each video at 100 frames:

```bash
entomokit extract-frames --input-dir videos/ --out-dir frames/ \
    --interval 500 --out-image-format png --max-frames 100
```

Continue an interrupted run, keeping the original extraction parameters:

```bash
entomokit extract-frames --input-dir videos/ --out-dir frames/ \
    --interval 500 --out-image-format png --max-frames 100 --resume
```

## Notes

- Recursive discovery, the mirrored output layout and output-directory safety are
  shared rules: [directory policy](../../README.md#directory-policy). Logging and
  version display are also shared:
  [common behaviours](../../README.md#common-behaviours).
- Interruption: a SIGINT handler is installed, but the extraction loop never reads
  the shutdown flag, so the first `Ctrl+C` only sets it and prints a notice and a
  second `Ctrl+C` exits; no partial-result guarantee is documented.
- A non-empty `--out-dir` stops with an error unless `--resume` (continue) or
  `--overwrite` (fresh start) is passed.
- `--verbose` and `--quiet` only change logging volume; neither changes which
  frames are written.

## Version Notes

- `0.7.0`: directory input is scanned recursively and frames mirror each video's
  input-relative directory instead of being flattened under `--out-dir`.
