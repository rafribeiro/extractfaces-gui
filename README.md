# Extract Faces

A Python desktop app built with Qt for extracting PNG face crops from videos
using [forensicface](https://github.com/rafribeiro/forensicface).
Play a video, draw a rectangular search area, choose an output folder, and
configure the detector and extraction settings. The app also displays video
dimensions, frame rate, duration, and media codecs.

## Run locally

Requires Python 3.13 or newer. From this project folder:

```powershell
uv sync
uv run extract-faces
```

Alternatively, install with `pip install .`, then run `extract-faces` or
`python -m extract_faces`. Once published to PyPI, installation will be
`pip install extractfaces-gui`.

## Face detector setup

The app uses forensicface's SCRFD detector with optional attributes and
embeddings disabled. Place its `det_10g.onnx` model at:

```text
~/.forensicface/models/detection/scrfd/det_10g.onnx
```

On Windows, `~` means your user folder, such as `C:\Users\rafael`.
See [forensicface's model documentation](https://github.com/rafribeiro/forensicface#layout-de-modelos-por-tarefa-e-alias)
for the layout and model credits. The app does not download model weights.
CPU processing is the default; enable GPU only with a compatible GPU/runtime.
Forensicface installs CUDA dependencies even when using CPU, so installation
can be a large download. ONNX Runtime is constrained to the 1.25 series on
all platforms because newer CUDA extras currently lack compatible Windows wheels.

## Use the app

1. Open a video with the file picker or drop a video file onto the app (the first
   local file is opened if several are dropped). Playback is silent. Metadata includes dimensions, frame rate,
   duration, and codecs when available.
2. Click or drag the playback timeline to seek; Left and Right seek by 10 seconds.
   Pause or drag a rectangle on the preview. Detection searches that area in
   every processed frame. Clear the region to search the full frame.
3. Choose an empty output folder, crop size multiplier, sampling interval, and
   start time. A multiplier of 2 doubles the detected box's dimensions.
   **Process every Nth frame**: 1 processes every frame; 5 processes one in five.
   Enable **Stop at** to specify an optional stop time in seconds, later than
   the start time. Frames at or after the stop time are excluded. Leave it off
   to process to the end. Both times have a **Use current position** button.
   Set **Detector size (det_size)** and **Detection threshold (det_thresh)**
   before extraction to configure SCRFD. Defaults are 320 and 0.5. Larger
   detector sizes take longer; lower thresholds may accept more false positives.
4. Click **Extract faces**. Progress counts sampled frames. Cancel takes effect
   after the current detector operation; crops already saved are kept.

Click **Show console output** to expand the console pane. It captures Python
standard output and errors, including forensicface initialization, progress,
and extraction tracebacks, even while collapsed. The most recent 2,000 lines
are kept. Native-library output written directly to the operating system's
console is not captured. Dropping another video is disabled during extraction.

Extraction reads the original video. The rectangle restricts detection;
added crop margins may extend beyond the rectangle, within the full frame.
Closing during extraction requests cancellation; close again after it stops.

## Code layout

All UI and extraction code is in `src/extract_faces/app.py`:

- `VideoView` paints frames and handles the rectangle in image coordinates.
- `ExtractionWorker` runs forensicface in a background thread. A small adapter
  crops the detector input and restores coordinates before forensicface saves
  crops from the original frame.
- `MainWindow` creates the controls and connects their actions.

`__init__.py` is the installed command's entry point; `__main__.py` supports
`python -m extract_faces`. No separate services or application layers.

Run the tests with `uv run python -m unittest discover -s tests -v`.
They exercise the GUI without a visible window and use a fake detector with
real video reading and crop saving, so model weights are unnecessary.
