# Contributing

Thanks for your interest in improving Gesture-Controlled Volume! Contributions
of all kinds are welcome — bug reports, feature ideas, documentation fixes, and
code.

## Getting started

1. **Fork** the repository and clone your fork.
2. Make sure you have [uv](https://docs.astral.sh/uv/) and **Python 3.12**
   (MediaPipe currently has no wheels for 3.13/3.14).
3. Set up the project:

   ```bash
   uv sync
   curl -L -o hand_landmarker.task \
     https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
   ```

4. Run it to confirm your setup works:

   ```bash
   uv run main.py
   ```

   On first run, macOS will ask for camera permission for whichever app launched
   the script (Terminal, iTerm, VS Code, …). Approve it under **System Settings →
   Privacy & Security → Camera**.

## Making changes

- Keep the code style consistent with what's already there: standard library
  first, small focused functions, and comments only where the intent isn't
  obvious from the code.
- This is currently a single-file app (`main.py`). If a change grows beyond that,
  open an issue first so we can agree on structure.
- Tuning constants (pinch range, smoothing, throttle) live at the top of
  `main.py`. If you change defaults, explain why in the PR.

## Submitting a pull request

1. Create a branch: `git checkout -b my-change`.
2. Make your change and test it on a real webcam.
3. Write a clear commit message describing **what** changed and **why**.
4. Push and open a pull request against `main`. Fill in the PR template,
   including how you tested.

## Reporting bugs

Open an issue using the **Bug report** template. Please include:

- macOS version and Mac model (Apple Silicon vs Intel).
- Python version (`uv run python --version`).
- The full error output.
- What you expected to happen.

## Platform note

Volume control is implemented with macOS `osascript` and so is macOS-only today.
A cross-platform volume backend (Windows `pycaw`, Linux `pactl`/`amixer`) would
be a very welcome contribution — `set_system_volume()` in `main.py` is the single
place to extend.

## Code of conduct

By participating, you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md).
