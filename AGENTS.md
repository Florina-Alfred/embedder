Agents guide for embedder repository

Purpose
- This file gives agentic coding agents concise, actionable instructions for building,
  linting, testing, and following repository conventions in /home/user/python/embedder.

Quick pointers
- Repo root: `.` — main entry point is `main.py` (camera + live inference).
- Python: requires Python >= 3.13 (see `pyproject.toml`).
- Main runtime dependencies: `opencv-python`, `timm`, `torch`, `torchvision` (see `pyproject.toml`).
- There is a local `.venv/` in the repo. Do not commit runtime virtualenvs.
- The `features/` directory contains saved .pt tensors used for examples — treat large binary artifacts carefully.

Environment & install
- Create a virtualenv and install:
  - `python -m venv .venv`
  - `source .venv/bin/activate` (Linux/macOS) or `.venv\\Scripts\\activate` (Windows)
  - `python -m pip install --upgrade pip`
  - Install package in editable mode: `python -m pip install -e .`

- If your team uses `uv`, use the project's `pyproject.toml` `tool.uv` index settings to install torch/vision wheels.

Build / developer commands
- Install development/test tools:
  - `python -m pip install -e .[dev]` (if extras are configured) or install tools individually.
- Common quick commands:
  - `black .`
  - `isort --profile=black .`
  - `ruff check .`
  - `ruff fix .` (auto-fix where safe)
  - `python -m pytest -q`

Run (development)
- Interactive demo (requires camera + X/GUI): `python main.py`
  - main.py currently creates the model and opens the camera at import-time. Avoid importing `main.py` in unit tests.
- Headless/CI: avoid running `main.py` in CI. For experiments that require GUI, wrap with `xvfb-run -a python main.py` in Linux CI.

Tests
- Test runner: `pytest` (recommended).
- Install test deps: `python -m pip install pytest pytest-mock`
- Run the whole test suite: `pytest -q`
- Run a single test file: `pytest -q tests/test_file.py`
- Run a single test function/method (node id):
  - `pytest -q tests/test_file.py::test_function`
- Run tests by name fragment (fast selection): `pytest -k "fragment" -q`
- Useful flags: `-q` (quiet), `-s` (show prints), `-x` (stop on first fail), `-k` (expression).
- Force CPU-only for CUDA-sensitive tests: `CUDA_VISIBLE_DEVICES="" pytest -q tests/`.
- For GUI code that uses `cv2.imshow`, run under Xvfb in CI: `xvfb-run -a pytest -q`.

Running a single test quickly (examples)
- Exact node id: `pytest -q tests/test_preprocess.py::test_preprocess_returns_tensor`
- By substring: `pytest -k "preprocess and not slow" -q`
- Debug and show prints for a single test: `pytest -q tests/test_file.py::test_name -s`

Lint / format / static analysis
- Formatting: Black (default line length 88)
  - `black .` or configure with `[tool.black]` in `pyproject.toml`.
- Import ordering: isort with Black profile
  - `isort --profile=black .`
- Fast linter: ruff
  - `ruff check .`
  - `ruff fix .` to auto-apply many fixes.
- Optional: `flake8 .` and `mypy .` if you enable stricter checks.

Repository-specific style & conventions
- General
  - Follow PEP 8 for layout and naming. Run `black` and `isort --profile=black` as part of your edit/test loop.
  - Keep modules import-safe: importing a module must not start camera capture or heavy GPU work. Move side-effectful code into `if __name__ == "__main__":` or dedicated entrypoints.
    - Example: `main.py` currently instantiates `Backbone`, moves it to device, and opens `cv2.VideoCapture(0)` at module level — prefer moving that into a `main()` function.

- Imports
  - Group order: (1) standard library, (2) third-party, (3) local imports. Leave one blank line between groups.
    - Example:
      import os
      import logging

      import numpy as np
      import torch

      from .utils import useful_fn
  - Use absolute imports for package code; avoid relative imports deeper than one level.
  - Use `isort` to keep imports deterministic.

- Formatting
  - Use Black for formatting; prefer default 88-line wrap. Use f-strings for formatting: `f"frame_{i:05d}.pt"`.

- Typing and annotations
  - Add type hints to all public functions and methods. Keep signatures explicit, e.g.:
    def preprocess(frame: np.ndarray) -> torch.Tensor:
  - Use `typing` / `typing_extensions` for complex generics. Consider `dataclasses` or `NamedTuple` for structured return types.
  - If enabling `mypy`, add `# type: ignore[import]` for optional heavy deps in tests or configure `mypy` to ignore missing imports.

- Naming
  - Functions and modules: snake_case (e.g. `preprocess`, `save_features`).
  - Classes: PascalCase / CamelCase (e.g. `Backbone`).
  - Constants: UPPER_SNAKE_CASE (e.g. `SAVE_DIR`).
  - Private attributes/methods: single leading underscore (e.g. `_helper`).

- Error handling
  - Catch specific exceptions (e.g. `except FileNotFoundError as exc:`) — avoid bare `except:`.
  - Re-raise with context when appropriate: `raise RuntimeError("message") from exc`.
  - Use context managers for resources (`with torch.no_grad():`, `with open(...) as f:`) and ensure cleanup in `finally:` if needed.
  - Prefer raising explicit exceptions rather than silent `print()` calls.

- Logging
  - Use the `logging` module and create a module logger: `logger = logging.getLogger(__name__)`.
  - Avoid `print()` in library code—use `logger.debug/info/warning/error/exception`.

- PyTorch / numeric patterns
  - Move models and tensors to device explicitly: `model.to(device)` and `tensor.to(device)`.
  - Use `model.eval()` and `with torch.no_grad():` for inference.
  - Use `.half()` only on CUDA devices. Protect with `if device.type == "cuda": model.half()`.
  - When saving tensors/models, move them to CPU and detach: `tensor.cpu().detach()`.
  - Seed numeric libs in tests for determinism: `torch.manual_seed(...)` and `np.random.seed(...)`.

- CV2 / Images
  - OpenCV uses BGR by default. Convert to RGB for model input: `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`.
  - Normalize arrays to `np.float32` and scale to [0, 1] before converting to tensors.

- Side-effect safety & import-time behavior
  - Avoid running hardware or GUI operations at import time. Examples in this repo:
    - `main.py` creates `Backbone()` and opens `cv2.VideoCapture(0)` when imported. Tests should import only small utility functions (e.g. `preprocess`) or the `Backbone` class without triggering capture.
  - Suggested refactor: move device/model/camera initialization into a `main()` function and keep `if __name__ == "__main__": main()` at file bottom.

Tests to add (recommended)
- `tests/test_preprocess.py`: unit tests for `preprocess` using small numpy arrays; assert dtype, shape, and range.
- `tests/test_backbone.py`: test that `Backbone` forwards shape and dtype correctly on CPU (use `torch.no_grad()` and small random input).
- `tests/test_saving.py`: test save/load of .pt features to verify keys/shapes (use `tmp_path` fixture and small tensors`).

Pre-commit / CI suggestions
- Add a `.pre-commit-config.yaml` with hooks: `black`, `isort`, `ruff` and a lightweight test hook for fast unit tests.
- CI job matrix ideas:
  1) lint/check: run `ruff check .` and `black --check .` + `isort --check-only .`
  2) tests-cpu: `CUDA_VISIBLE_DEVICES="" pytest -q`
  3) optional tests-gpu: run on runners with CUDA available.

Files of note in this repo
- `main.py` — primary demo script (camera capture, embedding extraction). It currently has import-time side effects.
- `pyproject.toml` — project metadata + `tool.uv` index entries for installing PyTorch wheels.
- `features/` — binary .pt artifacts (large files) used for examples.

Cursor / Copilot rules
- No `.cursor` / `.cursorrules` directories found in the repository root.
- No `.github/copilot-instructions.md` file found.

Next steps (suggested)
1) Move `main.py` side-effectful initialization (model construction, `.half()`, camera open) into a `main()` function so tests can safely import helpers.
2) Add minimal `tests/` scaffold with the three recommended tests and verify `pytest -q` passes under CPU.

If you want, I can: (a) add the test scaffold described above, or (b) add a `.pre-commit-config.yaml` and `pyproject.toml` sections for `ruff`/`black`/`isort` defaults.

Reference: follow PEP 8 + Black + isort + ruff for automated enforcement.
