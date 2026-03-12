Agents guide for embedder repository

Purpose
- This file gives agentic coding agents concise, actionable instructions for building,
  linting, testing, and following repository conventions in /home/user/python/embedder.

Quick pointers
- Repo root: `.` — main entry point is `main.py` (camera + live inference).
- Python: requires Python >= 3.13 (see `pyproject.toml`).
- No existing test suite detected; add tests under `tests/` following examples below.
- No `.cursor` or `.cursorrules` files found; no Copilot instruction file found under `.github/`.

1) Environment & install
- Create a virtualenv and install: 
  - `python -m venv .venv`
  - `source .venv/bin/activate` (Linux/macOS) or `.venv\\Scripts\\activate` (Windows)
  - `python -m pip install --upgrade pip`
  - Install the package inplace: `python -m pip install -e .`

- Note: this project has a `pyproject.toml` and `uv.lock` — if your team uses `uv`, run
  the equivalent `uv` install/sync commands according to your local tooling. The
  standard `pip install -e .` approach works for development without extra tools.

2) Run (development)
- Run the interactive demo (requires camera and GUI): `python main.py`
- Headless / CI: `main.py` uses OpenCV camera and GUI; avoid running it in headless CI.
  For unit tests, import small functions (e.g. `preprocess`) or refactor camera logic
  behind an interface so tests can run headlessly.

3) Tests
- Test runner: pytest (recommended).
- Install test deps (example): `python -m pip install pytest pytest-mock`.
- Run the whole test suite: `pytest -q`
- Run a single test file: `pytest -q tests/test_file.py`
- Run a single test function/method: `pytest -q tests/test_file.py::test_function`
- Run tests by name fragment: `pytest -k "fragment" -q`
- Useful flags: `-q` (quiet), `-s` (show prints), `-x` (stop on first fail), `-k` (expression).

- GPU-sensitive tests: to force CPU in CI or locally, run with an empty CUDA device list:
  `CUDA_VISIBLE_DEVICES="" pytest -q tests/` — this is useful when CUDA is not available.

4) Lint / format / static analysis
- Formatting (auto): Black. Recommended settings: default Black config (88-char wrap).
  - `black .`
- Import ordering: isort with black profile
  - `isort --profile=black .`
- Combined fast linter: ruff (recommended for modern workflows)
  - `ruff check .`
  - Auto-fix where useful: `ruff fix .`
- Optional linters: flake8 / mypy
  - `flake8 .` (if used)
  - `mypy .` (for type checking, add configuration if you enable it)

5) Running a single test quickly (examples)
- Run one test function exactly:
  - `pytest -q tests/test_preprocess.py::test_preprocess_returns_tensor`
- Run tests by substring match (faster to select many tests):
  - `pytest -k "preprocess and not slow" -q`
- Debug/see prints while running a single test:
  - `pytest -q tests/test_file.py::test_name -s`

6) Repository-specific style & conventions
- General
  - Follow PEP 8 for Python layout and naming; run `black` and `isort --profile=black` as
    part of your edit/test loop.
  - Keep modules small and import-safe: importing a module must not start camera capture
    or heavy GPU work. Move side-effectful code into `if __name__ == "__main__":` or
    explicit entry points.

- Imports
  - Group order: (1) standard library, (2) third-party, (3) local imports. Leave one
    blank line between groups. Example:
    import os
    import logging

    import numpy as np
    import torch

    from .utils import useful_fn
  - Use absolute imports for package code; avoid relative imports deeper than one level.
  - Use `isort` to keep imports deterministic.

- Formatting
  - Use Black for formatting. Prefer default line length (88); if you need a different
    line length, add a `[tool.black]` section in `pyproject.toml`.
  - Use f-strings for formatting: `f"frame_{i:05d}.pt"`.

- Typing and annotations
  - Add type hints to all public functions and methods. Keep signatures explicit:
    def preprocess(frame: np.ndarray) -> torch.Tensor:
  - Use `typing`/`typing_extensions` for complex types. Consider `dataclasses` or
    `NamedTuple` for structured return types.
  - If enabling `mypy`, add `# type: ignore[import]` for optional heavy deps in tests
    or use `mypy` config to ignore missing imports.

- Naming
  - Modules and functions: snake_case (e.g. `preprocess`, `save_features`).
  - Classes: PascalCase / CamelCase (e.g. `Backbone`).
  - Constants: UPPER_SNAKE_CASE (e.g. `SAVE_DIR`), module level config near top.
  - Private attributes/methods: single leading underscore (e.g. `_helper`).

- Error handling
  - Catch specific exceptions (e.g. `except FileNotFoundError as exc:`) — avoid bare
    `except:`. Re-raise with context when appropriate: `raise CustomError(...) from exc`.
  - Use context managers for resources (`with torch.no_grad():`, `with open(...) as f:`)
    and ensure cleanup in `finally:` if needed.
  - Prefer returning errors or raising explicit exceptions vs silent `print` logging.

- Logging
  - Use the `logging` module; create module logger: `logger = logging.getLogger(__name__)`.
  - Avoid `print()` in library code. Use `logger.debug/info/warning/error/exception`.

- PyTorch / numeric patterns
  - Move models and tensors to device explicitly: `model.to(device)` and `tensor.to(device)`.
  - Use `model.eval()` and `with torch.no_grad():` for inference.
  - Use `.half()` only on CUDA devices. Protect with `if device.type == "cuda": model.half()`.
  - When saving tensors/models, move them to CPU and detach: `tensor.cpu().detach()`.
  - Seeds for deterministic behavior in tests: `torch.manual_seed(...)` and `np.random.seed(...)`.

- CV2 / Images
  - OpenCV uses BGR by default. Convert to RGB for model input: `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`.
  - Normalize arrays to `np.float32` and scale to [0, 1] before converting to tensors.

7) Tests to add (recommended)
- `tests/test_preprocess.py`: unit tests for `preprocess` using small numpy arrays.
- `tests/test_backbone.py`: test that Backbone forwards shape and dtype correctly on CPU.
- `tests/test_saving.py`: test that saved .pt files contain expected keys/shapes (use tmpdir).

8) Pre-commit / CI suggestions
- Add `pre-commit` with hooks: `black`, `isort`, `ruff` and a test-runner hook that runs
  a targeted set of fast unit tests. Example `.pre-commit-config.yaml` is recommended.

9) Files of note discovered while scanning the repo
- `main.py` — primary demo script (camera capture, embedding extraction).
- `pyproject.toml` — project metadata + `tool.uv` index entries.
- No `.cursor`/`.cursorrules` and no `.github/copilot-instructions.md` were found.

If you want, next steps:
1) I can add a minimal `tests/` scaffold with 2-3 pytest files (preprocess + backbone CPU tests).
2) I can add a `pyproject.toml` section for `black`, `isort`, and `ruff` defaults and a
   `.pre-commit-config.yaml` to install hooks.

Reference: follow PEP 8 + Black + isort/ruff for automated enforcement.
