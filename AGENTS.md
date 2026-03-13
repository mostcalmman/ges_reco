# AGENTS.md

This repository is dedicated to video-based gesture recognition research using the "20BN-JESTER" dataset (or a subset thereof).

## Project Overview

- **Dataset**: Located in `dataset/` and `dataset_cut/`.
- **Primary Script**: `baseline.py` (currently a placeholder for the baseline model implementation).
- **Environment**: Python 3.14+ with `numpy`, `pillow`, `matplotlib`, and standard scientific libraries.

## 🛠 Build & Development Commands

Since this is a research/data-science project, there are no traditional "build" steps.

- **Run Baseline**: `python baseline.py`
- **Install Dependencies**: `pip install numpy pillow matplotlib` (and other libraries like `torch` or `tensorflow` as needed for your specific implementation).
- **Check Environment**: `python --version`

### Testing
There is currently no testing framework (like `pytest`) configured. If you add tests:
- **Run all tests**: `pytest`
- **Run a single test**: `pytest tests/test_file.py::test_name`

## 📏 Code Style Guidelines

Follow standard Python (PEP 8) conventions.

### Imports
- Group imports: standard library, third-party libraries, local modules.
- Use absolute imports where possible.
- Prefer `from module import specific_item` for frequently used items to keep code concise.

### Formatting & Types
- Use **4 spaces** for indentation.
- Use **Type Hints** for all function signatures (e.g., `def process_image(path: str) -> np.ndarray:`).
- Keep lines under **88-100 characters** (consistent with Black/Ruff standards).

### Naming Conventions
- **Files/Modules**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions/Variables**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`

### Error Handling
- Use specific exceptions (e.g., `FileNotFoundError`, `ValueError`) rather than generic `Exception`.
- Use `try...except` blocks for I/O operations and data loading.
- Log errors or use `print()` for debugging in research scripts, but prefer logging for production-ready code.

## 📁 Data Structure

- `dataset/`: Contains the full dataset.
  - `Train.csv`, `Validation.csv`, `Test.csv`: Metadata and labels.
  - `Train/`, `Validation/`, `Test/`: Folders named by `id`, containing JPEG frames (`00001.jpg`, etc.).
- `dataset_cut/`: A subset of the dataset for faster iteration.

## 🤖 Agent Instructions

- **Baseline Implementation**: When implementing `baseline.py`, focus on efficient data loading (using `Pillow` and `numpy`) and model definition.
- **Git Protocol**: Atomic commits with clear messages. Do not commit large data files or CSVs unless explicitly asked.
- **Diagnostics**: Run `lsp_diagnostics` or basic syntax checks before completing a task.
- **Patterns**: Look at the structure of `dataset/` before writing data loaders. Note that `frames` in CSV indicates the number of images in the corresponding directory.
