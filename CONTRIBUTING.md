# Contributing

Thank you for your interest in contributing to this educational course.

---

## How to contribute

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/your-feature`
3. Make your changes.
4. Run tests: `pytest tests`
5. Submit a pull request with a clear description of what you changed and why.

---

## Guidelines

- Keep code simple and educational. Clarity is more important than performance.
- Add docstrings to every public function and class.
- Add comments when tensor shapes change in a non-obvious way.
- Follow the existing code style (black, line length 100).
- If adding a new notebook, follow the section structure in COURSE_OVERVIEW.md.
- If adding a new model, add a corresponding test in `tests/`.
- Do not add pretrained models, Hugging Face dependencies, or timm.

---

## Running tests

```bash
pytest tests -v
```

---

## Code style

Format with black:
```bash
pip install black
black src/ scripts/ tests/
```

---

## Reporting bugs

Open an issue with:
- Your operating system
- Python version
- PyTorch version
- The full error message
- Steps to reproduce
