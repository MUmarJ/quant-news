# Claude Code Settings for Quant News Tracker

## Terminal Environment

**IMPORTANT**: Always use zsh terminal with conda environment activated:

```bash
conda activate quant-news
```

- Shell: `zsh`
- Conda environment: `quant-news`
- Activate before running any Python commands, tests, or the app

---

## Project Documentation

**Main Project Specification**: See [PROJECT.md](../PROJECT.md) for complete project documentation including:
- Architecture and directory structure
- Technology stack and rationale
- Technical indicators (MACD, RSI, Bollinger Bands, etc.)
- Implementation phases
- Code standards and best practices

## Code Standards Reference

All code in this project must follow:

1. **DRY Principles** - No duplication, modular structure
2. **PEP 8** - Python style guide compliance
3. **Type Hints** - All functions must have type annotations
4. **Docstrings** - Google style for all public functions/classes
5. **Black Formatter** - Run before commits (88 char line length)
6. **Constants** - Single source of truth in `config.py`

## Pre-Commit Workflow

```bash
black .      # Format code
isort .      # Sort imports
flake8 .     # Check style
mypy .       # Check types
pytest       # Run tests
```

## Key Files

| File | Purpose |
|------|---------|
| `PROJECT.md` | Full project specification |
| `config.py` | All constants (SINGLE SOURCE OF TRUTH) |
| `requirements.txt` | Production dependencies |
| `requirements-dev.txt` | Development dependencies |
| `pyproject.toml` | Tool configuration (black, isort, mypy) |
