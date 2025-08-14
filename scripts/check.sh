#!/bin/bash
# Code quality check script (dry-run mode)

set -e

echo "🔍 Running code quality checks (dry-run)..."

# Check formatting with black
echo "📝 Checking Black formatting..."
if ! uv run black backend/ --check --diff; then
    echo "❌ Black formatting issues found. Run 'scripts/format.sh' to fix."
    exit 1
fi

# Check import sorting
echo "📦 Checking import sorting..."
if ! uv run isort backend/ --check-only --diff; then
    echo "❌ Import sorting issues found. Run 'scripts/format.sh' to fix."
    exit 1
fi

# Run flake8 for linting
echo "🔍 Running flake8 linting..."
if ! uv run flake8 backend/ --max-line-length=88 --extend-ignore=E203,W503,E501; then
    echo "❌ Linting issues found. Check output above."
    exit 1
fi

echo "✅ All code quality checks passed!"
