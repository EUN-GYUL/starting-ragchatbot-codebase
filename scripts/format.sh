#!/bin/bash
# Code formatting script

set -e

echo "🔧 Running code quality checks and formatting..."

# Format with black
echo "📝 Formatting code with Black..."
uv run black backend/

# Sort imports with isort
echo "📦 Sorting imports with isort..."
uv run isort backend/

# Run flake8 for linting
echo "🔍 Running flake8 linting..."
uv run flake8 backend/ --max-line-length=88 --extend-ignore=E203,W503,E501

echo "✅ Code quality checks completed!"
