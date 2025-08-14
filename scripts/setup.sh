#!/bin/bash
# Development environment setup script

set -e

echo "🚀 Setting up development environment..."

# Install pre-commit hooks
echo "📋 Installing pre-commit hooks..."
uv run pre-commit install

echo "✅ Development environment setup completed!"
echo ""
echo "Available commands:"
echo "  ./scripts/format.sh    - Format code and fix style issues"
echo "  ./scripts/check.sh     - Check code quality (dry-run)"
echo "  ./scripts/test.sh      - Run tests with quality checks"
echo "  ./scripts/setup.sh     - Set up development environment"
