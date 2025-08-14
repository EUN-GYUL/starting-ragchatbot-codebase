#!/bin/bash
# Test running script with quality checks

set -e

echo "🧪 Running comprehensive test suite..."

# Run code quality checks first
echo "🔍 Running code quality checks..."
./scripts/check.sh

# Run tests
echo "🧪 Running pytest..."
cd backend && uv run pytest tests/ -v

echo "✅ All tests and quality checks passed!"
