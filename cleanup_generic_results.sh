#!/bin/bash
# Script to delete grammar and readability result files that don't have a model name
# This removes generic files like grammar_bea_dev.csv, grammar_jfleg_test.csv, etc.

RESULTS_DIR="results"

if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: $RESULTS_DIR directory not found"
    exit 1
fi

cd "$RESULTS_DIR"

echo "Deleting generic result files (without model names)..."

# Delete grammar files without model names
rm -f grammar_bea_dev.csv
rm -f grammar_bea_dev_metrics.json
rm -f grammar_jfleg_test.csv
rm -f grammar_jfleg_test_metrics.json

# Delete readability files without model names
rm -f readability_asset_test.csv
rm -f readability_asset_test_metrics.json
rm -f readability_asset_validation.csv
rm -f readability_asset_validation_metrics.json

echo "Cleanup complete!"
echo ""
echo "Remaining files:"
ls -1 | grep -E "(grammar|readability)" | sort

