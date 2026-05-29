#!/usr/bin/env bash
# monitor.bash
# Monitors data/processed/ for HDF5 files with 5-digit indices and runs
# prediction scripts for any that lack a DEM output file.
# Since files are written in ascending order, only the last (highest-numbered)
# file is age-checked — it may still be mid-write. All earlier files are safe
# to process regardless of age.

set -euo pipefail

PROCESSED_DIR="data/processed"
PREDICTIONS_DIR="data/predictions/DEM"
MIN_AGE_MINUTES=10
SCRIPT="python scripts/res_predict_model.py"
PREFIX="sys_cal"
DATASET="DEM"

# ── Collect matching files ────────────────────────────────────────────────────
# Pattern: any filename ending in exactly 5 digits before .hdf5
mapfile -t FILES < <(
    find "$PROCESSED_DIR" -maxdepth 1 -type f \
        -regextype posix-extended \
        -regex ".*/[^/]*[0-9]{5}\.hdf5" \
    | sort
)

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "No matching files found in $PROCESSED_DIR"
    exit 0
fi

LAST_FILE="${FILES[-1]}"

echo "Found ${#FILES[@]} candidate file(s)."
echo "────────────────────────────────────────────────────────────────"

# ── Process each file in ascending order ─────────────────────────────────────
for FILEPATH in "${FILES[@]}"; do
    BASENAME=$(basename "$FILEPATH")

    # Extract the 5-digit number (last 5 digits before .hdf5)
    if [[ "$BASENAME" =~ ([0-9]{5})\.hdf5$ ]]; then
        NUMBER="${BASH_REMATCH[1]}"
    else
        echo "SKIP  $BASENAME  (could not parse 5-digit index)"
        continue
    fi

    # ── Age check on last file only ───────────────────────────────────────────
    # Earlier files are complete by definition; only the highest-numbered file
    # might still be mid-write.
    if [[ "$FILEPATH" == "$LAST_FILE" ]]; then
        AGE_CHECK=$(find "$FILEPATH" -mmin "+${MIN_AGE_MINUTES}" 2>/dev/null || true)
        if [[ -z "$AGE_CHECK" ]]; then
            echo "SKIP  $BASENAME  (last file, less than ${MIN_AGE_MINUTES} min old — may still be writing)"
            continue
        fi
    fi

    # ── Output file check ─────────────────────────────────────────────────────
    OUTPUT_FILE="${PREDICTIONS_DIR}/${PREFIX}_craterdist_${NUMBER}.hdf5"
    if [[ -f "$OUTPUT_FILE" ]]; then
        echo "DONE  $BASENAME  (prediction already exists)"
        continue
    fi

    # ── Run prediction pipeline ───────────────────────────────────────────────
    echo "RUN   $BASENAME  (index=$NUMBER)"

    echo "cnn-prediction..."
    $SCRIPT cnn-prediction \
        --index="$NUMBER" \
        --prefix="$PREFIX" \
        --dataset="$DATASET"

    echo "  → make-prediction..."
    $SCRIPT make-prediction \
        --index="$NUMBER" \
        --prefix="$PREFIX" \
        --dataset="$DATASET"

    echo "Done for index $NUMBER"
done

echo "────────────────────────────────────────────────────────────────"
echo "Monitor pass complete."
