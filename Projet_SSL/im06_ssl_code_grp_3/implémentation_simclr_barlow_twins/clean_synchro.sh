#!/bin/bash
# archive_all_runs_final.sh
#
# This script robustly finds all offline wandb runs across multiple possible
# directories, associates them with their SLURM job log, and creates a
# separate, clean .tar.gz archive for each run.
# This version uses grep/sed for maximum compatibility.

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
BASE_DIR=$(eval echo "~/im06-ssl")
LOGS_DIR="$BASE_DIR"
OUTPUT_DIR="$BASE_DIR/clean_archives_all"

# --- Define all possible locations for wandb runs ---
# Add any other potential parent directories to this array
SEARCH_PATHS=(
    "$BASE_DIR/wandb_offline/wandb"
    "$BASE_DIR/wandb/wandb"
    "$BASE_DIR/wandb"
    "$BASE_DIR/wandb_offline"
)

# --- Main Logic ---
echo "Starting individual run archiving (multi-path search version)."
mkdir -p "$OUTPUT_DIR"
echo "Archives will be saved to: $OUTPUT_DIR"

# Use a temporary file to store unique run paths to avoid duplicates
# in case SEARCH_PATHS have overlapping subdirectories.
tmp_runs_file=$(mktemp)

# Find all unique run directories from all search paths
for path in "${SEARCH_PATHS[@]}"; do
    if [ -d "$path" ]; then
        find "$path" -maxdepth 2 -name "offline-run-*" -type d >> "$tmp_runs_file"
    fi
done

# Sort and keep only unique paths
sort -u "$tmp_runs_file" -o "$tmp_runs_file"

if [ ! -s "$tmp_runs_file" ]; then
    echo "Error: No wandb runs found in any of the specified search paths." >&2
    rm "$tmp_runs_file"
    exit 1
fi

echo "Found $(wc -l < "$tmp_runs_file") unique runs to process."

while read -r run_dir; do
    run_folder=$(basename "$run_dir")
    echo "---------------------------------"
    echo "Processing: $run_folder (from $run_dir)"

    metadata_file="$run_dir/files/wandb-metadata.json"
    if [ ! -f "$metadata_file" ]; then
        echo "  -> Warning: metadata.json not found. Skipping run."
        continue
    fi

    # Extract SLURM Job ID using grep and sed.
    job_id=$(grep '"job_id":' "$metadata_file" | sed 's/[^0-9]*//g' | head -n 1)
    if [ -z "$job_id" ]; then
        echo "  -> Warning: SLURM Job ID not found in metadata. Skipping run."
        continue
    fi
    echo "  -> Associated Job ID: $job_id"

    # Build the arguments for the tar command.
    run_id_short=$(echo "$run_folder" | sed 's/.*-//')
    archive_name="job_${job_id}_run_${run_id_short}.tar.gz"
    archive_path="$OUTPUT_DIR/$archive_name"
    
    tar_args=("-czf" "$archive_path" "-C" "$(dirname "$run_dir")" "$run_folder")

    # Find the corresponding log file.
    log_file=$(find "$LOGS_DIR" -maxdepth 1 -name "simclr_train_job_${job_id}.out" 2>/dev/null)
    if [ -n "$log_file" ]; then
        echo "  -> Log file found: $(basename "$log_file")"
        tar_args+=("-C" "$LOGS_DIR" "$(basename "$log_file")")
    else
        echo "  -> Warning: Log file for job $job_id not found. Archiving run only."
    fi

    tar "${tar_args[@]}"
    echo "  -> Archive created: $archive_name"

done < "$tmp_runs_file"

# Cleanup
rm "$tmp_runs_file"

echo "---------------------------------"
echo "Processing complete."
echo "All runs have been individually archived in '$OUTPUT_DIR'."
echo
echo "--- Next Step ---"
echo "On your local machine, run the following command to download the archives:"
echo
echo "scp -r $(whoami)@$(hostname):$OUTPUT_DIR ."
echo
