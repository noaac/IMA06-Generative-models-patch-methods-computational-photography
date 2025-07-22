#!/bin/bash

echo "WandB Clean Archive Synchronizer"
echo "================================="

# --- Helper Functions ---

show_help() {
    echo "Usage: $0 [ARCHIVE_FILE] [PROJECT_NAME]"
    echo
    echo "Synchronizes a single, clean wandb run archive."
    echo
    echo "Modes:"
    echo "  $0                            - Interactive mode: lists all archives and prompts for selection."
    echo "  $0 clean_results_job_354092.tar.gz  - Semi-interactive: prompts for project name for the specified archive."
    echo "  $0 clean_results_job_354092.tar.gz MyProject - Direct mode: synchronizes immediately."
}

# This function extracts key info from the .out file inside an archive
# without fully extracting the archive.
get_archive_summary() {
    local archive_file=$1
    # Use tar to list contents, find the .out file, and extract it to stdout
    local out_content
    out_content=$(tar -xzOf "$archive_file" --wildcards "*.out" 2>/dev/null || echo "No .out file")
    
    if [[ "$out_content" != "No .out file" ]]; then
        # Extract transformations or other key args
        local transforms
        transforms=$(echo "$out_content" | grep -o -E 'transform_[a-z]+ (true|false)' | tr '\n' ' ')
        echo "    -> Params: ${transforms:-N/A}"
    fi
}

# --- Main Sync Logic ---

sync_run() {
    local archive_file=$1
    local project_name=$2
    local temp_dir="wandb_sync_temp"

    echo "----------------------------------------"
    echo "Synchronizing Archive: $archive_file"
    echo "Project: $project_name"
    echo "----------------------------------------"

    # 1. Clean up and create a temporary extraction directory
    rm -rf "$temp_dir"
    mkdir "$temp_dir"

    # 2. Extract the single run into the temp directory
    echo "Extracting archive..."
    if ! tar -xzf "$archive_file" -C "$temp_dir"; then
        echo "Error: Failed to extract archive '$archive_file'."
        rm -rf "$temp_dir"
        exit 1
    fi

    # 3. Find the run path inside the temp directory
    # New structure: clean_results_job_*/job_outputs_*/wandb/wandb/offline-run-*
    echo "Searching for wandb run directory..."
    local run_path
    # Prendre seulement le premier résultat trouvé (-quit)
    run_path=$(find "$temp_dir" -path "*/wandb/wandb/offline-run-*" -type d -print -quit)

    if [ -z "$run_path" ]; then
        echo "Error: No 'offline-run-*' directory found in the archive."
        echo "Searching for any wandb directory..."
        find "$temp_dir" -name "wandb" -type d
        rm -rf "$temp_dir"
        exit 1
    fi

    echo "Found run directory: $run_path"

    # 4. Synchronize with wandb
    echo "Syncing run..."
    if wandb sync "$run_path" --entity ssl-im06 --project "$project_name"; then
        local run_id
        run_id=$(basename "$run_path" | sed 's/.*-//')
        echo
        echo "Synchronization successful!"
        echo "  Run ID: $run_id"
        echo "  Project: $project_name"
        echo "  URL: https://wandb.ai/ssl-im06/$project_name"
    else
        echo "Error: wandb sync command failed."
    fi

    # 5. Clean up
    echo "Cleaning up temporary files..."
    rm -rf "$temp_dir"
    echo "Done."
}

# --- Script Entry Point ---

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_help
    exit 0
fi

# Direct Mode: Archive and Project provided
if [ -n "$1" ] && [ -n "$2" ]; then
    if [ ! -f "$1" ]; then
        echo "Error: Archive file not found: $1"
        exit 1
    fi
    sync_run "$1" "$2"
    exit 0
fi

# Semi-Interactive Mode: Archive provided
if [ -n "$1" ]; then
    if [ ! -f "$1" ]; then
        echo "Error: Archive file not found: $1"
        exit 1
    fi
    echo "Enter the WandB project name for this run:"
    read -p "Project: " project_name
    if [ -z "$project_name" ]; then
        echo "Error: Project name cannot be empty."
        exit 1
    fi
    sync_run "$1" "$project_name"
    exit 0
fi

# Fully Interactive Mode: No arguments
echo "Searching for clean run archives..."
# Use a portable while-read loop instead of mapfile for macOS compatibility
archives=()
while IFS= read -r line; do
    archives+=("$line")
done < <(ls -1 clean_results_job_*.tar.gz 2>/dev/null)

if [ ${#archives[@]} -eq 0 ]; then
    echo "No clean archives (clean_results_job_*.tar.gz) found in this directory."
    exit 1
fi

echo "Available archives:"
for i in "${!archives[@]}"; do
    echo "  $((i+1)). ${archives[$i]}"
    get_archive_summary "${archives[$i]}"
done
echo

read -p "Enter the NUMBER of the archive to sync: " choice
if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt "${#archives[@]}" ]; then
    echo "Error: Invalid selection."
    exit 1
fi

selected_archive=${archives[$((choice-1))]}

echo "Enter the WandB project name for this run:"
read -p "Project: " project_name
if [ -z "$project_name" ]; then
    echo "Error: Project name cannot be empty."
    exit 1
fi

sync_run "$selected_archive" "$project_name"