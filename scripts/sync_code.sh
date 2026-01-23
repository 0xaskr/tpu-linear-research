#!/bin/bash

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Cluster name files
TPU_FILE="${SCRIPT_DIR}/../.cluster_name_tpu"
GPU_FILE="${SCRIPT_DIR}/../.cluster_name_gpu"

# Argument handling
TARGET=$1

if [ "$TARGET" == "gpu" ]; then
    CLUSTER_NAME_FILE="$GPU_FILE"
elif [ "$TARGET" == "tpu" ]; then
    CLUSTER_NAME_FILE="$TPU_FILE"
else
    # Auto-detection based on file existence and modification time
    if [ -f "$TPU_FILE" ] && [ ! -f "$GPU_FILE" ]; then
        CLUSTER_NAME_FILE="$TPU_FILE"
        echo "Auto-detected TPU cluster."
    elif [ ! -f "$TPU_FILE" ] && [ -f "$GPU_FILE" ]; then
        CLUSTER_NAME_FILE="$GPU_FILE"
        echo "Auto-detected GPU cluster."
    elif [ -f "$TPU_FILE" ] && [ -f "$GPU_FILE" ]; then
        if [ "$TPU_FILE" -nt "$GPU_FILE" ]; then
            CLUSTER_NAME_FILE="$TPU_FILE"
            echo "Auto-detected TPU cluster (newer)."
        else
            CLUSTER_NAME_FILE="$GPU_FILE"
            echo "Auto-detected GPU cluster (newer)."
        fi
    else
        echo "Error: No .cluster_name_tpu or .cluster_name_gpu file found."
        echo "Usage: $0 [tpu|gpu]"
        exit 1
    fi
fi

# Check if selected file exists
if [ ! -f "$CLUSTER_NAME_FILE" ]; then
    echo "Error: Cluster name file '$CLUSTER_NAME_FILE' not found."
    exit 1
fi

CLUSTER_NAME=$(cat "$CLUSTER_NAME_FILE")

if [ -z "$CLUSTER_NAME" ]; then
    echo "Error: Cluster name is empty in $CLUSTER_NAME_FILE."
    exit 1
fi

echo "Syncing code to cluster: $CLUSTER_NAME..."
# Use --workdir to explicitly sync the project root
sky exec "$CLUSTER_NAME" --workdir "$PROJECT_ROOT" "echo 'Code synced successfully.'"

if [ $? -eq 0 ]; then
    echo "Sync complete."
else
    echo "Error: Sync failed."
    exit 1
fi