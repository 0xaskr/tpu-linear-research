#!/bin/bash

# Check if correct number of arguments provided
if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "Usage: $0 <accelerator> [name]"
    echo "Example: $0 tpu-v6e-1 matmul"
    exit 1
fi

# Get arguments
ACCELERATOR="$1"
NAME="$2"

# Validate arguments
if [ -z "$ACCELERATOR" ]; then
    echo "Error: Accelerator type cannot be empty"
    exit 1
fi

if [ -z "$NAME" ]; then
    echo "Error: name cannot be empty"
    exit 1
fi

# Function to sanitize ref name for cluster naming
sanitize_ref() {
    local name="$1"
    # Replace invalid characters with dash
    name=$(echo "$name" | sed 's/[^a-zA-Z0-9._-]/-/g')
    # Remove leading non-letter characters
    name=$(echo "$name" | sed 's/^[^a-zA-Z]*//')
    # Ensure it doesn't end with dash, dot or underscore
    name=$(echo "$name" | sed 's/[-._]*$//')
    # Truncate to reasonable length (max 20 chars to leave room for prefix/suffix)
    name=$(echo "$name" | cut -c1-20)
    # If empty after sanitization, use "default"
    if [ -z "$name" ]; then
        name="default"
    fi
    echo "$name"
}

# Sanitize ref name
SANITIZED_NAME=$(sanitize_ref "$NAME")

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Create a temporary rendered yaml file in the script directory
TEMP_YAML="${SCRIPT_DIR}/tpu_resource_rendered.sky.yaml"

# Read the template and replace variables
# Use | as delimiter to handle slashes in branch names
sed -e "s|\$ACCELERATOR|${ACCELERATOR}|g" \
    -e "s|\$NAME|${NAME}|g" \
    "${SCRIPT_DIR}/tpu_resource.sky.yaml" > "$TEMP_YAML"

# Create cluster name with ref
CLUSTER_NAME="tpu-$ACCELERATOR-$SANITIZED_NAME-$RANDOM"

# Save cluster name to file for other scripts to use
echo "$CLUSTER_NAME" > "${SCRIPT_DIR}/../.cluster_name_tpu"

# Execute sky launch command
echo ""
echo "Executing command with:"
echo "  Accelerator: ${ACCELERATOR}"
echo "  Name: ${NAME}"
echo "  Sanitized Name: ${SANITIZED_NAME}"
echo "  Cluster Name: ${CLUSTER_NAME}"
echo ""

sky launch "$TEMP_YAML" \
    --cluster="$CLUSTER_NAME" \
    --infra=gcp \
    -i 30 \
    --down \
    -y

# Store the exit code
EXIT_CODE=$?

# Clean up temporary file
rm -f "$TEMP_YAML"

# Check if launch command was successful
if [ $EXIT_CODE -ne 0 ]; then
    echo "Error: Sky launch command failed"
    exit $EXIT_CODE
fi

# Wait for cluster to be UP
echo ""
echo "Waiting for cluster $CLUSTER_NAME to be UP..."

TIMEOUT=600  # 600 seconds = 10 minutes
START_TIME=$(date +%s)

while true; do
    # Check if cluster is UP
    if sky status --refresh | grep "^$CLUSTER_NAME" | grep -q "UP"; then
        echo "Success: Cluster $CLUSTER_NAME is UP"
        exit 0
    fi

    # Check timeout
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))

    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo "Error: Timeout waiting for cluster to be UP (waited ${TIMEOUT} seconds)"
        # Show current status for debugging
        echo "Current status:"
        sky status --refresh | grep "^$CLUSTER_NAME" || echo "Cluster not found in status"
        exit 1
    fi

    # Show progress
    echo "Checking status... (elapsed: ${ELAPSED}s / ${TIMEOUT}s)"

    # Wait before checking again
    sleep 10
done

echo "Waiting for 10 seconds, then submit job to the cluster"
sleep 10
