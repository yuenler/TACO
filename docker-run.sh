#!/bin/bash
set -e

# Ensure the script is executable
# chmod +x docker-run.sh

# Function to display usage information
function show_help {
    echo "TACO Docker Helper Script"
    echo "Usage: ./docker-run.sh [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build       Build the Docker image"
    echo "  run         Run the Docker container"
    echo "  shell       Start a shell inside the container"
    echo "  download    Download COCO validation dataset"
    echo "  process     Process COCO annotations for TACO"
    echo "  help        Show this help message"
    echo ""
}

# Check if command is provided
if [ $# -eq 0 ]; then
    show_help
    exit 1
fi

COMMAND=$1
shift

case $COMMAND in
    build)
        echo "Building Docker image..."
        docker-compose build
        ;;
    run)
        echo "Running Docker container..."
        docker-compose up -d
        ;;
    shell)
        echo "Starting shell in Docker container..."
        docker-compose exec taco bash
        ;;
    download)
        echo "Downloading COCO validation dataset inside Docker container..."
        # Run the download script inside the Docker container
        docker-compose exec taco bash -c "cd /app && ./download_val_coco.sh"
        ;;
    process)
        echo "Processing COCO annotations for TACO inside Docker container..."
        # Run the processing script inside the Docker container
        docker-compose exec taco python3 create_val_captions.py
        ;;
    help)
        show_help
        ;;
    *)
        echo "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac

exit 0
