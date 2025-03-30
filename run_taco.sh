#!/bin/bash
# Activate the virtual environment
source .venv/bin/activate

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run TACO on a single image
python generate_images_using_image_cap_dataset.py \
  --image_folder_root coco/val2014 \
  --image_cap_dict_root single_image_test.json \
  --checkpoint checkpoint/lambda_0.0016.pth.tar
