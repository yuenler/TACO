#!/bin/bash

# Create directories
mkdir -p coco/images
cd coco

echo "Downloading MS-COCO 2014 validation images (around 1GB)..."
curl -L -O http://images.cocodataset.org/zips/val2014.zip

echo "Downloading annotations (around 240MB)..."
curl -L -O http://images.cocodataset.org/annotations/annotations_trainval2014.zip

echo "Extracting validation images..."
unzip -q val2014.zip

echo "Extracting annotations..."
unzip -q annotations_trainval2014.zip

echo "Cleaning up zip files to save space..."
rm val2014.zip annotations_trainval2014.zip

echo "Creating compatible caption file for TACO model..."
python ../create_val_captions.py

echo "Download and extraction complete!"
echo "Images saved to coco/val2014"
echo "Captions saved to coco/val2014_captions.json"
