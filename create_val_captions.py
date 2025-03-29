#!/usr/bin/env python3
import os
import json

"""
This script processes the COCO validation annotations file and creates a JSON file
with captions in the format expected by the TACO model, including ALL validation images.
"""

# Paths to annotation files
ANNOTATIONS_PATH = 'annotations'
VAL_CAPTIONS = os.path.join(ANNOTATIONS_PATH, 'captions_val2014.json')
OUTPUT_FILE = 'val2014_captions.json'

def process_captions(caption_file):
    """
    Extracts image filenames and captions from COCO annotation file.
    Returns a dictionary mapping image filenames to captions.
    """
    with open(caption_file, 'r') as f:
        data = json.load(f)
    
    # Create image ID to filename mapping
    image_id_to_filename = {}
    for image in data['images']:
        image_id_to_filename[image['id']] = image['file_name']
    
    # Create filename to caption mapping
    # Only keep one caption per image (the first one found)
    filename_to_caption = {}
    for annotation in data['annotations']:
        image_id = annotation['image_id']
        filename = image_id_to_filename.get(image_id)
        if filename and filename not in filename_to_caption:
            filename_to_caption[filename] = annotation['caption']
    
    return filename_to_caption

def main():
    # Process validation captions
    print("Processing validation captions...")
    val_captions = process_captions(VAL_CAPTIONS)
    print(f"Found {len(val_captions)} validation images with captions")
    
    # Save validation captions to file
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(val_captions, f)
    
    print(f"Saved {len(val_captions)} captions to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
