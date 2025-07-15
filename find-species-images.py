#!/usr/bin/env python3
"""
Find and display images for a specific tree species by its Latin scientific name.

This script:
1. Takes a Latin scientific tree species name as input
2. Maps the species name to a label ID using the species mapping files
3. Finds all images corresponding to that species in the dataset
4. Displays or exports these images

Usage:
    python find_species_images.py "picea abies" --display
    python find_species_images.py "fagus sylvatica" --output-dir ./beech_images
"""

import argparse
import os
import sys
import yaml
import pandas as pd
from pathlib import Path
import shutil
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple


def load_config() -> dict:
    """Load the configuration from the config.yaml file."""
    config_path = Path("config.yaml")
    if not config_path.exists():
        sys.exit(f"Error: Configuration file not found at {config_path}")
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_latin_to_label_id_map(config: dict) -> Dict[str, int]:
    """Create a mapping from Latin scientific species names to label IDs.
    
    This performs the reverse mapping of what's done in BaseTreeSpeciesClassifier.load_class_names:
    1. Latin scientific name → species_ID (via species_info_file)
    2. species_ID → label_ID (via classes.txt)
    
    Args:
        config: Configuration dictionary containing file paths
        
    Returns:
        Dictionary mapping Latin scientific names to label IDs
    """
    dataset_path = Path(config["dataset_path"])
    classes_file = dataset_path / config["classes_file"]
    species_info_file = dataset_path / config["species_info_file"]
    
    # Validate files exist
    if not classes_file.exists():
        sys.exit(f"Error: Classes file not found at {classes_file}")
    
    if not species_info_file.exists():
        sys.exit(f"Error: Species info file not found at {species_info_file}")
    
    # Step 1: Read the species info Excel file to get the mapping from Sp_Class (Latin scientific name) to Sp_ID
    species_df = pd.read_excel(species_info_file)
    
    # Check if required columns exist
    if 'Sp_ID' not in species_df.columns or 'Sp_Class' not in species_df.columns:
        sys.exit(f"Error: Excel file does not contain required columns 'Sp_ID' and 'Sp_Class'")
    
    # Normalize whitespace in Latin scientific names (trim and collapse multiple spaces)
    species_df["Sp_Class"] = species_df["Sp_Class"].apply(
        lambda name: " ".join(str(name).strip().split()) if pd.notna(name) else ""
    )
    
    # Filter out rows with empty species names
    species_df = species_df[species_df["Sp_Class"] != ""]
    
    latin_to_species_id = dict(zip(
        species_df["Sp_Class"],
        species_df["Sp_ID"].astype(str)  # Convert to string for safer comparison
    ))
    
    # Step 2: Read the classes.txt file to get the mapping from species_ID to label_ID
    with open(classes_file, 'r') as f:
        species_ids = [line.strip() for line in f.readlines()]
    
    # Create a mapping from species_ID to label_ID
    species_id_to_label_id = {species_id: i+1 for i, species_id in enumerate(species_ids)}
    
    # Create the final mapping from Latin scientific name to label_ID
    latin_to_label_id = {}
    for latin_name, species_id in latin_to_species_id.items():
        if species_id in species_id_to_label_id:
            latin_to_label_id[latin_name] = species_id_to_label_id[species_id]
    
    return latin_to_label_id


def find_images_by_label_id(config: dict, label_id: int) -> List[Tuple[str, Tuple[float, float, float, float]]]:
    """Find all images containing the specified label ID and their bounding boxes.
    
    Args:
        config: Configuration dictionary containing file paths
        label_id: The label ID to search for
        
    Returns:
        List of (image_path, bounding_box) tuples, where bounding_box is (x_min, y_min, x_max, y_max)
    """
    print(label_id)

    dataset_path = Path(config["dataset_path"])
    image_dir = dataset_path / config["train_dir"] / config["image_subdir"]
    label_dir = dataset_path / config["train_dir"] / config["label_subdir"]
    
    # Add validation directory images as well
    val_image_dir = dataset_path / config["val_dir"] / config["image_subdir"]
    val_label_dir = dataset_path / config["val_dir"] / config["label_subdir"]
    
    image_dirs = [image_dir, val_image_dir]
    label_dirs = [label_dir, val_label_dir]
    
    # Find all images with the given label ID
    matching_images = []
    
    for img_dir, lbl_dir in zip(image_dirs, label_dirs):
        if not img_dir.exists() or not lbl_dir.exists():
            continue
            
        # Get all image files
        image_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png") or f.endswith(".jpg")])
        
        # Get all label files
        label_files = {
            os.path.splitext(f)[0]: lbl_dir / f
            for f in os.listdir(lbl_dir)
            if f.endswith(".txt")
        }
        
        # Check each image for the target label ID
        for img_name in image_files:
            base_name = os.path.splitext(img_name)[0]
            label_path = label_files.get(base_name)
            
            if label_path and label_path.exists():
                try:
                    df = pd.read_csv(label_path, header=None, delimiter=" ")
                    
                    # Find rows with the target label ID
                    matching_rows = df[df[0] == label_id]
                    
                    if not matching_rows.empty:
                        img_path = img_dir / img_name
                        
                        # Add each instance of the species in the image
                        for _, row in matching_rows.iterrows():
                            # Format is: species_id, xcenter, ycenter, width, height
                            # These values are normalized (0-1) relative to image dimensions
                            species_id, xcenter, ycenter, width, height = row
                            
                            # Load image to get its dimensions
                            img = Image.open(img_path)
                            img_width, img_height = img.size
                            
                            # Convert normalized coordinates to absolute pixel coordinates
                            abs_xcenter = xcenter * img_width
                            abs_ycenter = ycenter * img_height
                            abs_width = width * img_width
                            abs_height = height * img_height
                            
                            # Convert center-based coordinates to corner-based for PIL's crop
                            x_min = max(0, abs_xcenter - (abs_width / 2))
                            y_min = max(0, abs_ycenter - (abs_height / 2))
                            x_max = min(img_width, abs_xcenter + (abs_width / 2))
                            y_max = min(img_height, abs_ycenter + (abs_height / 2))
                            
                            # Skip images with any dimension less than 10 pixels
                            if (x_max - x_min) < 10 or (y_max - y_min) < 10:
                                continue
                                
                            matching_images.append((str(img_path), (x_min, y_min, x_max, y_max)))
                except Exception as e:
                    print(f"Error processing {label_path}: {e}")
    
    return matching_images


def crop_and_display_images(image_info_list: List[Tuple[str, Tuple[float, float, float, float]]], 
                            species_name: str, 
                            output_dir: str = None, 
                            display: bool = False,
                            page: int = 1) -> None:
    """Crop, display or save images of the specified species.
    
    Args:
        image_info_list: List of (image_path, bounding_box) tuples
        species_name: Latin scientific name of the species (for display purposes)
        output_dir: Directory to save cropped images to (if None, won't save)
        display: Whether to display the images using matplotlib
    """
    if not image_info_list:
        print(f"No images found for species: {species_name}")
        return
    
    print(f"Found {len(image_info_list)} instances of {species_name}")
    
    # Prepare output directory if specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Images will be saved to: {output_path}")
    
    # Display/save images
    if display:
        # Limit to a reasonable number for display
        display_limit = min(20, len(image_info_list))
        num_cols = min(5, display_limit)
        num_rows = (display_limit + num_cols - 1) // num_cols
        
        plt.figure(figsize=(15, 3 * num_rows))
        plt.suptitle(f"Tree Species: {species_name} (showing {display_limit} of {len(image_info_list)} images)", 
                     fontsize=16)
        
        if page:
            image_info_list=image_info_list[int(num_cols*num_rows*(page-1)):]

        for i, (img_path, bbox) in enumerate(image_info_list[:display_limit]):
            try:
                # Load and crop image
                img = Image.open(img_path).convert("RGB")
                x_min, y_min, x_max, y_max = bbox
                cropped_img = img.crop((x_min, y_min, x_max, y_max))

                # Display cropped image
                plt.subplot(num_rows, num_cols, i + 1)

                plt.imshow(cropped_img)
                plt.title(f"#{i+1}")
                plt.axis('off')
            except Exception as e:
                print(f"Error displaying image {img_path}: {e}")
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()
    
    # Save images if output directory is specified
    if output_dir:
        for i, (img_path, bbox) in enumerate(image_info_list):
            try:
                # Load and crop image
                img = Image.open(img_path).convert("RGB")
                x_min, y_min, x_max, y_max = bbox
                cropped_img = img.crop((x_min, y_min, x_max, y_max))
                
                # Save cropped image
                output_file = output_path / f"{species_name.replace(' ', '_')}_{i+1}.png"
                cropped_img.save(output_file)
                
                if (i + 1) % 50 == 0:
                    print(f"Saved {i+1}/{len(image_info_list)} images...")
                    
            except Exception as e:
                print(f"Error saving image {img_path}: {e}")
        
        print(f"Successfully saved {len(image_info_list)} images to {output_path}")


def list_available_species(config: dict) -> None:
    """List all available tree species with their Latin scientific names."""
    latin_to_label_id = create_latin_to_label_id_map(config)
    
    print("\nAvailable tree species:")
    print("=" * 50)
    for i, (name, label_id) in enumerate(sorted(latin_to_label_id.items(), key=lambda x: x[0])):
        print(f"{i+1:2d}. {name} (Label ID: {label_id})")
    print("=" * 50)
    print(f"Total: {len(latin_to_label_id)} species")
    print("Use the exact name as shown above when searching for species images.")


def main():
    parser = argparse.ArgumentParser(description="Find images for a specific tree species by Latin scientific name")
    parser.add_argument("species_name", nargs="?", type=str, help="Latin scientific name of the tree species to search for")
    parser.add_argument("--output-dir", type=str, help="Directory to save the cropped species images to")
    parser.add_argument("--display", action="store_true", help="Display found images using matplotlib")
    parser.add_argument("--list", action="store_true", help="List all available tree species")
    parser.add_argument("--page", type=int, help="page number")

    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # List available species if requested
    if args.list:
        list_available_species(config)
        return
    
    # Validate species name
    if not args.species_name:
        parser.print_help()
        print("\nError: You must specify a species name or use --list to see available species")
        return
    
    # Create mapping from Latin scientific names to label IDs
    latin_to_label_id = create_latin_to_label_id_map(config)
    
    # Normalize input species name the same way as in the mapping
    normalized_species_name = " ".join(args.species_name.strip().split())
    
    # Check if the species name exists (case-insensitive)
    # Create case-insensitive lookup dictionary
    case_insensitive_lookup = {name.lower(): name for name in latin_to_label_id.keys()}
    
    if normalized_species_name.lower() in case_insensitive_lookup:
        # Get the properly cased name
        normalized_species_name = case_insensitive_lookup[normalized_species_name.lower()]
    else:
        print(f"Error: Species '{args.species_name}' not found in the dataset")
        print("Use --list to see all available species")
        
        # Try to suggest close matches
        close_matches = [name for name in latin_to_label_id.keys() 
                         if normalized_species_name.lower() in name.lower()]
        if close_matches:
            print("\nDid you mean one of these?")
            for match in sorted(close_matches):
                print(f"  - {match}")
        
        return
    
    # Get the label ID for the requested species
    label_id = latin_to_label_id[normalized_species_name]
    print(f"Looking for images of '{normalized_species_name}' (Label ID: {label_id})")
    
    # Find all matching images
    matching_images = find_images_by_label_id(config, label_id)
    
    # Display or save the images
    crop_and_display_images(
        matching_images, 
        normalized_species_name, 
        output_dir=args.output_dir, 
        display=args.display,
        page=args.page
    )


if __name__ == "__main__":
    main()