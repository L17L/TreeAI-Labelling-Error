# Tree Species Image Finder

## Overview
This tool allows you to find and visualize images of specific tree species from a dataset by their Latin scientific names. It was created to verify the accuracy of species labeling in tree datasets and has uncovered significant labeling issues that need to be addressed.

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Make sure you have a new dataset 12_RGB_ObjDet_640_fL, downloaded and unzipped from https://zenodo.org/records/15351054. Worked last on 

## Usage

List all available species:
```
python find-species-images.py --list
```

Display images for a specific species:
```
python find-species-images.py "dead tree" --display
```

Save species images to a directory:
```
python find-species-images.py "fagus sylvatica" --output-dir ./beech_images
```

Navigate through multiple pages of results:
```
python find-species-images.py "betula alleghaniensis" --display --page 2
```

## Labeling Issues Found

This tool was created as a reproducibility tool to identify and document labeling inconsistencies in the dataset. The main issues discovered include:

1. **Mislabeled Dead Trees**:  Running `python find-species-images.py "dead tree" --display` shows no actual dead tree images, despite this class existing and apparently having this name (see --list).

2. **Incorrectly Labeled Yellow Birch (Betula alleghaniensis)**: Running `python find-species-images.py "betula alleghaniensis" --display` shows only dead trees, not yellow birch trees.

The focus on dead trees was deliberate, as their visual characteristics (lack of foliage, bare branches) make them easily distinguishable from living trees on first sight. This provided a clear test case for identifying labeling inconsistencies. Given the severity of these discovered issues, it is highly likely that other species labels in the dataset are also incorrect.

These findings suggest a systematic error in the mapping between species IDs and their Latin scientific names in the dataset. Since this label-name relationship was provided by the dataset creators, they need to be informed about these issues to correct the underlying data.

## How It Works

The tool works by:
1. Mapping Latin scientific species names to label IDs using the dataset's mapping files
2. Finding all images containing the specified label ID
3. Extracting the bounding box coordinates for each instance of the species
    Tree cutouts which are less then 10 pixels in any dimensions are skipped for comprehensibility.
4. Displaying or saving the cropped images showing only the tree species

## Reproducibility

This project serves as a reproducible demonstration of the labeling issues. Researchers and dataset providers can use this tool to verify the findings and correct the underlying data.

## Future Work

The discovery of these mislabeling issues suggests a need for a comprehensive audit of the entire dataset. While dead trees were chosen initially because they are visually distinctive and easy to verify without expert knowledge, future work should include:

1. Systematic review of other species labels with domain experts
2. Statistical analysis of labeling consistency across the dataset
3. Development of automated methods to detect potential mislabeling
