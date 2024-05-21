# Automatic 3D Registration Method

## Description
This Python script performs keypoint extraction, matching, and transformation between images. It supports processing both individual pairs of images and directories containing multiple images. The script uses SIFT or KAZE detectors and L2 or Hamming matchers to align and register images, and it allows customization of parameters such as the detector type, color scale, and the number of clusters.

## Installation
1. Clone the repository to your local machine:
    ```sh
    git clone <repository-url>
    ```
2. Navigate to the project directory:
    ```sh
    cd <repository-directory>
    ```
3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
1. Open a terminal or command prompt.
2. Navigate to the project directory.
3. Run the script with the necessary arguments:
    ```sh
    python main.py --source <source-image-path> --target <target-image-path> --out <output-folder> --nclusters <number-of-clusters> [options]
    ```

### Arguments
- `--type`: Type of operation (`Pair` for individual pairs, `Dir` for directories). Default is `Pair`.
- `--source`: Path to the source image or directory.
- `--target`: Path to the target image.
- `--out`: Path to the output folder.
- `--color`: Color scale to use (`RGB` or `Gray`). Default is `RGB`.
- `--detector`: Detector to use (`SIFT`, `AKAZE`). Default is `AKAZE`.
- `--threshold`: Detector threshold. Default is `0.0001`.
- `--matcher`: Matcher to use (`L2` or `Hamming`). Default is `L2`.
- `--nclusters`: Number of clusters.
- `--save`: Save outputs (flag).
- `--show`: Show outputs (flag).
- `--fix`: Fix outputs (flag).

### Example
```sh
python main.py --type Pair --source path/to/source.jpg --target path/to/target.jpg --out path/to/output --color RGB --detector AKAZE --threshold 0.0001 --matcher L2 --nclusters 5 --save --show --fix

### Project Structure
.
├── main.py        # The main script file
├── utils.py       # Utility functions used in the script
├── .gitignore     # Git ignore file
└── README.md      # This README file