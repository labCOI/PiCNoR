# PiCNoR
PiCNoR: Pixel-wise Cluster-driven Non-rigid Registration for 3D Microscopic Volumes


## Description
This Python script performs keypoint extraction, matching, and transformation between images. It supports processing both individual pairs of images and directories containing multiple images. The script uses SIFT, KAZE and Superpoint detectors and L2, Hamming or Lightglue matchers to align and register images, and it allows customization of parameters such as the detector type, color scale, and the number of clusters.

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
4. Be sure to install [LightGlue]("https://github.com/cvg/LightGlue/tree/main")

## Usage
1. Open a terminal or command prompt.
2. Navigate to the project directory.
3. Create a new config file or use the default one.
3. Run the script with the necessary arguments:
    ```sh
    python main.py --config <yaml-config-path> [options]
    ```

### Example
```sh
python main.py --config default_config.yaml
```

## Project Structure
```
.
├── main.py                 # The main script file
├── utils.py                # Utility functions used in the script
├── config.py               # Loading config file
├── customlogger.py         # Two types of custom Logging
├── PlotViewer.py           # TKinter GUI to view plots
├── registration.py         # Registration Pipeline
├── default_config.yaml     # Default Config file to run.
├── .gitignore              # Git ignore file
└── README.md               # This README file
```

## Custom Directories
The following directories are excluded from the repository using the `.gitignore` file Since they contained output images:
- `DataCorrected/`
- `DataWang/`
- `Outputs/`

## Logging
The script generates two log files `full_log.log` and `short_log.log` in the specified output directory, which contains detailed logs of the processing steps and any issues encountered.

## Licensing
This dataset is licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license. You are free to:

- **Share**: Copy and redistribute the material in any medium or format.
- **Adapt**: Remix, transform, and build upon the material for any purpose, even commercially.

**Under the following terms:**

- **Attribution**: You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.

For details, see the full license text here: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).


## Citation
The result are going to be submitted to IEEE transactions on Computational Imaging. Citation will be available after submission.

## Contact
For any questions or issues regarding this dataset, please contact:
- **Name**: Parsa Mojarad Adi
- **Email**: parsa.78.99@gmail.com
- **Affiliation**: Shahid Beheshti University

