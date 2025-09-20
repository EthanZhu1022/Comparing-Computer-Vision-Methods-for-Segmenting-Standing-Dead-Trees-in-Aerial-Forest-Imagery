# A Comparative Analysis of Segmentation Methods for Aerial Tree Imagery

## Project Overview

This project provides a comprehensive comparison of different image segmentation techniques applied to a custom dataset of aerial imagery, with the primary goal of accurately identifying and masking trees. Three distinct families of methods are implemented and evaluated:

1.  **Traditional Computer Vision Methods**: Unsupervised algorithms including **Watershed** and **MeanShift**.
2.  **DeepLabv3 with Custom Fusion**: A deep learning approach using the DeepLabv3 architecture with a custom-modified, 6-channel multi-modal input.
3.  **Systematic U-Net/U-Net++ Pipeline**: An automated deep learning pipeline designed to systematically compare **U-Net** and **U-Net++** architectures across various channel configurations and data augmentation strategies.

Performance for all methods is quantitatively measured using the **Intersection over Union (IoU)** metric, supplemented by qualitative visual analysis.

---

### Part 1: Traditional Computer Vision Methods (Watershed & MeanShift)

###  Overview

This project implements two classical unsupervised image segmentation algorithms — **Watershed** and **MeanShift** — across two spectral image formats:

* **RGB (Red-Green-Blue) images**, and
* **NRG (Near-Infrared-Red-Green) images**.

The primary objective is to evaluate how traditional image segmentation looks like on segmentation accuracy on tree remote sensing imagery. Performance is measured using **Intersection over Union (IoU)**.

---

##  Dataset

* Two sets of images are used:

  * **RGB images**: Located in `USA_segmentation/RGB_images/`
  * **NRG images**: Located in `USA_segmentation/NRG_images/`
* Corresponding binary ground-truth masks are in `USA_segmentation/masks/`
* Images and masks are resized to **(366, 385)** to ensure uniform input size
* File matching is done by suffix (e.g., `RGB_xxx.png` or `NRG_xxx.png` matched with `mask_xxx.png`)

---

##  Preprocessing Pipeline

###  Common Enhancements (Both RGB & NRG)

* Convert the image from BGR to **LAB color space**
* Apply **CLAHE (Contrast Limited Adaptive Histogram Equalization)** to the **L channel** for contrast improvement
* Merge processed L with original A/B channels and convert back to BGR

###  Special Note

* In the **NRG pipeline**, CLAHE is still applied in LAB space despite NIR data
* In the **RGB pipeline**, the procedure is identical but input bands differ

---

##  Segmentation Methods

###  1. Watershed Segmentation

Steps:

1. Convert CLAHE-enhanced image to **grayscale**
2. Apply **Gaussian Blur** and **Triangle Thresholding**
3. Use **Morphological Opening** and **Dilation** to clean binary image
4. Compute **Distance Transform** and extract **sure foreground**
5. Subtract to get **unknown region**
6. Use **connected components** to label markers
7. Apply `cv2.watershed()` to get segmentation output

###  2. MeanShift Segmentation

Steps:

1. Downscale CLAHE-enhanced image (for speed)
2. Apply `cv2.pyrMeanShiftFiltering()` for region smoothing and segmentation
3. Resize output to original shape
4. Apply **grayscale conversion**, **thresholding**, and **morphology** to extract mask

---

##  Evaluation Metric: IoU (Intersection over Union)

The **IoU score** is computed for each predicted mask:

$$
\text{IoU} = \frac{\text{Intersection of prediction and ground truth}}{\text{Union of prediction and ground truth}}
$$

* Binary masks are thresholded at >127 before evaluation
* Mean IoU (**mIoU**) is calculated across the entire dataset

---

##  Results Summary

| Method    | Format | Avg. IoU (mIoU)              
| --------- | ------ | ---------------------------- 
| Watershed | RGB    | 0.0155    
| Watershed | NRG    | 0.0097    
| MeanShift | RGB    | 0.0286
| MeanShift | NRG    | 0.0218 

---

##  Visualization

Each notebook visualizes:

* Input image
* Ground truth mask
* Predicted mask by each method

Sample visual outputs help assess qualitative segmentation performance.

---

##  Timing

* Execution time for each method is measured using `time.time()`
* NRG images (due to added NIR channel) may require slightly more processing

---

##  Requirements

```bash
pip install opencv-python numpy scikit-learn matplotlib
```

---

##  Directory Structure

```
USA_segmentation/
├── RGB_images/             # RGB input images
├── NRG_images/             # NRG input images (with Near-IR)
├── masks/                  # Ground truth binary masks
WatershedAndMeanShiftRGB.ipynb     # RGB segmentation notebook
WatershedAndMeanShiftNRG.ipynb     # NRG segmentation notebook
```

---

##  Final Remarks

* Watershed offers pixel-accurate segmentation but is sensitive to marker quality
* MeanShift is robust and fast, but may oversmooth complex shapes
* NRG data enhances contrast, benefiting both methods

Traditional methods (such as Watershed and MeanShift) have the advantages of **requiring no training, low computational cost, and easy deployment**, but their drawbacks include **sensitivity to image quality and lack of semantic understanding**. In this project, they showed **low accuracy and poor prediction results**, with many **false positives** in the segmented output. In contrast, deep learning methods (such as U-Net and DeepLabv3), though **more complex and data/resource-intensive**, can **automatically learn features and achieve high-accuracy semantic segmentation**, making them suitable for complex scenarios.


## part2 Semantic Segmentation of Trees using DeepLabv3 with NRG Imagery

## 1. Overview

This project implements a deep learning model for the semantic segmentation of trees from aerial imagery. The primary goal is to accurately identify and generate pixel-level masks for vegetation. The model uses the **DeepLabv3 architecture with a ResNet-50 backbone**, pretrained on the ImageNet dataset.

This document details the complete workflow, from data loading and preparation to model training, evaluation, and visualization. The implementation uses the PyTorch framework and focuses on a single-modality approach using **NRG (Near-Infrared, Red, Green) images**.

## 2. Methodology

The project employs a direct, single-modality strategy to solve the segmentation task.

### 2.1. NRG-Only Model

The approach is centered on training the DeepLabv3 model using only **NRG (Near-Infrared, Red, Green) images**. The underlying hypothesis is that the Near-Infrared (NIR) channel, which is highly reflective of chlorophyll in healthy vegetation, provides a distinct and powerful signal that the model can leverage for accurate tree identification.

* **Implementation**: A standard, pretrained DeepLabv3 model is used, which expects a 3-channel image input. The three channels of the NRG image (Near-Infrared, Red, Green) are directly fed into the model. The model's final classification layer is adapted for a binary segmentation task (Class 0: Background, Class 1: Tree).
* **Training**: The model is trained for 100 epochs using the Adam optimizer and Cross-Entropy Loss. The model's performance is monitored on a validation set, and the version with the lowest validation loss is saved as the best model.
* **Evaluation**: The primary metric for evaluating performance is the **Intersection over Union (IoU)**, calculated specifically for the foreground (tree) class. This provides a clear measure of how well the predicted mask overlaps with the ground truth mask.

## 3. Results

The model was trained for **100 epochs**. Throughout the training, the model with the best performance on the validation set was checkpointed. The final result is a trained model (`deeplabv3_nrg_best_model.pth`) capable of segmenting trees from NRG images. The quality of the segmentation is assessed both quantitatively through the foreground IoU metric and qualitatively through visual inspection of the predicted masks against the ground truth.

## 4. Code Explanation

This section details the purpose of each code block in the script.

* **Section 1: Imports and Configuration**: Imports all necessary libraries like `torch`, `os`, `numpy`, and `matplotlib`. It checks for PyTorch version and CUDA availability, and most importantly, defines the file paths for the dataset, specifically pointing to the `NRG_images` and `masks` directories.

* **Section 2: Prepare and Split the Dataset**: Uses `sklearn.model_selection.train_test_split` to partition the list of image filenames into a training set (80%) and a validation set (20%), ensuring a separate dataset for unbiased model evaluation.

* **Section 3: Create a Custom PyTorch Dataset (`SegmentationDataset`)**: Defines a custom `Dataset` class to handle the specific data loading requirements. For each sample, it loads an NRG image and its corresponding mask. It correctly maps image files to mask files by replacing `"NRG_"` with `"mask_"`. The NRG image is converted to `RGB` format to ensure a 3-channel tensor, and the mask is converted to grayscale (`L`).

* **Section 4: Define Data Transforms and DataLoaders**: Sets up the data preprocessing pipeline using `torchvision.transforms`. Both images and masks are resized to `256x256`. The images are normalized using standard ImageNet statistics, which serve as a reasonable starting point. `DataLoader` instances are then created to feed data to the model in batches.

* **Section 5: Define the DeepLabv3 Model, Loss, and Optimizer**:
    * Loads a `deeplabv3_resnet50` model pretrained on ImageNet.
    * Replaces the final classifier head (`model.classifier[4]`) with a new convolutional layer that outputs `NUM_CLASSES=2` channels for our binary segmentation task.
    * Defines the loss function (`torch.nn.CrossEntropyLoss`) and the optimizer (`torch.optim.Adam` with a learning rate of `1e-4`).

* **Section 6: Helper Function for Metric Calculation**: Contains the `calculate_foreground_iou` function. This utility computes the Intersection over Union specifically for the foreground class (trees, index 1), which is a more informative metric for this task than an average over all classes.

* **Section 7: Train and Validate the Model**: This is the core training loop. For each of the `NUM_EPOCHS`, it performs:
    1.  A training phase (`model.train()`) where it processes the training data, calculates loss, and updates model weights via backpropagation.
    2.  A validation phase (`model.eval()`) where it evaluates the model on the validation set without updating weights. It calculates validation loss and the average foreground IoU.
    3.  A checkpointing step that saves the model's state dictionary to `deeplabv3_nrg_best_model.pth` whenever a new best validation loss is achieved.

* **Section 8: Visualize Model Predictions**: After training is complete, this section loads the best-performing model weights. It then generates a plot with a side-by-side comparison of the original NRG image, the ground truth mask, and the model's predicted mask for several random samples from the validation dataset, allowing for a qualitative assessment of the results.

## 5. How to Run

1.  **Directory Structure**: Ensure your data is organized as follows:
    ```
    C:/Users/pxc02/OneDrive/Desktop/archive/USA_segmentation/
    ├── NRG_images/
    │   ├── NRG_1.png
    │   ├── NRG_2.png
    │   └── ...
    └── masks/
        ├── mask_1.png
        ├── mask_2.png
        └── ...
    ```

2.  **Install Dependencies**: Make sure you have the required Python libraries installed.
    ```bash
    pip install torch torchvision numpy Pillow matplotlib scikit-learn tqdm
    ```

3.  **Update Path**: Open the script and verify that the `base_dir` variable in **Section 1** correctly points to your `USA_segmentation` folder.

4.  **Execute Script**: Run the entire script in a Python environment (e.g., a Jupyter Notebook or as a standalone `.py` file). The script will:
    * Print progress for data preparation and model setup.
    * Train the model for 100 epochs, printing the loss and IoU for each epoch.
    * Save the best model as `deeplabv3_nrg_best_model.pth` in the same directory as the script.
    * Display a final plot visualizing the model's predictions on sample images.

### Part 3: DeepLabv3 with Custom Multi-Modal Fusion

## 1. Overview

This project documents the process of developing a deep learning model for semantic segmentation. The primary goal is to accurately identify and create pixel-level masks for trees in a custom dataset of aerial imagery. The core model architecture used is **DeepLabv3 with a ResNet-50 backbone**, implemented in PyTorch and developed within a Jupyter Notebook environment.

This document outlines the experimental journey, from an initial single-modality approach to a more complex multi-modal fusion model, detailing the code improvements made along the way to enhance performance.

## 2. Project Journey & Methodology

The project followed an iterative process of experimentation to find the optimal approach for this specific dataset.

### 2.1. Initial Experiment: NRG-Only Model

The first attempt involved training the DeepLabv3 model using only the **NRG (Near-Infrared, Red, Green) images**. The hypothesis was that the Near-Infrared channel, which is highly sensitive to vegetation, would provide a strong signal for the model.

* **Implementation**: A standard DeepLabv3 model was used, which expects a 3-channel input. The three channels of the NRG image were fed directly into the model.
* **Training**: The model was initially trained for 25 epochs. When the performance proved to be unsatisfactory, the training was extended to 100 epochs.
* **Outcome**: Despite the extended training, the model's performance remained suboptimal. The foreground IoU metric was relatively low, and visual inspection of the predicted masks showed noticeable inaccuracies. This suggested that relying solely on NRG data was not sufficient for achieving high-quality segmentation.

### 2.2. Final Approach: Multi-Modal Fusion (RGB + NRG)

Based on the results of the first experiment, the strategy was revised to a multi-modal approach. The new hypothesis was that combining the rich visual information from **RGB images** with the vegetation-specific data from **NRG images** would provide a more comprehensive input for the model, leading to better performance.

This required significant modifications to the codebase.

#### Key Code Improvements:

* **Multi-Modal Data Loading (6-Channel Input)**:
    * A new `MultiModalDataset` class was implemented.
    * In each step, this class loads both the RGB and the corresponding NRG image for a given sample.
    * After applying transformations to each 3-channel image, the two resulting tensors are concatenated along the channel dimension using `torch.cat`, creating a single 6-channel tensor `[C, H, W]` where `C=6`. This combined tensor serves as the input to the model.

* **Model Architecture Modification**:
    * The standard pretrained ResNet-50 backbone expects a 3-channel input. To handle the new 6-channel data, the model's first convolutional layer (`model.backbone.conv1`) was replaced.
    * A new `nn.Conv2d` layer was created with `in_channels=6`, while all other parameters (output channels, kernel size, stride, etc.) were kept identical to the original layer to maintain the integrity of the downstream architecture.

* **Data Augmentation**:
    * To improve the model's generalization ability and prevent overfitting, data augmentation was introduced for the training set.
    * The `train_image_transform` pipeline now includes random geometric and color transformations:
        * `transforms.RandomHorizontalFlip(p=0.5)`
        * `transforms.RandomVerticalFlip(p=0.5)`
        * `transforms.ColorJitter(...)`
    * The validation dataset was intentionally not augmented to ensure an unbiased and consistent evaluation of the model's performance.

* **Dynamic Learning Rate**:
    * A learning rate scheduler, `torch.optim.lr_scheduler.ReduceLROnPlateau`, was added to the training process.
    * This scheduler monitors the validation loss at the end of each epoch. If the loss stops improving for a defined number of epochs ("patience"), the learning rate is automatically reduced. This allows for more effective convergence and fine-tuning in the later stages of training.

## 3. Final Results

The multi-modal model was trained for **100 epochs**. The combination of a richer 6-channel input and the advanced training strategies resulted in a noticeable improvement in performance compared to the initial NRG-only model. The final foreground IoU metric was higher, and a visual inspection of the predicted masks confirmed a better alignment with the ground truth.

## 4. Setup and Installation

Before running the code, you need to install the necessary Python libraries.

1.  **Create an Environment (Recommended)**: It's best practice to use a virtual environment (like conda or venv) to avoid conflicts with other projects.
2.  **Install Libraries**: Open a terminal or Anaconda Prompt and install the required packages.
    ```bash
    pip install torch torchvision scikit-learn matplotlib tqdm ipywidgets Pillow
    ```
3.  **Enable Jupyter Widgets**: For the `tqdm` progress bars to display correctly in Jupyter, run this command in your terminal:
    ```bash
    jupyter nbextension enable --py widgetsnbextension
    ```

## 5. Hardware Environment

The model training and experiments were conducted on a system equipped with an **NVIDIA GeForce RTX 3070** GPU.

## 6. How to Run

1.  Ensure the file structure matches the one described in this document.
2.  Install all required libraries as listed in the "Setup and Installation" section.
3.  Open the `deeplabv3fianlversion.ipynb` file in a Jupyter environment.
4.  Verify that the `base_dir` path in the first code cell correctly points to your data folder.
5.  Run all cells in the notebook in order. The script will train the model, save the best version as `.pth`, and display the final visualization.

### Part 4: Systematic Pipeline with U-Net & U-Net++

## Overview

This project implements a comprehensive deep learning pipeline for forest dead tree segmentation using multi-spectral satellite imagery. The system compares different neural network architectures, channel configurations, and data augmentation strategies to identify the optimal approach for detecting dead trees in forest areas.

## Features

- **Multi-spectral Image Support**: Processes RGB and NRG (Near-infrared, Red, Green) satellite images
- **Multiple Channel Configurations**: RGB-only, NRG-only, RGB+NIR, and RGB+NRG
- **Advanced Neural Architectures**: U-Net and U-Net++ with ResNet34 encoder
- **Comprehensive Data Augmentation**: Four levels of augmentation strategies (none, light, medium, strong)
- **Automated Experiment Pipeline**: Systematic comparison of 16 different model configurations
- **Detailed Performance Analysis**: IoU, accuracy, precision, recall, and F1-score metrics

## Dataset Structure

The project expects the following directory structure:
```
USA_segmentation/
├── RGB_images/          # RGB satellite images (RGB_*.png)
├── NRG_images/          # NRG satellite images (NRG_*.png)
└── masks/               # Binary segmentation masks (mask_*.png)
```

## System Requirements

### Hardware Requirements
- **GPU**: CUDA-compatible GPU with at least 8GB VRAM (recommended)
- **RAM**: Minimum 16GB system RAM
- **Storage**: At least 10GB free space for models and results

### Software Requirements
- **Platform**: Google Colab (recommended) or local Python environment
- **Python**: 3.7 or higher
- **CUDA**: 11.8 (for GPU acceleration)

## Dependencies

The project uses the following major libraries:

### Deep Learning & Computer Vision
- `torch==2.0+` (PyTorch with CUDA 11.8 support)
- `torchvision`
- `segmentation-models-pytorch` (for U-Net and U-Net++ architectures)
- `albumentations` (for advanced data augmentation)

### Image Processing
- `opencv-python-headless`
- `Pillow`
- `scikit-image`
- `imageio`

### Data Science & Visualization
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

### Utilities
- `tqdm` (for progress bars)
- `wandb` (for experiment tracking - optional)

## Installation and Setup

### Option 1: Google Colab (Recommended)

1. **Open Google Colab** and create a new notebook
2. **Run the installation code** (provided in the first code segment):
```python
# Install required libraries
!pip uninstall torch torchvision torchaudio -y
!pip install --upgrade pip
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install opencv-python-headless Pillow scikit-image imageio
!pip install numpy pandas matplotlib seaborn scikit-learn tqdm
!pip install albumentations segmentation-models-pytorch wandb
```

3. **Connect Google Drive**:
```python
from google.colab import drive
drive.mount('/content/gdrive')
```

4. **Upload your dataset** to Google Drive in the following structure:
   - `MyDrive/9517/USA_segmentation/RGB_images/`
   - `MyDrive/9517/USA_segmentation/NRG_images/`
   - `MyDrive/9517/USA_segmentation/masks/`

### Option 2: Local Environment

1. **Install Python 3.7+** and CUDA 11.8
2. **Install dependencies**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python Pillow scikit-image imageio
pip install numpy pandas matplotlib seaborn scikit-learn tqdm
pip install albumentations segmentation-models-pytorch wandb
```

3. **Modify paths** in the code to point to your local dataset directory

## How to Run the Code

### Step 1: Environment Setup
Run the environment verification code to ensure all dependencies are correctly installed and GPU is available.

### Step 2: Data Exploration
Execute the data preprocessing and exploration code to:
- Verify dataset structure and file correspondence
- Analyze image properties and mask distributions
- Visualize sample images
- Split dataset into training and validation sets

### Step 3: Model Training
Run the complete training pipeline:

```python
# For full experiment pipeline (recommended)
pipeline, results = run_complete_training_pipeline(
    train_files, val_files,
    RGB_PATH, NRG_PATH, MASK_PATH,
    num_epochs=50,
    learning_rate=1e-4
)

# For quick testing (reduced epochs)
pipeline = run_quick_pipeline_test(
    train_files, val_files,
    RGB_PATH, NRG_PATH, MASK_PATH
)
```

## Experiment Configuration

The pipeline automatically runs the following experiments:

### Group 1: Channel Comparison (U-Net)
- RGB-only (3 channels)
- NRG-only (3 channels)
- RGB+NIR (4 channels)
- RGB+NRG (6 channels)

### Group 2: Architecture Comparison
- U-Net vs U-Net++ on RGB channels
- U-Net vs U-Net++ on RGB+NIR channels
- U-Net++ on RGB+NRG channels

### Group 3: Data Augmentation Comparison (U-Net++)
- Four augmentation levels: none, light, medium, strong
- Tested on RGB, RGB+NIR, and RGB+NRG configurations

## Code Structure

### Core Modules

1. **`MultiChannelColorTransform`**: Custom augmentation class for multi-spectral images
2. **`MultiChannelForestDataset`**: PyTorch dataset class supporting various channel configurations
3. **`CompleteTrainingPipeline`**: Main training pipeline management class
4. **Loss Functions**: DiceLoss and CombinedLoss (BCE + Dice)
5. **Evaluation Metrics**: IoU, accuracy, precision, recall, F1-score calculation functions

### Key Functions

- `explore_dataset_structure()`: Dataset exploration and validation
- `visualize_samples()`: Sample visualization with overlays
- `analyze_mask_distribution()`: Target area distribution analysis
- `split_dataset()`: Training/validation set splitting
- `run_complete_training_pipeline()`: Main execution function
- `run_quick_pipeline_test()`: Quick testing function

## Output and Results

The pipeline automatically generates comprehensive experiment reports and analysis:

### Saved Files
- `complete_training_results.json`: Detailed experiment results with training history
- `complete_training_summary.csv`: Summary table of all experiments
- Training logs and model checkpoints (when enabled)

### Performance Metrics
The system evaluates models using multiple metrics:
- **IoU (Intersection over Union)**: Primary evaluation metric for segmentation quality
- **Accuracy, Precision, Recall, F1-score**: Additional classification metrics
- **Training time and model parameters**: Efficiency and complexity analysis

### Automated Analysis Reports
The pipeline generates systematic comparisons across:
- **Channel Configuration Analysis**: Evaluates the contribution of different spectral bands (RGB vs NRG vs RGB+NIR vs RGB+NRG)
- **Architecture Comparison**: Quantifies performance differences between U-Net and U-Net++ architectures
- **Data Augmentation Effectiveness**: Measures the impact of different augmentation strategies on model performance
- **Overall Best Model Identification**: Automatically identifies the optimal configuration across all experiments

### Result Summary Format
The system outputs results in both detailed JSON format for further analysis and CSV format for easy viewing. The comprehensive report includes percentage improvements, statistical comparisons, and training efficiency metrics to facilitate informed model selection.

## Key Design Decisions

### Multi-Channel Processing
- Custom color augmentation that preserves NIR channel integrity
- Flexible channel configuration supporting RGB, NRG, RGB+NIR, and RGB+NRG

### Training Strategy
- Combined loss function (BCE + Dice) for better segmentation performance
- Early stopping with patience mechanism
- Learning rate scheduling based on validation loss
- Gradient clipping for training stability

### Experiment Design
- Systematic comparison across three dimensions: channels, architecture, augmentation
- Reproducible results with fixed random seeds
- Comprehensive logging and result saving

## Usage Notes

### Memory Requirements
- Batch size may need adjustment based on available GPU memory
- Default batch size: 8 (suitable for GPUs with 8GB+ VRAM)

### Training Time
- Full pipeline: ~4-6 hours on modern GPU
- Quick test: ~30-45 minutes
- Time varies based on dataset size and hardware

### Customization
- Modify `num_epochs`, `learning_rate`, and `batch_size` parameters as needed
- Add new architectures by extending the `create_model()` method
- Implement additional augmentation strategies in `_get_transforms()`

## External Libraries and Code

This project builds upon several open-source libraries:

- **segmentation-models-pytorch**: Pre-trained encoder-decoder architectures
- **albumentations**: Advanced image augmentation library
- **PyTorch**: Deep learning framework
- **scikit-learn**: Machine learning utilities for metrics and data splitting

All external dependencies are clearly listed in the installation section. The core implementation is original work developed specifically for this forest segmentation task.

## License and Attribution

This code is developed for the COMP9517 Computer Vision course group project. Please ensure proper attribution when using or modifying this code for academic or research purposes.


```python

```
