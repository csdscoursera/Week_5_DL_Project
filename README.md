## Overview

This project implements a **CycleGAN** (Generative Adversarial Network) for **unpaired image-to-image translation**. The goal is to transform real-world photographs into the artistic style of Claude Monet paintings. The model was trained using TensorFlow within a Google Colab environment.

# Photo-to-Monet Style Transfer using CycleGAN

## Overview

This project implements a **CycleGAN** (Generative Adversarial Network) for **unpaired image-to-image translation**. The goal is to transform real-world photographs into the artistic style of Claude Monet paintings. The model was trained using TensorFlow within a Google Colab environment.

## Dataset

The model was trained on the ["I'm Something of a Painter Myself"](https://www.kaggle.com/competitions/gan-getting-started/overview) dataset from Kaggle. This dataset includes:
* ~300 Monet paintings (Domain B)
* ~7000 photographs (Domain A)

Both datasets were provided in JPEG and TFRecord formats (256x256 resolution). We utilized the **TFRecord** format for efficient training.

## Methodology

### 1. Exploratory Data Analysis (EDA)
* Visual inspection, color histogram analysis, and **t-SNE** projection confirmed a significant **domain gap** between the sharp, realistic photos and the softer, painterly Monet images.
* A key challenge identified was the **dataset imbalance** (300 Monet vs. 7000+ photos).

### 2. Model Architecture
* A **CycleGAN** framework was chosen due to the unpaired nature of the data.
* **Generators:** U-Net architecture with skip connections. `InstanceNormalization` was implemented using a `GroupNormalization(groups=-1)` workaround due to Colab environment issues.
* **Discriminators:** PatchGAN architecture to enforce realistic textures across image patches.

### 3. Training
* **Loss Functions:**
    * Adversarial Loss: Least Squares GAN (LSGAN)
    * Cycle Consistency Loss: L1 Loss (Lambda = 10)
    * Identity Loss: L1 Loss
* **Optimizer:** Adam (learning rate 2e-4, beta_1=0.5)
* **Augmentation:** Random jitter (resize & crop) and random horizontal flipping were applied to the training data.
* **Duration:** Trained for **25 epochs** (~4 hours on an L4 GPU in Colab), utilizing checkpointing to save progress.

## Results

### Quantitative
* The final **Fréchet Inception Distance (FID)** score between the generated Monet-style images and the real Monet paintings (calculated using `clean-fid`) was **114.65**. This is well below the initial target of 1000, indicating strong statistical similarity.


The model successfully learned to apply painterly textures and modify color palettes, although the fidelity to Monet's specific style varies.

## Setup and Usage

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/csdscoursera/Week_5_DL_Project](https://github.com/csdscoursera/Week_5_DL_Project)
    cd Week_5_DL_Project
    ```
2.  **Download Data:** Obtain the `gan-getting-started.zip` file from the [Kaggle competition](https://www.kaggle.com/competitions/gan-getting-started/data) page.
3.  **Google Drive:** Upload `gan-getting-started.zip` to the root directory of your Google Drive.
4.  **Open in Colab:** Open the `[Your_Notebook_Name].ipynb` file in Google Collaboratory.
5.  **Set Runtime:** Ensure the runtime type is set to use a GPU accelerator (Runtime -> Change runtime type -> GPU). An L4, V100, or A100 is recommended.
6.  **Run Notebook:** Execute the cells sequentially. The notebook will:
    * Mount your Google Drive.
    * Unzip the data.
    * Perform EDA.
    * Build the data pipeline and model.
    * Train the model (this will take several hours and requires Google Drive for checkpointing).
    * Generate results and calculate the FID score.
    * Create and download the `images.zip` submission file.

## Future Work

* Train for more epochs (50-100+) to potentially improve style refinement.
* Experiment with hyperparameters (learning rate, loss weights).
* Explore alternative generator architectures (e.g., ResNet-based).
