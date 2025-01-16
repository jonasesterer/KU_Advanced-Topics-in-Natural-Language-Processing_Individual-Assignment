# Transformer Training Project

Welcome to the Transformer Training Project! This package provides all the necessary scripts to set up your environment, train transformer models on the [SCAN dataset](https://github.com/brendenlake/SCAN), and evaluate their performance with ease.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Setup](#setup)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Evaluating the Model](#evaluating-the-model)
- [License](#license)

## Introduction

This project leverages transformer architectures to perform sequence-to-sequence tasks on the SCAN dataset. The SCAN dataset is designed to test the compositional learning capabilities of models by providing a set of navigation commands paired with their corresponding action sequences. With streamlined scripts for setup, training, and evaluation, you can quickly get started and adapt the models to your specific needs.

## Prerequisites

Before getting started, ensure you have the following installed on your system:

- **Operating System**: Linux or Windows (with a bash-compatible shell like Git Bash or WSL)
- **Python**: Version 3.7 or higher
- **Zip Utility**: To extract the provided zip file

## Installation

For the Individual part of the Assignment, I did all my computations in Google Colab as I my computer was unable to handle such computation loads and does not have a GPU suitable for fine-tuning of a relatively large transformer. Therefore, I suggest using the attached .ipynb files to run the code in Google Colab. There are three files: KU_ATNLP_Group.ipynb (for running the group results), KU_ATNLP_Individual_t5_1.ipynb (for running the code for the first experiment with fine-tuned T5), and KU_ATNLP_Individual_t5_2.ipynb (for running the second experiment with fine-tuned T5). To run the .ipynb files, create a GitHub repository (adjust the name in the .ipynb file) and a folder in Google Drive (if implemented for this notebook: adjust the path in the code in the first "real" code cell.). Then simply run through all code cells to get the results downloaded and potentially uploaded to your Google Drive (ensure enough computing units). I suggest the use of the L4 GPU which has a nice balance of speed and cost. 

Otherwise, it is also possible to run it via the bash files as described below. The first bash files (train.sh and evaluate.sh) are for running the goup results while train_me.sh and evaluate_me.sh are for running my individual adaptions.

### Setup

Follow the steps below to set up the project environment:

1. **Download and Extract the Project**

   - **Download**: Obtain the project zip file from the provided source (e.g., email, download link).

   - **Extract**: Unzip the downloaded file to your desired directory.

     - **Using Command Line:**

       - **Linux:**

         ```bash
         unzip transformer-training-project.zip
         cd transformer-training-project
         ```

       - **Windows:**

         Use a bash-compatible shell like Git Bash or WSL, then run:

         ```bash
         unzip transformer-training-project.zip
         cd transformer-training-project
         ```

     - **Using GUI:**

       - Right-click the zip file and select **"Extract All..."**, then navigate to the extracted folder.

2. **Run the Setup Script**

   Execute the setup script corresponding to your operating system to install all necessary dependencies.

   - **For Linux:**

     ```bash
     bash linux_setup.sh
     ```

   - **For Windows:**

     ```bash
     bash windows_setup.sh
     ```

   > **Note:** Ensure you have the necessary permissions to execute the scripts. You might need to modify execution permissions accordingly:

   ```bash
   chmod +x linux_setup.sh windows_setup.sh
   ```

## Usage: Group implementation

After setting up the environment, you can proceed to train and evaluate the transformer models on the SCAN dataset.

### Training the Model

To train the transformer models, run the `train.sh` script:

- **Linux:**

  ```bash
  bash train.sh
  ```

- **Windows:**

  Open your bash-compatible shell (e.g., Git Bash or WSL) and run:

  ```bash
  bash train.sh
  ```

This script will initiate the training process on the SCAN dataset. You can monitor the progress through the console output. Training parameters and configurations can be adjusted within the script as needed.

### Evaluating the Model

Once training is complete, evaluate the model's performance by executing the `evaluate.sh` script:

- **Linux:**

  ```bash
  bash evaluate.sh
  ```

- **Windows:**

  Open your bash-compatible shell (e.g., Git Bash or WSL) and run:

  ```bash
  bash evaluate.sh
  ```

This script will generate evaluation plots and metrics, providing insights into the model's effectiveness on the SCAN dataset. The plots will be saved in the designated `results` directory for your review.

## Usage: Individual implementation

After setting up the environment, you can proceed to train and evaluate the transformer models on the SCAN dataset.

### Training the Model

To train the transformer models, run the `train.sh` script:

- **Linux:**

  ```bash
  bash train_me.sh
  ```

- **Windows:**

  Open your bash-compatible shell (e.g., Git Bash or WSL) and run:

  ```bash
  bash train_me.sh
  ```

This script will initiate the training process on the SCAN dataset. You can monitor the progress through the console output. Training parameters and configurations can be adjusted within the script as needed.

### Evaluating the Model

Once training is complete, evaluate the model's performance by executing the `evaluate.sh` script:

- **Linux:**

  ```bash
  bash evaluate_me.sh
  ```

- **Windows:**

  Open your bash-compatible shell (e.g., Git Bash or WSL) and run:

  ```bash
  bash evaluate_me.sh
  ```

This script will generate evaluation plots and metrics, providing insights into the model's effectiveness on the SCAN dataset. The plots will be saved in the designated `results` directory for your review.

## License

This project is licensed under the [MIT License](LICENSE).
