# Image Classification: Cats vs Dogs

This project demonstrates how to build and train a Convolutional Neural Network (CNN) to classify images of cats and dogs using the Kaggle Cats and Dogs dataset. The project uses TensorFlow and Keras for model building and training.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Setup](#setup)
- [Training the Model](#training-the-model)
- [Evaluating the Model](#evaluating-the-model)
- [Making Predictions](#making-predictions)
- [Acknowledgements](#acknowledgements)

## Project Overview

The objective of this project is to classify images of cats and dogs using a CNN. The project involves the following steps:
1. Load and preprocess the dataset.
2. Build a CNN model.
3. Train the model on the training dataset.
4. Validate the model on the validation dataset.
5. Save the trained model.
6. Use the saved model to make predictions on new images.

## Dataset

The dataset used in this project is the [Kaggle Cats and Dogs dataset](https://www.microsoft.com/en-us/download/details.aspx?id=54765) provided by Microsoft. The dataset consists of 25,000 images of cats and dogs.

## Setup

To set up the project, follow these steps:

1. **Clone the repository**:
    ```sh
    git clone https://github.com/ManhCang/image-classification.git
    cd image-classification
    ```

2. **Create a virtual environment**:
    ```sh
    python -m venv myenv
    ```

3. **Activate the virtual environment**:
    - On Windows:
        ```sh
        myenv\Scripts\activate
        ```
    - On macOS and Linux:
        ```sh
        source myenv/bin/activate
        ```

4. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

5. **Download the dataset** and place it in the `data` directory with the following structure:
    ```
    data/
        train/
            Cat/
            Dog/
        validation/
            Cat/
            Dog/
    ```

## Training the Model

To train the model, run the following command:
```sh
python main.py
```
This will load the dataset, preprocess the images, build the CNN model, and train the model. The trained model will be saved as cats_and_dogs_classifier.h5.

## Evaluating the Model

The training process will output training and validation accuracy and loss for each epoch. After training, the model's performance can be visualized using the generated plots of accuracy and loss.

## Acknowledgements
This project uses the Kaggle Cats and Dogs dataset provided by Microsoft.
Thanks to TensorFlow and Keras for providing the tools to build and train the CNN model.


