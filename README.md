# Currency Classification System

Currency Classification System is an end-to-end deep learning project that classifies Indian currency notes into seven denominations (₹10, ₹20, ₹50, ₹100, ₹200, ₹500, ₹2000) using transfer learning with MobileNetV2, image preprocessing with OpenCV, and an interactive web application powered by Streamlit.

## Overview

This project builds a convolutional neural network (CNN) using transfer learning to accurately classify currency notes. The project consists of three main components:

1. Data Preparation and Preprocessing  
   - The dataset is organized into a folder structure where each subfolder is named according to the note type (e.g., 10Rupeenote, 20Rupeenote, 50Rupeenote, 1Hundrednote, 2Hundrednote, 5Hundrednote, 2Thousandnote).  
   - OpenCV is used to load images, convert colors from BGR to RGB, resize them to 224x224, normalize pixel values, and apply augmentation (such as random horizontal flips).

2. Model Building and Training
   - The model is built using MobileNetV2 (pre-trained on ImageNet) as a feature extractor with a custom Dense layer for seven-class classification.  
   - The model is compiled using TensorFlow and Keras and trained on the prepared data. Extensive hyperparameter tuning achieved approximately 86.4% accuracy in classifying the currency denominations.

3. Deployment  
   - The trained model is deployed as an interactive web application using Streamlit.  
   - Users can upload an image of a currency note, and the app will display the image along with the predicted denomination in real time.

## Files and Structure

The project is organized as follows:

- Data_Utilities.ipynb  
  Contains functions for:
  - Loading and preprocessing images using OpenCV.
  - Creating TensorFlow datasets from the folder-structured data.
  - A mapping dictionary (`LABEL_MAPPING`) that converts folder names (e.g., "10Rupeenote", "20Rupeenote", "50Rupeenote", "1Hundrednote", "2Hundrednote", "5Hundrednote", "2Thousandnote") into numeric labels for one-hot encoding.

- Model_Building.ipynb  
  Defines and compiles the CNN classifier using MobileNetV2 as the base model with a final Dense softmax layer for 7 classes.

- Train_Script.ipynb  
  Integrates the data utilities and model building:
  - It creates training and validation datasets (using directories, for example, located at `D:\Note Classification System\train` and `D:\Note Classification System\test`).
  - It trains the model using Keras, with callbacks like `ModelCheckpoint` and `EarlyStopping`.
  - The best model is saved (for example, as `currency_classifier.h5` or `currency_classifier.keras`).

- Streamlit_App.py  
  A Python script for deploying the model as a web application using Streamlit:
  - It loads the trained model.
  - Provides an interface to upload an image.
  - Displays the uploaded image and shows the predicted currency note denomination.

