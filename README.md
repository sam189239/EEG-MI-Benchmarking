# EEG Motor Movement/Imagery Signal Classification: Benchmark

This repository contains code and resources for the classification of EEG motor movement/imagery signals using various deep learning models. The project benchmarks the effectiveness and robustness of different models, specifically for the BCI Competition IV 2A dataset.

[YouTube (Report/Demo)](https://youtu.be/-uwCrgJB880)

- [EEG Motor Movement/Imagery Signal Classification](#eeg-motor-movementimagery-signal-classification)
  - [Project Overview](#project-overview)
  - [Dataset](#dataset)
  - [Methods and Models](#methods-and-models)
    - [Pre-processing](#pre-processing)
    - [Data Augmentation](#data-augmentation)
    - [Robustness Testing](#robustness-testing)
  - [Results](#results)
  - [Limitations and Future Work](#limitations-and-future-work)

## Project Overview
This project evaluates multiple deep learning techniques to classify motor imagery tasks using EEG signals, aiming to support advancements in Brain-Computer Interface (BCI) technologies. The primary objectives include:
1. Benchmarking deep learning models for EEG signal classification.
2. Testing model robustness with noise to simulate real-world conditions.
3. Exploring the impact of data augmentation on model generalization.

## Dataset
We utilize the **BCI Competition IV 2A** dataset, which includes EEG data from 9 subjects performing 4 different motor imagery tasks:
- **Left Hand**
- **Right Hand**
- **Both Feet**
- **Tongue**

The data was pre-processed with band-pass filtering (0.5-100 Hz) and sampled at 250 Hz.

## Methods and Models
The following deep learning models were benchmarked:
- **EEGNet** (v1, v4): Convolutional model optimized for EEG data.
- **ShallowConvNet** and **DeepConvNet**: CNN models with varying complexities.
- **EEG Conformer**: Combines convolutional and attention layers.
- **ATCNet**: Integrates convolutional, attention, and temporal convolutional layers.
- **EEG-ITNet**: Utilizes inception and temporal convolution for robust performance.

### Pre-processing
- Band-pass filtering (4-38 Hz)
- Exponential Moving Standardization
- Common Spatial Pattern (CSP) for feature extraction

### Data Augmentation
Frequency shifting and sign flipping transformations were applied to improve model generalization.

### Robustness Testing
Models were evaluated for robustness by introducing Gaussian noise at SNR levels of 8 dB and 15 dB.

## Results
- **Best-performing model**: ATCNet with an accuracy of 67.82% and an F1 score of 0.6751.
- **Top models under noise**: ShallowFBCSPNet and EEG Conformer, showing minimal performance drop at lower SNR levels.
- Data augmentation improved performance for most models, particularly ATCNet.

## Limitations and Future Work
- **Compute Limitations**: Constraints restricted testing for certain complex models like FusionNet.
- **Dataset Size**: Small sample size (9 subjects) limits generalizability.
- **Future Directions**:
  - Extend to other EEG datasets.
  - Explore advanced embedding techniques for feature extraction.
  - Test robustness with additional noise/distortion types.
