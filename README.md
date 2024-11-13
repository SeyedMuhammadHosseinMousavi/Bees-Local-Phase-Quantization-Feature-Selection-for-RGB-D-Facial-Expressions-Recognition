# Bees Local Phase Quantization Feature Selection for RGB-D Facial Expressions Recognition
# Bees LPQ Feature Selection for RGB-D Facial Expressions Recognition

This repository implements **Bees Local Phase Quantization (LPQ) Feature Selection** for **RGB-D Facial Expressions Recognition**, as outlined in our paper. We leverage the Bees Algorithm (BA) to optimize LPQ-extracted features for classifying RGB-D images of facial expressions, using SVM, KNN, Shallow Neural Network, and Ensemble Subspace KNN classifiers.

### Overview

This project uses the **Iranian Kinect Face Database (IKFDB)**, containing RGB and depth images for five emotions. Our pipeline includes preprocessing, LPQ feature extraction, Bees Algorithm (BA) feature selection, and classification. Key findings show that BA significantly boosts accuracy, especially with the Ensemble Subspace KNN classifier, reaching up to 99.8%.

### Getting Started

1. **Requirements**: MATLAB (R2021a or later), with Image Processing and Statistics Toolboxes.
2. **Install**:
   ```bash
   git clone https://github.com/your-username/Bees-LPQ-Facial-Expression-Recognition.git
   cd Bees-LPQ-Facial-Expression-Recognition

