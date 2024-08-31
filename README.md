# 🚀 **GPU-Powered Object Detection and Classification with YOLOv8 and TensorFlow-Keras**

## 🌟 What is it?

This project leverages the power of GPUs and state-of-the-art deep learning models for object detection and classification. It includes scripts for setting up a training environment using YOLO (You Only Look Once) for object detection and TensorFlow for building a custom classification network. The repository offers a complete pipeline, from data preparation to training, evaluation, and deployment of object detection and classification models.

## 🎯 Why did we do it?

Our goal was to create a comprehensive example demonstrating:
- **Efficient Object Detection:** Using YOLO to quickly and accurately detect objects in images.
- **Custom Image Classification:** Building a custom image classification model using TensorFlow-Keras and MobileNetV2.
- **Leveraging GPU Acceleration:** Ensuring all operations run smoothly on a GPU to save time and increase performance.
  
## 👥 Who is this for?

This repository is perfect for:
- **Data Scientists and ML Engineers** looking to explore object detection and image classification.
- **Students and Educators** who want practical examples of deep learning in Python.
- **Developers** needing a template for creating efficient, GPU-accelerated ML models.

### 🖥️ Demos and Results

1. **YOLO Object Detection:** Train a YOLO model to detect welding defects with high accuracy.
2. **Custom Image Classifier:** Build and evaluate a custom classifier using MobileNetV2 to classify images into multiple categories.
3. **Visualizations:** Real-time training metrics, mAP scores, confusion matrices, and prediction probability plots for easy analysis.

## ⚙️ How did we do it?

### 1. **Setting up the Environment:**
   - Ensured GPU availability using `torch` and `tensorflow` checks.
   - Installed necessary libraries like `Ultralytics` for YOLO and dependencies for TensorFlow.

### 2. **Data Preparation:**
   - Organized datasets into `train`, `val`, and `test` folders.
   - Preprocessed data to be compatible with the YOLO model and custom classifiers.

### 3. **Training Object Detection Model:**
   - Utilized the YOLOv8 model for training on a custom dataset.
   - Implemented data augmentation and real-time visualization during training.

### 4. **Custom Classifier Training:**
   - Built a MobileNetV2-based custom neural network for classification.
   - Used ImageDataGenerator for data augmentation and created callbacks for saving the best model.

### 5. **Model Evaluation and Visualization:**
   - Generated detailed evaluation metrics such as mAP scores, confusion matrices, precision-recall scores.
   - Visualized results using matplotlib.

### 6. **Deployment:**
   - Created a Python class for easy deployment and prediction of new data with the trained models.

## 📘 What did we learn?

- **Efficient Data Handling:** How to organize datasets effectively for different machine learning tasks.
- **Model Training and Evaluation:** The nuances of training YOLO and custom TensorFlow models, including handling overfitting and choosing the right hyperparameters.
- **Visualization for Analysis:** Importance of real-time and post-training visualizations for understanding model performance.
- **Deployment Strategies:** Packaging models for easy deployment and usage in real-world applications.

## 🏆 Achievements

- **Achieved high mAP scores** for object detection, demonstrating robust model performance.
- **Developed a versatile pipeline** for both object detection and image classification tasks.
- **Seamlessly integrated GPU support** to accelerate training and inference.
- **Provided a detailed walkthrough and resources** to help new developers understand and replicate the results.

---
