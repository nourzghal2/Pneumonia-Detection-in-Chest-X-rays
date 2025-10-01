# Medical Image Binary Classification with CNNs

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](CONTRIBUTING.md)

A comprehensive deep learning project for classifying chest X-ray images into NORMAL and PNEUMONIA categories using Convolutional Neural Networks.

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Features](#-features)
- [Results](#-results)
- [Installation & Setup](#️-installation--setup)
- [Usage](#-usage)
- [Model Evaluation](#-model-evaluation)
- [Technical Highlights](#-technical-highlights)
- [Clinical Relevance](#-clinical-relevance)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)
- [Acknowledgments](#-acknowledgments)
- [References](#-references)

## 🏥 Project Overview

This project implements a complete machine learning pipeline for medical image classification, specifically targeting chest X-ray diagnosis. It compares two different CNN approaches:
- **Transfer Learning**: Using pre-trained VGG16 model
- **Custom CNN**: Built-from-scratch architecture optimized for medical imaging

## 📊 Dataset

The project uses the Chest X-Ray Images (Pneumonia) dataset containing:
- **Training Set**: 5,216 images (1,341 NORMAL, 3,875 PNEUMONIA)
- **Validation Set**: 16 images (8 NORMAL, 8 PNEUMONIA) 
- **Test Set**: 624 images (234 NORMAL, 390 PNEUMONIA)

**Key Characteristics:**
- Significant class imbalance (1:2.9 ratio)
- Variable image dimensions
- Grayscale chest X-ray images

## 🚀 Features

### Data Analysis & Preprocessing
- **Comprehensive EDA**: Dataset distribution analysis, image property analysis, pixel intensity studies
- **Data Augmentation**: Rotation, shifting, shearing, zooming, and horizontal flipping
- **Class Imbalance Handling**: Strategic augmentation and weighted training

### Model Architectures

#### 1. VGG16 Transfer Learning Model
```
- Base: Pre-trained VGG16 (frozen layers)
- Classification Head: Flatten → Dense(512) → Dropout → Dense(1)
- Parameters: ~15M
- Activation: ReLU → Sigmoid
```

#### 2. Custom CNN Model
```
- 4 Convolutional Blocks (32→64→128→256 filters)
- Batch Normalization after each conv layer
- MaxPooling and Dropout for regularization
- Dense layers: 512 → 256 → 1
- Parameters: ~8M
```

### Advanced Features
- **Hyperparameter Tuning**: Grid search optimization
- **Model Interpretability**: Grad-CAM visualizations
- **Comprehensive Evaluation**: ROC curves, confusion matrices, classification reports
- **Model Comparison**: Side-by-side performance analysis

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| **VGG16 Transfer Learning** | 94.2% | 95.1% | 96.4% | 95.7% | 0.982 |
| **Custom CNN** | 91.8% | 92.3% | 94.1% | 93.2% | 0.967 |

### Key Insights
- Both models achieved excellent performance for medical image classification
- VGG16 transfer learning showed superior overall performance
- Custom CNN demonstrated competitive results with fewer parameters
- Grad-CAM visualizations confirmed clinically relevant feature detection

## 🛠️ Installation & Setup

### Prerequisites
```bash
Python 3.7+
TensorFlow 2.x
```

### Required Libraries
```bash
pip install tensorflow
pip install numpy pandas matplotlib seaborn
pip install scikit-learn
pip install pillow opencv-python
pip install tqdm
```

### For Kaggle Environment
```python
# Dataset path configuration
base_dir = '/kaggle/input/chest_xray'
```

### For Local Environment
```python
# Update paths to your local dataset location
base_dir = 'path/to/your/chest_xray_dataset'
```

## 📁 Project Structure

```
deep-learning-medical-classification/
│
├── deep-learning.ipynb          # Main Jupyter notebook
├── README.md                    # Project documentation
├── output/                      # Generated outputs
│   ├── models/
│   │   ├── best_model.keras
│   │   └── final_custom_model.keras
│   ├── plots/
│   │   ├── dataset_distribution.png
│   │   ├── roc_comparison.png
│   │   ├── confusion_matrix_comparison.png
│   │   └── gradcam_visualizations.png
│   └── results/
│       └── performance_metrics.csv
│
└── data/                        # Dataset (not included)
    ├── train/
    ├── val/
    └── test/
```

## 🔧 Usage

### Running the Complete Pipeline

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/medical-image-classification.git
cd medical-image-classification
```

### Dataset Download
The dataset used in this project is the **Chest X-Ray Images (Pneumonia)** dataset. You can download it from:
- [Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- [Original Paper Dataset](https://data.mendeley.com/datasets/rscbjbr9sj/2)

2. **Set up the dataset**
   - Download the Chest X-Ray dataset
   - Update dataset paths in the notebook

3. **Run the notebook**
```bash
jupyter notebook deep-learning.ipynb
```

### Key Code Sections

#### Data Loading and Preprocessing
```python
# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
```

#### Model Creation
```python
# VGG16 Transfer Learning
base_model = VGG16(include_top=False, weights='imagenet')
model = create_base_model(base_model)

# Custom CNN
custom_model = create_custom_model()
```

#### Training with Callbacks
```python
callbacks = [
    ModelCheckpoint('best_model.keras', save_best_only=True),
    EarlyStopping(patience=10),
    ReduceLROnPlateau(factor=0.5, patience=5)
]
```

## 📊 Model Evaluation

### Performance Metrics
- **Accuracy**: Overall classification correctness
- **Precision**: Ratio of true positives to predicted positives
- **Recall**: Ratio of true positives to actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the receiver operating characteristic curve

### Visualization Tools
- Confusion matrices for both models
- ROC curve comparisons
- Training history plots
- Grad-CAM heatmaps for model interpretability

## 🔬 Technical Highlights

### Hyperparameter Optimization
- **Learning Rate**: 0.0001 (optimal)
- **Batch Size**: 32 (optimal)
- **Dropout Rate**: 0.5 (optimal)
- **Epochs**: 30 with early stopping

### Model Interpretability
- **Grad-CAM**: Gradient-based Class Activation Mapping
- Visualizes which regions the model focuses on
- Confirms clinically relevant feature detection

### Handling Class Imbalance
- Data augmentation techniques
- Stratified sampling
- Performance evaluation on minority class

## 🎯 Clinical Relevance

This project demonstrates:
- **High Sensitivity**: Effective pneumonia detection (96.4% recall)
- **High Specificity**: Minimal false positives for normal cases
- **Interpretability**: Visual evidence of decision-making process
- **Efficiency**: Fast inference suitable for clinical workflows

## 🚀 Future Improvements

1. **Dataset Expansion**
   - Multi-class classification (different pneumonia types)
   - Larger, more diverse datasets
   - Cross-institutional validation

2. **Advanced Architectures**
   - ResNet, DenseNet, EfficientNet comparisons
   - Ensemble methods
   - Attention mechanisms

3. **Clinical Integration**
   - DICOM format support
   - Real-time inference API
   - Integration with PACS systems

4. **Model Robustness**
   - Adversarial training
   - Cross-validation strategies
   - Uncertainty quantification

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### How to Contribute
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Issues
Found a bug or have a feature request? Please open an issue with:
- A clear description of the problem/feature
- Steps to reproduce (for bugs)
- Your environment details

## 📞 Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **LinkedIn**: [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)
- **GitHub**: [@yourusername](https://github.com/yourusername)

## 🙏 Acknowledgments

- Chest X-Ray Images (Pneumonia) dataset creators
- TensorFlow and Keras development teams
- Medical imaging research community
- Open source contributors

## 📚 References

1. Kermany, D. S., et al. (2018). Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. Cell, 172(5), 1122-1131.
2. Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition. arXiv preprint arXiv:1409.1556.
3. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. ICCV.

---

⭐ **Star this repository if you found it helpful!** ⭐