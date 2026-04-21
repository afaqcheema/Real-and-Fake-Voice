<img width="1282" height="440" alt="image" src="https://github.com/user-attachments/assets/e59509a7-aef5-4257-8b70-91a9fd0c5eff" />Project Overview
This project implements a machine learning pipeline to detect fake (spoofed) and real (bonafide) audio samples using the ASVspoof 2019 dataset.
The system extracts MFCC features and mel-spectrograms from audio files and trains both traditional machine learning models
(Logistic Regression, Random Forest) and deep learning models (CNN) for binary classification.

Dataset
Source: ASVspoof 2019 Dataset (LA and PA protocols)
Classes: Bonafide (real) and Spoof (fake)
Original Size: 79,380 samples (highly imbalanced)
Balanced Subset: 15,960 samples (7,980 per class)

Features
1D Features (Traditional ML)
40 MFCC coefficients per audio sample
Standardized using StandardScaler
Suitable for Random Forest and Logistic Regression

2D Features (Deep Learning)
Mel-spectrograms (128 × 94 pixels)
Log-scaled power spectrograms
Input shape: (128, 94, 1) for CNN

Models Implemented
Baseline Models
Logistic Regression - 81.42% accuracy

Random Forest - 89.22% accuracy
Optimized Model
Tuned Random Forest - 90.13% accuracy
Hyperparameters optimized via RandomizedSearchCV
Parameters: n_estimators=500, max_depth=40, min_samples_split=5, class_weight='balanced'

Deep Learning Model
CNN - 96%+ accuracy
Architecture: 3 Conv2D layers with BatchNormalization
Dense layers with Dropout (0.4)
Adam optimizer with learning rate scheduling
Early stopping to prevent overfitting

Project Structure
├── Data Loading & Preprocessing
│   ├── Download ASVspoof 2019 dataset via kagglehub
│   ├── Load LA and PA protocol files
│   ├── Create balanced dataset (50/50 split)
│   └── Organize files into real/fake folders
│
├── Feature Extraction
│   ├── 1D MFCC features (40 coefficients)
│   └── 2D Mel-spectrograms (128×94)
│
├── Model Training
│   ├── Logistic Regression (baseline)
│   ├── Random Forest (baseline + tuned)
│   └── CNN (deep learning)
│
└── Prediction Demo
    └── Upload and test custom audio files

    Version
kagglehub>=0.3.13
pandas>=2.0.0
numpy>=1.22.0
librosa>=0.11.0
scikit-learn>=1.6.0
matplotlib>=3.5.0
seaborn>=0.12.0
tensorflow>=2.15.0
tqdm>=4.65.0
joblib>=1.3.0
