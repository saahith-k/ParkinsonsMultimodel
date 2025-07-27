# Parkinson’s Disease Multimodal Detection AI


## Overview

This project implements a **state-of-the-art multimodal AI pipeline** for Parkinson’s disease detection by combining:

- **Handwriting Image Analysis** (using EfficientNet deep learning model)  
- **Voice Signal Analysis** (using an ensemble of classical ML models on extracted voice features)

The fusion of these two modalities results in robust and clinically promising diagnostic predictions, achieving >85% accuracy on voice and up to 96% accuracy on handwriting data.

The system is deployed as an **easy-to-use Streamlit web app**, where users upload a handwriting sample and a voice recording, and receive a final fused diagnosis with confidence metrics.

---

## Features

- End-to-end multimodal Parkinson’s detection pipeline  
- Advanced feature extraction from voice signals, including jitter, shimmer, and MFCC derivatives  
- Fine-tuned EfficientNet model for handwriting image classification  
- Class imbalance handling with SMOTE  
- Ensemble voting classifier combining RandomForest, KNN, and MLP for voice data  
- Weighted fusion of handwriting and voice predictions  
- User-friendly Streamlit interface for easy uploads and instant results  
- Models and preprocessing rigorously tested and saved for reproducibility  

---

## Folder Structure

ParkinsonsMultimodel/
├── app.py # Streamlit app
├── handwriting_parkinsons_96acc.keras # Trained handwriting model
├── voice_ensemble_model.joblib # Trained voice ML ensemble
├── voice_scaler.joblib # Scaler for voice features
├── requirements.txt # Python dependencies
├── README.md # This file
└── assets/ # Optional: images, logos, documentation assets


---

## Getting Started

### Prerequisites

- Python 3.7+
- pip package manager

### Installation

1. Clone the repository:

git clone https://github.com/saahith-k/ParkinsonsMultimodel.git
cd ParkinsonsMultimodel


2. Install dependencies:

pip install -r requirements.txt


---

## Running the App Locally

Start the Streamlit app:

streamlit run app.py


A browser window will open with the app interface where you can:

1. Upload a handwriting image (PNG/JPG) of the patient’s writing.  
2. Upload a voice recording file (.wav) of the patient speaking or sustaining a vowel.  
3. Click the "Analyze" button to get the fused Parkinson’s disease likelihood prediction and confidence scores from both modalities.

---

## Model Details

### Handwriting Model

- Architecture: EfficientNetB0 pre-trained on ImageNet and fine-tuned on handwriting dataset  
- Input size: 256x256 RGB images  
- Achieved ~96% accuracy on handwriting classification  

### Voice Model

- Features: MFCC (incl. delta and delta-delta), jitter, shimmer, and Harmonics-to-Noise Ratio (HNR)  
- Classifier: Ensemble of RandomForest, KNN, and Multi-Layer Perceptron (MLP)  
- Data balancing: SMOTE oversampling to handle class imbalance  
- Achieved >85% cross-validated accuracy on Parkinson’s voice dataset  

### Fusion

- Weighted average of handwriting and voice model probabilities (default weights: 65% handwriting, 35% voice)  
- Optionally extendable to meta-classifier fusion for improved performance with paired validation data  

---

## Notes and Disclaimer

- This app is for research and educational purposes only and **does not replace professional medical diagnosis.**  
- Accuracy depends on quality and representativeness of input samples.  
- Always consult healthcare professionals for clinical decisions.  

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for bug fixes, feature requests, or improvements.

---

## License

Specify your license here, e.g., MIT License.  
(If you want to make it open source: add LICENSE file and here note its terms.)

---


*Thank you for exploring this Parkinson’s AI detection project!*

