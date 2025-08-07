# chest_xray

# 📌 Chest X-Ray Classification using CNN

This project uses deep learning (Convolutional Neural Networks) to classify chest X-ray images as **Pneumonia** or **Normal**. It leverages the [Kaggle Chest X-ray dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and builds a custom CNN model from scratch for binary classification.

---

## 🧠 Objectives

- Understand and preprocess medical imaging data.
- Build and train a CNN model for image classification.
- Evaluate model performance using accuracy, precision, recall, and confusion matrix.
- Visualize predictions and model metrics.

---

## 📁 Dataset

- **Source:** [Kaggle Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Classes:** 
  - `PNEUMONIA`
  - `NORMAL`
- **Split:**
  - Training: ~5,000 images
  - Validation: ~600 images
  - Testing: ~600 images

---

## 🔧 Tech Stack

- Python
- TensorFlow / Keras
- NumPy / Pandas / Matplotlib
- Scikit-learn
- OpenCV (optional)

---

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Mansivarshney10/chest_xray.git
   cd chest_xray
2. **Install dependencies::**
   pip install -r requirements.txt
3. **Add the dataset**
   Place the chest_xray/ dataset folder in the root project directory.
4. **Run the notebook:**
   jupyter notebook chest_xray_classification.ipynb

## 📊 Model Performance

- Final Accuracy: ~XX% _(update after training)_
- Confusion Matrix: _(include if available)_
- Sample Predictions: _(optional visualization)_

---

## ✅ Key Learnings

- Image preprocessing techniques (resizing, normalization).
- CNN architecture design.
- Handling imbalanced classes.
- Evaluation metrics in medical imaging.

---

## 🧠 Future Improvements

- Hyperparameter tuning
- Use of pre-trained models (transfer learning: ResNet, VGG)
- Grad-CAM or saliency maps for explainability
- Web interface for live prediction

---

## 🧑‍💻 Author

**Mansi Varshney**  
Connect on (https://www.linkedin.com/in/mansivarshney10)

