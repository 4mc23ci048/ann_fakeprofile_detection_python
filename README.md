# ann_fakeprofile_detection_python


## Overview

This project detects whether an Instagram account is **Real** or **Fake** using Machine Learning and Deep Learning models. It uses profile-related features such as followers, following, posts, profile picture, account age, activity score, and username characteristics to classify accounts.

The application is built using **Python**, **Flask**, **Scikit-learn**, and **TensorFlow** with an easy-to-use web interface.

---

## Features

- Detects Fake and Real Instagram accounts
- Uses both Support Vector Machine (SVM) and Artificial Neural Network (ANN)
- Ensemble prediction for improved accuracy
- Flask-based web application
- Automatic feature scaling
- Displays trained model accuracies
- Simple and responsive user interface

---

## Technologies Used

- Python 3.x
- Flask
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Joblib

---

## Project Structure

```
Fake-Instagram-Detection/
│
├── app.py                    # Flask application
├── train_model.py            # Model training script
├── scaler.pkl                # Saved StandardScaler
├── svm_model.pkl             # Trained SVM model
├── ann_model.h5              # Trained ANN model
├── model_accuracies.json     # Stores model accuracies
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── static/
│   ├── style.css
│   └── images/
│
├── requirements.txt
└── README.md
```

---

## Dataset

The project uses a **synthetically generated dataset** consisting of Instagram profile features.

### Features

- Username
- Followers
- Following
- Number of Posts
- Bio Length
- Profile Picture
- External URL
- Account Age (Days)
- Activity Score

The dataset is automatically labeled using rule-based logic to generate Real and Fake account classes.

---

## Machine Learning Models

### Support Vector Machine (SVM)

- Kernel: RBF
- Probability Enabled
- Standardized Input Features

### Artificial Neural Network (ANN)

Architecture:

- Input Layer
- Dense Layer (32 neurons, ReLU)
- Dense Layer (16 neurons, ReLU)
- Output Layer (Sigmoid)

Loss Function:
- Binary Crossentropy

Optimizer:
- Adam

Epochs:
- 30

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/Fake-Instagram-Detection.git

cd Fake-Instagram-Detection
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually

```bash
pip install flask
pip install pandas
pip install numpy
pip install scikit-learn
pip install tensorflow
pip install joblib
```

---

## Train the Models

Run

```bash
python train_model.py
```

This generates

- scaler.pkl
- svm_model.pkl
- ann_model.h5
- model_accuracies.json

---

## Run the Application

```bash
python app.py
```

Open your browser

```
http://127.0.0.1:5000/
```

---

## Prediction Process

1. User enters Instagram profile details.
2. Username features are extracted.
3. Features are scaled using StandardScaler.
4. Predictions are made using:
   - SVM
   - ANN
5. Final prediction is obtained using ensemble voting.
6. Result is displayed as:

- REAL ✅
- FAKE ❌

---

## Username Feature Extraction

The project extracts two additional features from the username:

- Username Length
- Has Name (alphabetic characters > numeric characters)

These improve the model's prediction performance.

---

## Model Output

The application displays:

- Predicted Account Type
- Username
- SVM Accuracy
- ANN Accuracy

---

## Future Improvements

- Train using a real Instagram dataset
- Add Random Forest and XGBoost models
- Integrate Instagram API
- Improve feature engineering
- Deploy using Docker
- Cloud deployment using Render or Heroku
- Mobile-friendly UI

---

## Requirements

```
Python >= 3.9

Flask
TensorFlow
Scikit-learn
Pandas
NumPy
Joblib
```

---

## Author

**Saniya banu**

Machine Learning | Deep Learning | Python | Flask

---

## License

This project is intended for educational and research purposes.
