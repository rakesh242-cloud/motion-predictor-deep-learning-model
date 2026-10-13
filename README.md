# 🎯 Motion Pattern Prediction using Transformer

This project focuses on **human motion pattern prediction** using pose estimation and a Transformer-based deep learning model.
It processes videos, extracts human keypoints using MediaPipe, and predicts motion classes.

---

## 📌 Features

* 🎥 Video-based motion analysis (UCF-101 dataset)
* 🧍 Pose estimation using MediaPipe
* 🤖 Transformer-based deep learning model
* 📊 Evaluation using confusion matrix & classification report
* 📈 Training visualization (accuracy/loss graphs)
* 🎬 Motion sequence visualization (skeleton animation)

---

## 📁 Project Structure

```
motion-pattern-prediction/
│
├── model.py                # Transformer model
├── train.py                # Training script
├── test.py                 # Testing script
├── evaluate.py             # Evaluation (confusion matrix, report)
├── visualize.py            # Skeleton animation
├── handledataset.py        # Dataset preprocessing
├── videcount.py            # Video utilities
├── filter.py               # Data filtering
│
├── requirements.txt
├── README.md
│
├── confusion_matrix.png
├── training_graph.png
│
└── (External via Drive)
    ├── UCF-101 dataset
    ├── Processed dataset (.npy)
    ├── Saved models (.pth)
```

---

## ⚙️ Requirements

* Python **3.10 **
* Install dependencies:

```
pip install -r requirements.txt
```

---

## 📦 Dataset & Models Download

Download all required files from Google Drive:

[GitHub Link](https://github.com/kairo786/attention-based-Transformer-models-for-motion-pattern-analysis)

[Drive Link](https://drive.google.com/drive/folders/1n7W1V5KksnHhLEg8NcDRZ1csSFvgXQJr?usp=sharing)

Contents:

* Original Dataset (UCF-101)
* Processed `.npy` files
* Trained models (`.pth`)

After downloading, place them like:

```
project_root/
├── UCF-101/
├── X_train.npy
├── y_train.npy
├── X_test.npy
├── y_test.npy
├── best_model.pth
```

---

## 🚀 How to Run

### 1️⃣ Dataset Preparation

If you want to regenerate dataset:

```
python handledataset.py
```

This will:

* Read videos from UCF-101
* Extract pose keypoints
* Save `.npy` files

---

### 2️⃣ Training

```
python train.py
```

Output:

* Saved model (`.pth`)
* Training graph (`training_graph.png`)

---

### 3️⃣ Testing

```
python test.py
```

---

### 4️⃣ Evaluation

```
python evaluate.py
```

Output:

* Confusion Matrix
* Classification Report

---

### 5️⃣ Visualization

```
python visualize.py
```

Shows:

* Skeleton motion animation from sequences
* ![Motion Demo](sample.gif)
---

## 🧠 Model Details

* Architecture: Transformer-based model (`MotionTransformer`)
* Input: Sequence of pose keypoints
* Output: Motion class prediction

---

## 📊 Results

* Accuracy: ~93%

* Confusion Matrix:

* Training Graph:

---

## ⚠️ Notes

* MediaPipe requires Python 3.10
* Large files are not included in the repo
* Use GPU (if available) for faster training

---

## 📌 Future Improvements

* Real-time prediction
* WebRTC integration

---

## 👨‍💻 Author

**Rakesh**
EP, IIT Patna

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
