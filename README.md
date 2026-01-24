# 🌊 Flood Prediction Using Graph Neural Networks (GNN)

## 📌 Project Description
Flooding is a major environmental and urban challenge, especially in low-lying and densely populated regions. Traditional machine learning models often fail to capture spatial relationships between different areas.

This project uses Graph Neural Networks (GNNs) to predict flood risk by modeling geographic regions as nodes in a graph and their relationships as edges. The system is trained on synthetic flood data and deployed using Streamlit for interactive flood prediction.

---

## 🧠 Core Concept
Instead of treating each area independently, this project:
- Represents regions as graph nodes
- Connects similar regions using K-Nearest Neighbors (KNN)
- Learns spatial dependencies using Graph Convolutional Networks (GCN)

---

## 🧪 Dataset (Synthetic Data)
The dataset is synthetically generated to simulate realistic flood scenarios.

Input Features:
- Rainfall (cm)
- Elevation (meters)
- Land Use (encoded)

Target Variable:
- 0 → No Flood
- 1 → Flood

Flood labels are generated using rule-based conditions combining rainfall, elevation, and land use.

---

## ⚖️ Handling Class Imbalance
SMOTE (Synthetic Minority Over-sampling Technique) is applied only on the training data to handle class imbalance and prevent model bias.

---

## 🏗️ Graph Construction
- All samples are combined into a single graph
- KNN (K = 5) is used to generate edges
- Nodes represent regions, edges represent similarity

---

## 🧠 Model Architecture

Input Features (3)

      ↓

GCN Layer (32)
      
      ↓

ReLU

      ↓

GCN Layer (16)

      ↓

ReLU

      ↓

GCN Layer (2)

      ↓

Flood / No Flood

---

## 📊 Evaluation Metrics
- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1-score

---

## 🖥️ Streamlit Web Application
The Streamlit application allows users to input rainfall, elevation, and land use to predict flood risk in real time.

---

## 🚀 How to Run the Project

1. Clone the repository:
git clone https://github.com/your-username/Major-project-flood-gnn.git

2. Navigate to project directory:
cd Major-project-flood-gnn

3. Install dependencies:
pip install -r requirements.txt

4. Train the model to generate:
flood_gnn_model.pth

5. Run the Streamlit app:
streamlit run app.py

---

## 📦 Requirements
- torch
- torch-geometric
- streamlit
- numpy
- pandas
- scikit-learn
- imblearn

---

## 👨‍🎓 Author
Ankit Verma  
B.Tech Major Project  
Domain: Graph Neural Networks | Flood Prediction
