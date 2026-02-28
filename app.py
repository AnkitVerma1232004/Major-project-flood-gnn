# ============================================================
# STREAMLIT FRONTEND FOR FLOOD PREDICTION USING GNN
# ============================================================

import streamlit as st
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Flood Risk AI System",
    page_icon="🌊",
    layout="centered"
)


# -------------------- STYLING --------------------
st.markdown(
    """
    <style>

    /* Deep navy background (clean, modern) */
    .stApp {
        background-color: #0c1e35;
    }

    /* Remove default padding */
    section.main > div {
        padding-top: 0rem;
    }

    /* Wider and slightly higher card */
    .block-container {
        max-width: 640px;
        margin-top: 10vh;   /* Slightly above previous */
        padding: 3rem 3.5rem;
        background-color: #1f3b57;   /* Complementary slate blue */
        border-radius: 22px;
        border: 1.5px solid #2d4d6e;
        box-shadow: 0px 30px 70px rgba(0,0,0,0.45);
    }

    h1 {
        white-space: nowrap;   /* Keep title in one line */
    }

    h1, h2, h3, label, p {
        color: #f1f5f9 !important;
    }

    .stNumberInput input {
        background-color: #0c1e35 !important;
        color: white !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #0c1e35 !important;
        color: white !important;
    }

    .stButton > button {
        width: 100%;
        background-color: #3b82f6;
        color: white;
        font-weight: bold;
        border-radius: 14px;
        padding: 0.8em 1em;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# -------------------- MODEL DEFINITION --------------------
class FloodGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn_layer1 = GCNConv(3, 32)
        self.gcn_layer2 = GCNConv(32, 16)
        self.gcn_layer3 = GCNConv(16, 2)

    def forward(self, data):
        x = self.gcn_layer1(data.x, data.edge_index)
        x = F.relu(x)
        x = self.gcn_layer2(x, data.edge_index)
        x = F.relu(x)
        x = self.gcn_layer3(x, data.edge_index)
        return x


# -------------------- LOAD MODEL --------------------
model = FloodGNN()
model.load_state_dict(torch.load("flood_gnn_model.pth"))
model.eval()


# -------------------- UI --------------------
st.title("🌊 Flood Risk Prediction")
st.caption("AI-Based Urban Flood Detection using GNN")

st.divider()

rainfall = st.number_input("Rainfall (cm)", min_value=0.0, step=1.0)
elevation = st.number_input("Elevation (meters)", min_value=0.0, step=1.0)

land_use = st.selectbox(
    "Land Use Type",
    ["Default", "Residential", "Commercial", "Industrial", "Agricultural"]
)

st.divider()

if st.button("Predict Flood Risk"):

    if land_use == "Default":
        st.warning("Please select a valid Land Use Type.")
        st.stop()

    land_use = ["Residential", "Commercial", "Industrial", "Agricultural"].index(land_use)

    node_features = torch.tensor([[rainfall, elevation, land_use]], dtype=torch.float)
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)

    graph_data = Data(x=node_features, edge_index=edge_index)

    output = model(graph_data)
    prediction = output.argmax(dim=1).item()

    if prediction == 1:
        st.error("⚠️ High Flood Risk Detected")
        st.write("""
        - Avoid low-lying areas  
        - Ensure proper drainage  
        - Stay updated with alerts  
        - Move valuables to higher ground  
        """)
    else:
        st.success("✅ Area is Safe")
        st.write("No immediate flood risk detected.")