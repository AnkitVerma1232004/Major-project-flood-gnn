# ============================================================
# STREAMLIT FRONTEND FOR FLOOD PREDICTION USING GNN
# ============================================================

import streamlit as st
import torch
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


# -------------------- GNN MODEL DEFINITION --------------------
# MUST be same as training model
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


# -------------------- LOAD TRAINED MODEL --------------------

model = FloodGNN()
model.load_state_dict(torch.load("flood_gnn_model.pth"))
model.eval()   # evaluation mode


# -------------------- STREAMLIT UI --------------------

st.title("🌊 Flood Prediction System (GNN)")
st.write("Enter area details to predict flood risk")

# User inputs
rainfall = st.number_input("Rainfall (cm)", min_value=0.0, step=1.0)
elevation = st.number_input("Elevation (meters)", min_value=0.0, step=1.0)
land_use = st.selectbox(
    "Land Use Type",
    options=[0, 1, 2, 3],
    format_func=lambda x: f"Land Use {x}"
)


# -------------------- PREDICTION LOGIC --------------------

if st.button("Predict Flood"):

    # Create node feature tensor
    node_features = torch.tensor(
        [[rainfall, elevation, land_use]],
        dtype=torch.float
    )

    # Single-node dummy graph (required by GNN)
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)

    graph_data = Data(
        x=node_features,
        edge_index=edge_index
    )

    # Run model prediction
    output = model(graph_data)
    prediction = output.argmax(dim=1).item()

    # Display result
    if prediction == 1:
        st.error("⚠️ Flood Risk Detected")
    else:
        st.success("✅ No Flood Risk")
