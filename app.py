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

# -------------------- BACKGROUND IMAGE LOGIC --------------------
# We use Session State to track the current background image URL.
IMG_INITIAL = "https://images.unsplash.com/photo-1515694346937-94d85e41e6f0?q=80&w=1920&auto=format&fit=crop" 
IMG_FLOOD = "https://images.unsplash.com/photo-1547683901-f8db9f18e9a2?q=80&w=1920&auto=format&fit=crop"   
IMG_SAFE = "https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?q=80&w=1920&auto=format&fit=crop"    

if 'bg_image' not in st.session_state:
    st.session_state.bg_image = IMG_INITIAL

# -------------------- ADVANCED UI STYLING --------------------
st.markdown(
    f"""
    <style>
    /* DYNAMIC BACKGROUND IMAGE */
    .stApp {{
        background: linear-gradient(rgba(8, 18, 28, 0.85), rgba(8, 18, 28, 0.85)), 
                    url("{st.session_state.bg_image}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        transition: background-image 0.5s ease-in-out;
    }}

    /* GLASSMORPHISM CARD */
    .block-container {{
        max-width: 700px;
        margin-top: 8vh;
        padding: 4rem;
        background: rgba(31, 59, 87, 0.35); 
        backdrop-filter: blur(15px);
        border-radius: 28px;
        border: 1px solid rgba(108, 182, 255, 0.2);
        box-shadow: 0 25px 50px rgba(0,0,0,0.6);
    }}

    /* HIGH-CONTRAST NEON TITLE */
    h1 {{
        text-align: center;
        font-size: 3.5rem !important;
        font-weight: 900 !important;
        color: #ffffff !important;
        text-shadow: 0 0 15px rgba(59, 130, 246, 0.8), 0 0 30px rgba(59, 130, 246, 0.4);
        margin-bottom: 5px !important;
        letter-spacing: -1px;
        white-space: nowrap; /* Ensures title stays on one line */
    }}

    /* ONE-LINE SUBTITLE */
    .stCaption {{
        text-align: center;
        color: #6cb6ff !important;
        font-size: 1.1rem !important;
        font-weight: 500;
        letter-spacing: 0.5px;
        margin-bottom: 35px !important;
        white-space: nowrap; /* Forces the subtitle into one line */
        overflow: visible;
    }}

    /* INPUT FIELD REFINEMENT */
    .stNumberInput div[data-baseweb="input"], 
    .stSelectbox div[data-baseweb="select"] {{
        background-color: rgba(12, 30, 53, 0.9) !important;
        border: 1px solid rgba(108, 182, 255, 0.15) !important;
        border-radius: 12px !important;
    }}

    /* HOVER EFFECT ON BUTTON */
    .stButton > button {{
        width: 100%;
        background: linear-gradient(90deg, #3b82f6, #2563eb);
        border: none;
        color: white;
        padding: 0.8rem;
        font-weight: 700;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3);
        transition: 0.4s ease;
    }}

    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(59, 130, 246, 0.6);
        border: none;
        color: white;
    }}
    
    hr {{
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        margin: 30px 0;
    }}

    label p {{
        color: #e2e8f0 !important;
        font-weight: 600 !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------- MODEL DEFINITION (IDENTICAL) --------------------
class FloodGNN(torch.torch.nn.Module):
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

@st.cache_resource
def load_model():
    m = FloodGNN()
    try:
        m.load_state_dict(torch.load("flood_gnn_model.pth"))
    except:
        pass 
    m.eval()
    return m

model = load_model()

# -------------------- UI COMPONENTS --------------------
st.markdown("<h1>🌊 Flood Risk Prediction</h1>", unsafe_allow_html=True)
st.caption("AI-Based Urban Flood Detection using Graph Neural Networks")

st.divider()

# Inputs
rainfall = st.number_input("Rainfall (cm)", min_value=0.0, step=1.0)
elevation = st.number_input("Elevation (meters)", min_value=0.0, step=1.0)

land_use = st.selectbox(
    "Land Use Type",
    ["Default", "Residential", "Commercial", "Industrial", "Agricultural"]
)

st.divider()

# Placeholder for post-rerun results
prediction_placeholder = st.empty()

if st.button("Predict Flood Risk"):
    if land_use == "Default":
        st.warning("Please select a valid Land Use Type.")
        st.stop()

    land_use_idx = ["Residential", "Commercial", "Industrial", "Agricultural"].index(land_use)

    node_features = torch.tensor([[rainfall, elevation, land_use_idx]], dtype=torch.float)
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    graph_data = Data(x=node_features, edge_index=edge_index)

    output = model(graph_data)
    prediction = output.argmax(dim=1).item()

    # Update Background Image State
    if prediction == 1:
        st.session_state.bg_image = IMG_FLOOD
    else:
        st.session_state.bg_image = IMG_SAFE

    st.rerun() 

# -------------------- RENDER RESULTS --------------------
if st.session_state.bg_image == IMG_FLOOD:
    with prediction_placeholder.container():
        st.error("⚠️ High Flood Risk Detected")
        st.write("""
        - Avoid low-lying areas  
        - Ensure proper drainage  
        - Stay updated with alerts  
        - Move valuables to higher ground  
        """)
elif st.session_state.bg_image == IMG_SAFE:
    with prediction_placeholder.container():
        st.markdown("### ✅ Area is Safe")
        st.write("No immediate flood risk detected.")
