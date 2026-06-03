import streamlit as st
import os
import cv2
import requests
from ultralytics import YOLO
from PIL import Image
import numpy as np

# =====================================================================
# 1. WEB CORE & ENVIRONMENT SETTINGS
# =====================================================================
st.set_page_config(page_title="NutriVision", layout="centered")

# Environment paths
MODEL_PATH = "D:/CV_Final_Project/runs/detect/train-2/weights/best.pt"
BASE_URL = "https://api.nal.usda.gov/fdc/v1"

# SECRETS HANDSHAKE: Pulls your credential token seamlessly from your local .streamlit/secrets.toml
API_KEY = st.secrets["USDA_API_KEY"]

# Precise database query lookup strings matching official Foundation/Survey records
FRUIT_QUERY_MODIFIER = {
    "Apple": "Apples, red delicious, with skin, raw",
    "Banana": "Bananas, ripe and slightly ripe, raw",
    "Orange": "Oranges, raw, navels",
    "Grape": "Grapes, green, seedless, raw",
    "Pineapple": "Pineapple, raw",
    "Watermelon": "Watermelon, seedless, flesh only, raw"
}

@st.cache_resource
def load_yolo_model():
    """Loads the model once and caches it to keep inferencing fast."""
    return YOLO(MODEL_PATH)

# Initialize the computer vision layers
model = load_yolo_model()


# =====================================================================
# 2. USDA NETWORKING HANDSHAKE ENDPOINT
# =====================================================================
def fetch_nutrition(fruit_name):
    """Communicates securely with the US Government database from the backend."""
    search_term = FRUIT_QUERY_MODIFIER.get(fruit_name, f"{fruit_name} raw")
    target_url = f"{BASE_URL}/foods/search?api_key={API_KEY}"
    
    params = {
        "query": search_term,
        "pageSize": 1,
        "dataType": ["Foundation", "Survey (FNDDS)"]
    }
    
    try:
        response = requests.get(target_url, params=params)
        if response.status_code == 200:
            data = response.json()
            foods = data.get("foods", [])
            if foods:
                return foods[0]
    except Exception as e:
        st.error(f"Failed to query database ecosystem: {str(e)}")
    return None


# =====================================================================
# 3. STREAMLIT APPLICATION VIEW INTERFACE
# =====================================================================
st.title("Food Classification & Insight System")
st.write("Upload an image of a fruit to instantly run YOLOv8 object detection and fetch verified USDA nutrition facts.")

# Browser local image uploading component widget
uploaded_file = st.file_uploader("Choose a target fruit image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Load the uploaded file buffer into a PIL Image container
    raw_image = Image.open(uploaded_file)
    
    # 2. Force the format to standard 3-channel RGB to drop Alpha/transparency channels
    rgb_image = raw_image.convert("RGB")
    img_array_rgb = np.array(rgb_image)
    
    # 🔥 THE FIX: Swap Streamlit's RGB matrix into OpenCV's BGR matrix before YOLO sees it!
    # This guarantees the model sees real colors, allowing it to detect the stock orange easily.
    img_array_bgr = cv2.cvtColor(img_array_rgb, cv2.COLOR_RGB2BGR)
    
    st.info("Running object detection inference...")
    
    # Pass the corrected BGR array directly to YOLO
    results = model.predict(source=img_array_bgr, conf=0.25, imgsz=640, verbose=False)
    result_object = results[0]
    
    # Extract unique classes directly from bounding box layers
    detected_classes = []
    if len(result_object.boxes) > 0:
        for box in result_object.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            if class_name not in detected_classes:
                detected_classes.append(class_name)
                
    # YOLO's plotting function natively returns a BGR image array
    annotated_img_bgr = result_object.plot()
    
    # 🔥 THE SECOND FIX: Swap the plotted BGR image back to RGB for Streamlit to render correctly.
    annotated_img_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)
    
    # Generate structural layout split columns for clean presentation display mapping
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Result")
        # We explicitly pass our corrected RGB image to the web browser
        st.image(annotated_img_rgb, channels="RGB", use_container_width=True)
        
    with col2:
        st.subheader("📋 USDA Nutrition Facts")
        
        if detected_classes:
            st.success(f"Spotted: {', '.join(detected_classes)}")
            
            # Loop through unique objects localized without redundancy
            for fruit in detected_classes:
                food_profile = fetch_nutrition(fruit)
                
                if food_profile:
                    st.markdown(f"**🔬 {food_profile.get('description')} (Per 100g)**")
                    
                    # Parse individual rows inside the macro data payload arrays
                    for nutrient in food_profile.get("foodNutrients", []):
                        name = nutrient.get("nutrientName")
                        value = nutrient.get("value")
                        unit = nutrient.get("unitName").lower()
                        
                        # Match structural macro keywords for tracking display rules
                        if any(m in name.lower() for m in ["energy", "protein", "carbohydrate", "total lipid"]):
                            st.write(f"• **{name}**: {value} {unit}")
                            
                    st.markdown("---")
        else:
            st.warning("No tracked fruit items could be localized in this view plane.")