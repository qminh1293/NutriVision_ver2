import streamlit as st
import os
import cv2
import requests
from ultralytics import YOLO
from PIL import Image
import numpy as np

#1.WEB CORE & ENVIRONMENT SETTINGS
st.set_page_config(page_title="NutriVision", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "runs", "detect", "train-2", "weights", "best.pt")
BASE_URL = "https://api.nal.usda.gov/fdc/v1"
API_KEY = st.secrets["USDA_API_KEY"]

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
    if not os.path.exists(MODEL_PATH):
        st.error(f"Could not locate model file at runtime structure: {MODEL_PATH}")
        return None
    return YOLO(MODEL_PATH)
model = load_yolo_model()

#2.USDA NETWORKING HANDSHAKE ENDPOINT
def fetch_nutrition(fruit_name):
    """Communicates securely with the US Government database from the backend."""
    clean_name = fruit_name.strip().capitalize()
    search_term = FRUIT_QUERY_MODIFIER.get(clean_name, f"{clean_name} raw")   
    target_url = f"{BASE_URL}/foods/search?api_key={API_KEY}"    
    params = {
        "query": search_term,
        "pageSize": 1,
        "dataType": ["Foundation", "Survey (FNDDS)"]
    }
    try:
        response = requests.get(target_url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            foods = data.get("foods", [])
            if foods:
                return foods[0]
    except Exception as e:
        st.error(f"Failed to query database ecosystem: {str(e)}")
    return None
    
#3.STREAMLIT APPLICATION VIEW INTERFACE
st.title("Food Classification & Insight System")
st.write("Upload an image of a fruit to instantly run YOLOv8 object detection and fetch verified USDA nutrition facts.")

if model is not None:
    uploaded_file = st.file_uploader("Choose a target fruit image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        raw_image = Image.open(uploaded_file)
        rgb_image = raw_image.convert("RGB")
        img_array_rgb = np.array(rgb_image)
        img_array_bgr = cv2.cvtColor(img_array_rgb, cv2.COLOR_RGB2BGR)
        st.info("Running object detection inference...")
        
        results = model.predict(source=img_array_bgr, conf=0.25, imgsz=640, verbose=False)
        result_object = results[0]
        detected_classes = []
        if len(result_object.boxes) > 0:
            for box in result_object.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                if class_name not in detected_classes:
                    detected_classes.append(class_name)

        annotated_img_bgr = result_object.plot()
        annotated_img_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Result")
            st.image(annotated_img_rgb, channels="RGB", width="stretch")
            
        with col2:
            st.subheader("📋 USDA Nutrition Facts")            
            if detected_classes:
                st.success(f"Spotted: {', '.join(detected_classes)}")    
                for fruit in detected_classes:
                    with st.spinner(f"Fetching verified data for {fruit}..."):
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
                        st.error(f"⚠️ USDA Database could not find matching food profile data structures for: '{fruit}'")
            else:
                st.warning("No tracked fruit items could be localized in this view plane.")
