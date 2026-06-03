import streamlit as st
import os
import cv2
import requests
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time

#1.WEB CORE & ENVIRONMENT SETTINGS
st.set_page_config(page_title="NutriVision", layout="centered")

st.sidebar.title("⚙️ Detection Settings")
conf_thresh = st.sidebar.slider("Confidence Threshold", min_value=0.05, max_value=0.95, value=0.25, step=0.05)
st.sidebar.info("Lower this slider if the model is ignoring fruits in your picture!")

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

MOCK_DATABASE = {
    "Apple": {
        "description": "Apples, raw (API Fallback)",
        "foodNutrients": [
            {"nutrientName": "Energy", "value": 52, "unitName": "kcal"},
            {"nutrientName": "Protein", "value": 0.3, "unitName": "g"},
            {"nutrientName": "Carbohydrate, by difference", "value": 13.8, "unitName": "g"},
            {"nutrientName": "Total lipid (fat)", "value": 0.2, "unitName": "g"}
        ]
    },
    "Banana": {
        "description": "Bananas, raw (API Fallback)",
        "foodNutrients": [
            {"nutrientName": "Energy", "value": 89, "unitName": "kcal"},
            {"nutrientName": "Protein", "value": 1.1, "unitName": "g"},
            {"nutrientName": "Carbohydrate, by difference", "value": 22.8, "unitName": "g"},
            {"nutrientName": "Total lipid (fat)", "value": 0.3, "unitName": "g"}
        ]
    },
    "Orange": {
        "description": "Oranges, raw (API Fallback)",
        "foodNutrients": [
            {"nutrientName": "Energy", "value": 47, "unitName": "kcal"},
            {"nutrientName": "Protein", "value": 0.9, "unitName": "g"},
            {"nutrientName": "Carbohydrate, by difference", "value": 11.8, "unitName": "g"},
            {"nutrientName": "Total lipid (fat)", "value": 0.1, "unitName": "g"}
        ]
    },
    "Grape": {
        "description": "Grapes, raw (API Fallback)",
        "foodNutrients": [
            {"nutrientName": "Energy", "value": 69, "unitName": "kcal"},
            {"nutrientName": "Protein", "value": 0.7, "unitName": "g"},
            {"nutrientName": "Carbohydrate, by difference", "value": 18.1, "unitName": "g"},
            {"nutrientName": "Total lipid (fat)", "value": 0.2, "unitName": "g"}
        ]
    },
    "Pineapple": {
        "description": "Pineapple, raw (API Fallback)",
        "foodNutrients": [
            {"nutrientName": "Energy", "value": 50, "unitName": "kcal"},
            {"nutrientName": "Protein", "value": 0.5, "unitName": "g"},
            {"nutrientName": "Carbohydrate, by difference", "value": 13.1, "unitName": "g"},
            {"nutrientName": "Total lipid (fat)", "value": 0.1, "unitName": "g"}
        ]
    },
    "Watermelon": {
        "description": "Watermelon, raw (API Fallback)",
        "foodNutrients": [
            {"nutrientName": "Energy", "value": 30, "unitName": "kcal"},
            {"nutrientName": "Protein", "value": 0.6, "unitName": "g"},
            {"nutrientName": "Carbohydrate, by difference", "value": 7.6, "unitName": "g"},
            {"nutrientName": "Total lipid (fat)", "value": 0.2, "unitName": "g"}
        ]
    }
}

@st.cache_resource
def load_yolo_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Could not locate model file at runtime structure: {MODEL_PATH}")
        return None
    return YOLO(MODEL_PATH)
model = load_yolo_model()

#2.USDA NETWORKING HANDSHAKE ENDPOINT
@st.cache_data(show_spinner=False)
def fetch_nutrition(fruit_name):
    native_str = str(fruit_name)
    clean_name = native_str.strip().capitalize()
    
    search_term = FRUIT_QUERY_MODIFIER.get(clean_name, f"{clean_name} raw")
    target_url = f"{BASE_URL}/foods/search?api_key={API_KEY}"
    
    params = {
        "query": search_term,
        "pageSize": 1,
        "dataType": ["Foundation", "Survey (FNDDS)"]
    }
    try:
        response = requests.get(target_url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            foods = data.get("foods", [])
            if foods:
                return foods[0], "live"
    except Exception:
        pass 
    return MOCK_DATABASE.get(clean_name, None), "mock"

#3.STREAMLIT APPLICATION VIEW INTERFACE
st.title("Food Classification & Insight System")
st.write("A web that detects fruit using YOLOv8 models and using USDA official database for nutrition information.")

if model is not None:
    input_source = st.radio("Choose image source:", ("Upload File", "Use Camera"))
    
    if input_source == "Upload File":
        uploaded_file = st.file_uploader("Choose a target fruit image (Apple, Banana, Grape, Pineapple, Watermelon, Orange)", type=["jpg", "jpeg", "png"])
    else:
        uploaded_file = st.camera_input("Take a picture of the fruit")
        
    if uploaded_file is not None:
        raw_image = Image.open(uploaded_file)
        
        rgb_image = raw_image.convert("RGB")
        img_array_rgb = np.array(rgb_image)
        img_array_bgr = cv2.cvtColor(img_array_rgb, cv2.COLOR_RGB2BGR)
        
        st.info("Running object detection inference...")
        
        results = model.predict(source=img_array_bgr, conf=conf_thresh, imgsz=640, verbose=False)
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
            st.image(annotated_img_rgb, channels="RGB", use_container_width=True)
            
        with col2:
            st.subheader("📋 USDA Nutrition Facts")
            if detected_classes:
                st.success(f"Spotted: {', '.join(detected_classes)}")   
                for fruit in detected_classes:
                    food_profile, source_status = fetch_nutrition(fruit)        
                    if food_profile:
                        if source_status == "mock": #API Calling Failure
                            st.warning(f"Using Local Database Fallback (Possible API Rate Limited)")
                            st.toast(f"Notice: Offline data fallback used for {fruit}", icon="⚠️")
                        
                        st.markdown(f"**🔬 {food_profile.get('description')} (Per 100g)**")
                        
                        for nutrient in food_profile.get("foodNutrients", []):
                            name = nutrient.get("nutrientName")
                            value = nutrient.get("value")
                            unit = nutrient.get("unitName").lower()
                            
                            if any(m in name.lower() for m in ["energy", "protein", "carbohydrate", "total lipid"]):
                                st.write(f"• **{name}**: {value} {unit}")   
                        st.markdown("---")
                    else:
                        st.error(f"System Error: No data structure available for: '{fruit}'")
                        
                    time.sleep(3.0) 
            else:
                st.warning("No tracked fruit items could be localized in this view plane.")
