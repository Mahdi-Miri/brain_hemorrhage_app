import requests
import streamlit as st

API_URL = "https://brain-ct-api-849714763027.europe-west1.run.app"

def check_hemorrhage_api(npy_bytes: bytes) -> tuple[str, str]:
    """
    Sends .npy bytes to DL backend and returns:
    - prediction label
    - probability string
    """
    try:
        files = {"file": ("scan.npy", npy_bytes, "application/octet-stream")}
        response = requests.post(API_URL, files=files, timeout=40)

        if response.status_code != 200:
            st.error(f"API Error: {response.status_code}")
            return "Error", "Unknown"

        data = response.json()

        prediction = int(data.get("prediction", 0))
        probability = float(data.get("probability_class_1", 0.0))

        label = "Hemorrhage Detected" if prediction == 1 else "No Hemorrhage Detected"
        prob_str = f"{probability:.2%}"

        return label, prob_str

    except Exception as e:
        st.error(f"API Connection Failed: {e}")
        return "Error", "Unknown"