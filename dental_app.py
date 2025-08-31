import streamlit as st
from PIL import Image
import requests
import base64
import os
import io

# --- Colab API Setup ---
COLAB_API_URL = st.secrets["COLAB_API_URL"]

def query_colab_api(image, prompt):
    """Send request to your Colab API"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    image_bytes = buffered.getvalue()

    payload = {
        "prompt": prompt,
        "image": base64.b64encode(image_bytes).decode("utf-8")
    }

    response = requests.post(COLAB_API_URL, json=payload)
    if response.status_code != 200:
        return f"Error {response.status_code}: {response.text}"
    return response.json().get("result", str(response.json()))

# --- Streamlit UI ---
st.set_page_config(page_title="MedGemma Dental Analysis", page_icon="🦷", layout="wide")

st.title("🦷 MedGemma Dental Radiograph Analysis")
st.markdown("Upload dental radiographs for AI-powered analysis using Hugging Face Inference API.")

uploaded_file = st.file_uploader("Upload a dental radiograph", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Dental Radiograph", width="stretch")

    analysis_type = st.selectbox(
        "Analysis Type",
        ["General Analysis", "Specific Condition Check", "Custom Prompt"]
    )

    if analysis_type == "Specific Condition Check":
        condition = st.selectbox(
            "Select condition to check for:",
            ["Caries (cavities)", "Fillings/Restorations", "Implants", "Periodontal issues", "Bone loss"]
        )
        prompt = f"Please examine this dental radiograph for signs of {condition.lower()} and provide detailed findings."
    elif analysis_type == "Custom Prompt":
        prompt = st.text_area("Enter your analysis request:")
    else:
        prompt = "Please analyze this dental radiograph and identify any dental conditions, restorations, or abnormalities. Provide clinical recommendations."

    if st.button("🔍 Analyze Radiograph"):
        with st.spinner("Analyzing..."):
            result = query_hf_api(image, prompt)
        st.subheader("Analysis Results")
        st.write(result)
        st.download_button(
            "📄 Download Analysis Report",
            data=result,
            file_name=f"dental_analysis_{uploaded_file.name.split('.')[0]}.txt",
            mime="text/plain"
        )

