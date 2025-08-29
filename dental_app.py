import streamlit as st
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import pandas as pd
import io

# App configuration
st.set_page_config(
    page_title="MedGemma Dental Analysis",
    page_icon="🦷",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load MedGemma model and processor (cached for performance)"""
    with st.spinner("Loading MedGemma model... This may take a few minutes."):
        model = AutoModelForImageTextToText.from_pretrained(
            "google/medgemma-4b-it",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained("google/medgemma-4b-it")
    return model, processor

def analyze_dental_image(image, model, processor, custom_prompt=None):
    """Analyze dental radiograph using MedGemma"""
    
    # Default prompt if none provided
    if not custom_prompt:
        custom_prompt = "Please analyze this dental radiograph and identify any dental conditions, restorations, or abnormalities. Provide specific observations and clinical recommendations."
    
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert dental radiologist. Provide detailed, clinical analysis of dental radiographs."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": custom_prompt},
                {"type": "image", "image": image}
            ]
        }
    ]
    
    try:
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)
        
        input_len = inputs["input_ids"].shape[-1]
        
        with torch.inference_mode():
            generation = model.generate(**inputs, max_new_tokens=300, do_sample=False)
        
        generation = generation[0][input_len:]
        output_text = processor.decode(generation, skip_special_tokens=True)
        
        return output_text
        
    except Exception as e:
        return f"Error during analysis: {str(e)}"

def main():
    st.title("🦷 MedGemma Dental Radiograph Analysis")
    st.markdown("Upload dental radiographs for AI-powered analysis using Google's MedGemma model")
    
    # Sidebar
    st.sidebar.title("Options")
    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        ["General Analysis", "Specific Condition Check", "Custom Prompt"]
    )
    
    # Load model
    try:
        model, processor = load_model()
        st.sidebar.success("✅ MedGemma model loaded successfully")
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a dental radiograph",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a dental X-ray image for analysis"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Dental Radiograph", use_column_width=True)
            
            # Custom prompt options
            custom_prompt = ""
            if analysis_type == "Specific Condition Check":
                condition = st.selectbox(
                    "Select condition to check for:",
                    ["Caries (cavities)", "Fillings/Restorations", "Implants", "Periodontal issues", "Bone loss"]
                )
                custom_prompt = f"Please specifically examine this dental radiograph for signs of {condition.lower()}. Provide detailed findings and recommendations."
            
            elif analysis_type == "Custom Prompt":
                custom_prompt = st.text_area(
                    "Enter your analysis request:",
                    placeholder="E.g., 'Focus on the upper right quadrant and check for any abnormalities'"
                )
            
            # Analysis button
            if st.button("🔍 Analyze Radiograph", type="primary"):
                with st.spinner("Analyzing dental radiograph..."):
                    analysis = analyze_dental_image(image, model, processor, custom_prompt)
                
                # Display results in the second column
                with col2:
                    st.header("Analysis Results")
                    st.markdown("### MedGemma Analysis:")
                    st.write(analysis)
                    
                    # Download results
                    results_text = f"Dental Radiograph Analysis\n{'='*50}\n\nImage: {uploaded_file.name}\nAnalysis Type: {analysis_type}\n\nResults:\n{analysis}"
                    
                    st.download_button(
                        label="📄 Download Analysis Report",
                        data=results_text,
                        file_name=f"dental_analysis_{uploaded_file.name.split('.')[0]}.txt",
                        mime="text/plain"
                    )
    
    with col2:
        if uploaded_file is None:
            st.header("Analysis Results")
            st.info("👆 Upload a dental radiograph to see analysis results here")
            
            # Sample images section
            st.markdown("---")
            st.subheader("About this App")
            st.markdown("""
            This application uses Google's **MedGemma-4B** model to analyze dental radiographs.
            
            **Features:**
            - 🔍 AI-powered dental image analysis
            - 🦷 Detection of fillings, implants, caries
            - 📊 Clinical recommendations
            - 📄 Downloadable reports
            
            **Usage:**
            1. Upload a dental X-ray image
            2. Choose analysis type
            3. Click 'Analyze Radiograph'
            4. Review results and download report
            
            **Note:** This tool is for educational/research purposes only and should not replace professional dental diagnosis.
            """)
    
    # Batch processing option
    st.markdown("---")
    st.header("🔄 Batch Processing")
    with st.expander("Process Multiple Images"):
        uploaded_files = st.file_uploader(
            "Upload multiple dental radiographs",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True
        )
        
        if uploaded_files and st.button("Process All Images"):
            results = []
            progress_bar = st.progress(0)
            
            for i, file in enumerate(uploaded_files):
                image = Image.open(file).convert("RGB")
                
                with st.spinner(f"Processing {file.name}..."):
                    analysis = analyze_dental_image(image, model, processor)
                
                results.append({
                    "filename": file.name,
                    "analysis": analysis
                })
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            # Display batch results
            st.subheader("Batch Analysis Results")
            for result in results:
                with st.expander(f"📄 {result['filename']}"):
                    st.write(result['analysis'])
            
            # Create downloadable batch report
            batch_report = "\n\n" + "="*80 + "\n\n".join([
                f"File: {result['filename']}\nAnalysis: {result['analysis']}" 
                for result in results
            ])
            
            st.download_button(
                label="📦 Download Batch Report",
                data=batch_report,
                file_name="batch_dental_analysis.txt",
                mime="text/plain"
            )

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Powered by Google MedGemma-4B | For Educational Use Only"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
