import streamlit as st
import base64
import os
from dotenv import load_dotenv
from openai import OpenAI
import tempfile

# Load environment variables
load_dotenv()

# Initialize OpenAI client with API Key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API Key is missing.")
    st.stop()

client = OpenAI(api_key=api_key)

# Sample prompt
sample_prompt = """You are a medical practitioner and an expert in analyzing medical-related images working for a reputed hospital.
You will be provided with images and need to identify anomalies, diseases, or health issues. 
Write detailed findings, next steps, and recommendations.
Always add the disclaimer: 'Consult with a Doctor before making any decisions.'"""

# Initialize session state variables
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'result' not in st.session_state:
    st.session_state.result = None

def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        st.error(f"Error encoding image: {e}")
        return None

def call_gpt4_model_for_analysis(filename):
    base64_image = encode_image(filename)
    if not base64_image:
        return "Unable to process the image."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": sample_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                    }
                }
            ]
        }
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=messages,
            max_tokens=1500
        )

        return response.choices[0].message.content
    except Exception as e:
        st.error(f"API Error: {e}")
        return "Unable to analyze the image at this moment."

def chat_eli(query):
    eli5_prompt = "Explain the following information to a 5-year-old child:\n" + query
    messages = [{"role": "user", "content": eli5_prompt}]

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=1500
        )

        return response.choices[0].message.content
    except Exception as e:
        st.error(f"ELI5 API Error: {e}")
        return "Unable to simplify the explanation."

st.title("Medical Image Analysis App")

with st.expander("About this App"):
    st.write("Upload an image of a medical scan to receive an AI-based analysis.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file.flush()  # Important to ensure data is written
        st.session_state['filename'] = tmp_file.name

    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

if st.button("Analyze Image"):
    if 'filename' in st.session_state and os.path.exists(st.session_state['filename']):
        st.session_state['result'] = call_gpt4_model_for_analysis(st.session_state['filename'])
        st.markdown(st.session_state['result'], unsafe_allow_html=True)
        os.unlink(st.session_state['filename'])  # Delete temp file after use
    else:
        st.error("No valid image uploaded or file path is incorrect.")

if 'result' in st.session_state and st.session_state['result']:
    st.info("Would you like a simplified explanation?")
    eli5_option = st.radio("ELI5 - Explain Like I'm 5", ("No", "Yes"))

    if eli5_option == "Yes":
        simplified_explanation = chat_eli(st.session_state['result'])
        st.markdown(simplified_explanation, unsafe_allow_html=True)
