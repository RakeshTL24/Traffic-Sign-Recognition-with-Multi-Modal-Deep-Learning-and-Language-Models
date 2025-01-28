import streamlit as st
import sys
import os
import subprocess
import requests
import shutil
from pathlib import Path
from PIL import Image
import cv2

# Hugging Face Mistral API Key (replace with your API key)
MISTRAL_API_KEY = 

# Paths for processing files
TEST_DIR = "../Test/"
RESULT_DIR = "../Results/"

# Create directories if they don't exist
os.makedirs(TEST_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

def query_mistral(class_name):
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}"}
    prompt = f"Explain the following traffic sign: {class_name}. Provide a brief and clear description."
    payload = {"inputs": prompt, "parameters": {"max_tokens": 150}}  # Request longer output
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        full_text = response.json()[0]['generated_text']
        
        # # Remove the prompt text from the response
        # explanation_start = full_text.find(":") + 1  # Find the start of the explanation
        # explanation = full_text[explanation_start:].strip()
        lines = full_text.split("\n")
        explanation = "\n".join(lines[1:]).strip()

        # Check if the explanation ends with a valid punctuation mark
        if explanation and explanation[-1] not in ".!?":
            explanation = explanation.rsplit(".", 1)[0] + "."  # Trim to the last complete sentence
        
        return explanation
    except Exception as e:
        return f"Error querying Hugging Face API: {e}"




def read_detected_classes(results_dir):
    """Read detected classes from the detected_classes.txt file"""
    classes_file = Path(results_dir) / 'Results' / 'detected_classes.txt'
    if classes_file.exists():
        with open(classes_file, 'r') as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    return []

def convert_video_to_web_compatible(input_path, output_path):
    """Convert video to web-compatible format using ffmpeg"""
    try:
        # Using ffmpeg to convert to web-compatible MP4
        command = [
            'ffmpeg',
            '-i', str(input_path),
            '-vcodec', 'libx264',
            '-acodec', 'aac',
            '-movflags', '+faststart',
            '-y',  # Overwrite output file if it exists
            str(output_path)
        ]
        subprocess.run(command, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"Error converting video: {e}")
        return False

def clean_dir(directory_path):
    try:
        if os.path.exists(directory_path):
            shutil.rmtree(directory_path)
        os.makedirs(directory_path, exist_ok=True)
    except Exception as e:
        st.error(f"Error cleaning directory {directory_path}: {e}")

def run_yolo_detection(source_path):
    try:
        clean_dir(RESULT_DIR)
        python_executable = sys.executable
        command = [python_executable, "detect.py", "--source", source_path, "--project", RESULT_DIR, "--name", "Results"]
        subprocess.run(command, check=True, cwd=os.path.dirname(__file__))
    except subprocess.CalledProcessError as e:
        st.error(f"Error running YOLOv5 detection: {e}")

def display_detected_results():
    result_files = list(Path(RESULT_DIR).glob("**/*"))
    if not result_files:
        st.error("No detection results found.")
        return

    for file in result_files:
        if file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            st.image(str(file), caption=f"Result Image: {file.name}", use_container_width=True)
        elif file.suffix.lower() in [".mp4", ".avi"]:
            # Create a web-compatible version of the video
            web_compatible_path = file.parent / f"web_compatible_{file.name}"
            if convert_video_to_web_compatible(file, web_compatible_path):
                try:
                    # Read video file
                    video_file = open(str(web_compatible_path), 'rb')
                    video_bytes = video_file.read()
                    st.video(video_bytes)
                    video_file.close()
                except Exception as e:
                    st.error(f"Error displaying video: {e}")
            else:
                st.error("Error converting video to web-compatible format")

def main():
    
    st.set_page_config(
        page_title="Traffic Sign Detection",
        page_icon="🚦",
        layout="centered"
    )

    page_bg_img = """
        <style>
        [data-testid="stAppViewContainer"] {
        background-image: url("https://wallpaperaccess.com/full/3104711.jpg");
        background-size: 100% 100% ;
        }
                
        [data-testid="stHeader"] {
        background-color: rgba(0, 0, 0, 0);
        }

        [data-testid="stBottomBlockContainer"] {
        background-color: rgba(0, 0, 0, 0);
        }

        .st-emotion-cache-uhkwx6.ea3mdgi6 {
            background-color: rgba(0, 0, 0, 0);
        }

        </style>
    """

    st.markdown(page_bg_img, unsafe_allow_html=True)

    
    st.title("Traffic Sign Detection and Description App🚦")

    # Sidebar for file upload
    st.sidebar.header("Upload Image or Video")
    uploaded_file = st.sidebar.file_uploader("Upload a file", type=["jpg", "jpeg", "png", "mp4"])
    
    if uploaded_file:
        # Save uploaded file to TEST_DIR
        file_path = Path(TEST_DIR) / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File uploaded successfully: {uploaded_file.name}")

        # Run YOLO detection
        with st.spinner("Running traffic sign detection..."):
            run_yolo_detection(str(file_path))

        # Display detection results
        st.subheader("Detection Results")
        display_detected_results()

        # Read and process detected classes
        detected_classes = read_detected_classes(RESULT_DIR)
        
        if detected_classes:
            st.subheader("Detected Traffic Signs and Descriptions")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.write("**Detected Signs:**")
                for class_name in detected_classes:
                    st.write(f"- {class_name}")
            
            st.markdown(
                """
                <style>
                .stExpander {
                    background-color: white !important;
                    color: black !important;
                    border: 1px solid #ddd !important;
                    border-radius: 10px;
                }
                .stExpander div[role="button"] {
                    color: black !important;
                }
                .stExpanderContent {
                    color: black !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            if st.button("Get Sign Descriptions"):
                with st.spinner("Getting descriptions from Mistral..."):
                    # col1, col2 = st.columns(2)
                    with col2:
                        st.write("**Detailed Descriptions:**")
                        for class_name in detected_classes:
                            st.markdown(
                                f"""
                                <div style="font-size:20px; font-weight:bold; color: white; margin-bottom:-10px;">
                                    Description for {class_name}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            with st.expander(""):
                                description = query_mistral(class_name)
                                st.write(description)
        else:
            st.warning("No traffic signs detected in the image/video.")

if __name__ == "__main__":
    main()