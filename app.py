import streamlit as st
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import torch

# Check if GPU is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model and tokenizer with GPU configuration
tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
model = AutoModel.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True, low_cpu_mem_usage=True, device_map='auto', use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
model = model.eval().to(device)

# Streamlit app
st.title("OCR with Transformers")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Save the uploaded image to a file
    image_file = 'uploaded_image.jpg'
    image.save(image_file)

    # Perform OCR
    res = model.chat(tokenizer, image_file, ocr_type='ocr')

    # Display the OCR result
    st.write("OCR Result:")
    st.write(res)

    # Search input
    search_query = st.text_input("Enter a keyword to search within the extracted text:")

    # Search functionality
    if search_query:
        results = [line for line in res.split('\n') if search_query.lower() in line.lower()]
        if results:
            st.write("Search Results:")
            for result in results:
                st.write(f"- {result}")
        else:
            st.write("No results found.")

