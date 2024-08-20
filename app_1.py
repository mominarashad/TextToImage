import streamlit as st
from diffusers import DiffusionPipeline
import torch
from PIL import Image
import io
import numpy as np
import time

@st.cache_resource
def load_model():
    model = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model

generator = load_model()

st.title('Text to Image Generator')
prompt = st.text_input("Enter your prompt:", "A scenic landscape painting")

if st.button('Generate Image'):
    start_time = time.time()  # Start timing
    with st.spinner('Generating image...'):
        # Generate the image with a low number of inference steps
        output = generator(prompt, num_inference_steps=5, eta=0.0)  # Adjust ETA to possibly reduce computation
        image = output.images[0]

        # Convert to NumPy array
        image = np.array(image)

        # Convert to PIL Image
        pil_image = Image.fromarray(image.astype(np.uint8))  # Cast to uint8 for PIL compatibility

        # Display the image
        st.image(pil_image, caption='Generated Image', use_column_width=True)

        # Provide a download button
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG")
        byte_im = buf.getvalue()
        st.download_button(
            label="Download Image",
            data=byte_im,
            file_name="generated_image.jpg",
            mime="image/jpeg"
        )

    # Record and display time taken
    end_time = time.time()
    st.write(f"Time taken to generate image: {end_time - start_time:.2f} seconds")

st.write("Enter a textual prompt and press generate to create an image.")
