import os
import gradio as gr
import google.generativeai as genai
from PIL import Image


# Configure the Gemini API with environment variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please configure it in the Hugging Face Space settings.")


genai.configure(api_key=GOOGLE_API_KEY)


# Use Gemini 1.5 Flash model
model = genai.GenerativeModel('models/gemini-2.5-flash')


def detect_deepfake(image):
    if image is None:
        return "Error: Please upload or capture an image."
   
    # Resize image for consistency (optional, Gemini handles various sizes)
    image = image.resize((224, 224))
   
    prompt = """
    You are an image analysis assistant tasked with detecting whether an image is a deepfake or real.
    A deepfake is a synthetically generated or manipulated image, often of a human face, showing unnatural features like inconsistent lighting, unnatural textures, or blending artifacts.
    A real image typically has natural lighting, consistent facial features, and no synthetic artifacts.
    Classify the image as "Real" or "Deepfake" and provide a brief reason for your classification.
    Return the response in this format:
   
    Prediction: [Real / Deepfake]
    Reason: [Brief explanation]


    Examples:
    Image Description: A face with consistent lighting, natural skin texture, and realistic eye movements.
    Prediction: Real
    Reason: The image shows natural lighting and facial features consistent with a real human photograph.


    Image Description: A face with unnatural blending around the eyes, inconsistent lighting on the face, and slight pixelation.
    Prediction: Deepfake
    Reason: The unnatural blending and inconsistent lighting suggest synthetic manipulation typical of deepfakes.


    Image Description: A face with smooth, overly perfect skin and slightly distorted facial proportions.
    Prediction: Deepfake
    Reason: The overly smooth skin and distorted proportions are indicative of AI-generated or manipulated images.


    Image Description: A face with natural shadows, realistic hair texture, and consistent background lighting.
    Prediction: Real
    Reason: The natural shadows and realistic textures align with characteristics of a genuine photograph.


    Classify the provided image.
    Prediction:
    Reason:
    """


    try:
        response = model.generate_content([prompt, image])
        return response.text.strip()
    except Exception as e:
        return f"Error: {str(e)}\nTip: Ensure your API key is valid at https://aistudio.google.com/"


# Define Gradio interface
iface = gr.Interface(
    fn=detect_deepfake,
    inputs=gr.Image(type="pil", label="Upload or capture a face image", sources=["upload", "webcam"]),
    outputs=gr.Textbox(label="Deepfake Detection Result"),
    title="Deepfake Image Detector",
    description="Upload or capture a face image to determine if it is Real or a Deepfake using the Gemini API. Set your GOOGLE_API_KEY in the Hugging Face Space settings."
)


# Launch the interface
if __name__ == "__main__":
    iface.launch()
