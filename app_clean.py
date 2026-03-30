import os
import gradio as gr
import google.generativeai as genai
from PIL import Image


print(">>> Starting app... - app_clean.py:6")


# Configure the Gemini API with environment variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please configure it in the Hugging Face Space settings or system environment.")


genai.configure(api_key=GOOGLE_API_KEY)


# Use Gemini 2.5 Flash model
model = genai.GenerativeModel('models/gemini-2.5-flash')


def detect_deepfake(image):
    if image is None:
        return "Error", "Please upload or capture an image."
   
    # Resize image for consistency
    image = image.resize((224, 224))
   
    prompt = """
    You are an image analysis assistant tasked with detecting whether an image is a deepfake or real.
    A deepfake is a synthetically generated or manipulated image, often of a human face, showing unnatural features like inconsistent lighting, unnatural textures, or blending artifacts.
    A real image typically has natural lighting, consistent facial features, and no synthetic artifacts.
    Classify the image as "Real" or "Deepfake" and provide a brief reason for your classification.


    Return the response in this format:
    Prediction: [Real / Deepfake]
    Reason: [Brief explanation]
    """


    try:
        response = model.generate_content([prompt, image])
        result = response.text.strip()


        prediction = "Unknown"
        reason = "No explanation provided."


        if "Prediction:" in result:
            parts = result.split("Prediction:")[1].split("Reason:")
            prediction = parts[0].strip()
            if len(parts) > 1:
                reason = parts[1].strip()


        return prediction, reason


    except Exception as e:
        return "Error", f"{str(e)}\nTip: Ensure your API key is valid at https://aistudio.google.com/"


iface = gr.Interface(
    fn=detect_deepfake,
    inputs=gr.Image(type="pil", label="Upload or capture a face image", sources=["upload", "webcam"]),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Reason")
    ],
    title="Deepfake Detection App",
    description="Upload or capture a face image to detect whether it is Real or Deepfake using Gemini 2.5 Flash."
)


if __name__ == "__main__":
    print(">>> Launching Gradio app... - app_clean.py:66")
    iface.launch()
