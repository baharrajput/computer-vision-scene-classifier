import os

import cv2
import gradio as gr
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

MODEL_PATH = os.environ.get("MODEL_PATH", "./saved_model/scene_classifier.keras")
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. Please place your trained model there or set MODEL_PATH."
    )

model = load_model(MODEL_PATH)


def predict_scene(image):
    image = np.array(image).astype(np.uint8)
    original = image.copy()
    img = cv2.resize(image, (224, 224))
    img = preprocess_input(img.astype(np.float32))
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)[0]
    predicted_index = int(np.argmax(prediction))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(np.max(prediction) * 100)

    probabilities = {
        class_name: float(probability)
        for class_name, probability in zip(CLASS_NAMES, prediction)
    }

    return original, predicted_class, confidence, probabilities


def gradio_predict(image):
    original, pred, conf, probs = predict_scene(image)

    prob_text = ""
    for name, value in probs.items():
        prob_text += f"{name:<12} : {value * 100:.2f}%\n"

    return original, pred, f"{conf:.2f} %", prob_text


with gr.Blocks(theme=gr.themes.Soft(), title="Scene Classification") as demo:
    gr.Markdown("""
    # 🌍 Scene Classification with MobileNetV2

    Upload an outdoor image to predict its scene category.
    """)

    with gr.Row():
        input_image = gr.Image(type="pil", label="Upload Image")
        output_image = gr.Image(label="Preview")

    predict_btn = gr.Button("🔍 Predict Scene", variant="primary")

    with gr.Row():
        prediction = gr.Textbox(label="Prediction")
        confidence = gr.Textbox(label="Confidence")

    probabilities = gr.Textbox(label="Class Probabilities", lines=8)

    predict_btn.click(
        fn=gradio_predict,
        inputs=input_image,
        outputs=[output_image, prediction, confidence, probabilities],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, debug=True)
