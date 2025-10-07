import base64
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from io import BytesIO
from PIL import Image
from flask_cors import CORS
import os
app = Flask(__name__)


CORS(app, resources={r"/predict": {"origins": "https://g3weds.consolutechcloud.com"}})
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "final_model2_test2.h5")
# model = load_model(MODEL_PATH)

import tensorflow as tf
from tensorflow.keras import layers, models

class_names = ['Fungal leaves', 'Good leaves', 'Insect-eaten leaves']  # แก้ตามจำนวนคลาสจริง
# ขนาดภาพที่ใช้ตอนเทรน
img_height = 224
img_width = 224

# === สร้างสถาปัตยกรรมให้เหมือนตอนเทรน ===
def build_model(num_classes=3, input_shape=(224, 224, 3), dropout_rate=0.2):

    # 1️⃣ สร้าง base model (MobileNetV2)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(img_height, img_width, 3),
        include_top=False,
        weights='imagenet'  # โหลด pretrained weights
    )
    base_model.trainable = False  # freeze base model

    # 2️⃣ สร้าง head ของโมเดล
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(class_names))  # output layer
    ])
    return model

# === ใช้งานจริง ===
model = build_model(num_classes=3)
model.load_weights(MODEL_PATH)
print("✅ Model weights loaded successfully!")

def preprocess(img, target_size=(224, 224)):
    img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API is running"}), 200

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    data = request.get_json()

    if "image_base64" not in data:
        return jsonify({"error": "No image provided"}), 400

    try:
        # แปลง base64 -> Image
        image_data = data["image_base64"].split(",")[1]
        img = Image.open(BytesIO(base64.b64decode(image_data)))

        # Preprocess
        input_data = preprocess(img)

        # Predict
        prediction = model.predict(input_data)
        predicted_index = int(np.argmax(prediction, axis=1)[0])

        # แปลง logits -> probabilities
        prediction_prob = tf.nn.softmax(prediction, axis=1).numpy()

        # Map index -> class name
        predicted_class_name = class_names[predicted_index]

        # Confidence ของ class ที่ทำนาย
        confidence = float(prediction_prob[0][predicted_index])

        return jsonify({
            "prediction": prediction.tolist(),
            "predicted_index": predicted_index,
            "predicted_class_name": predicted_class_name,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4200, debug=True)
