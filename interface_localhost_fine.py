import cv2
import numpy as np
import joblib
import tensorflow as tf
from skimage.feature import local_binary_pattern, hog
import time
import gradio as gr
import os

# --- Model Dummies for Robustness ---
# If a model file is missing, we use a dummy class to prevent the app from crashing.
class DummyModel:
    """Returns 'neutral' prediction and 0 confidence if model loading fails."""
    def predict(self, X):
        # Assuming the label 'neutral' is index 6
        return np.array([6]) 
    def predict_proba(self, X):
        # Return low confidence for all classes
        return np.array([[0.14] * 6 + [0.16]])
    def decision_function(self, X):
        # Return a simple score for SVM dummy
        return np.array([0.1])

# ---------------- CONFIGURATION ----------------
LABEL_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
IMG_SIZE = (48, 48)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ---------------- LOAD MODELS ROBUSTLY ----------------
# Use try/except to ensure the app launches even if model files are missing
knn, svm, cnn = DummyModel(), DummyModel(), DummyModel()

try:
    print("Attempting to load LBP+KNN model...")
    knn = joblib.load("models/lbp_knn.pkl")
    print("LBP+KNN loaded successfully.")
except FileNotFoundError:
    print("Warning: LBP+KNN model file not found. Using dummy model.")
except Exception as e:
    print(f"Error loading LBP+KNN: {e}. Using dummy model.")

try:
    print("Attempting to load HOG+SVM model...")
    svm = joblib.load("models/hog_svm.pkl")
    print("HOG+SVM loaded successfully.")
except FileNotFoundError:
    print("Warning: HOG+SVM model file not found. Using dummy model.")
except Exception as e:
    print(f"Error loading HOG+SVM: {e}. Using dummy model.")

try:
    print("Attempting to load mini-Xception CNN model...")
    # TensorFlow/Keras specific loading
    cnn = tf.keras.models.load_model("models/mini_xception.keras")
    print("CNN loaded successfully.")
except FileNotFoundError:
    print("Warning: CNN model file not found. Using dummy model.")
except Exception as e:
    print(f"Error loading CNN: {e}. Using dummy model.")


# ---------------- FRAME PROCESSING PIPELINE ----------------
def process_frame(image):
    """
    Detects a face in the image, crops it, and runs three prediction models,
    returning the annotated image and an HTML table of results.
    
    image: numpy array in RGB format (from Gradio webcam)
    """
    # 1. Preprocessing and Face Detection
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    results = []
    output_image = frame.copy()  # Image for drawing bounding box

    if len(faces) > 0:
        (x, y, w, h) = faces[0]  # Take the largest/first detected face
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, IMG_SIZE)

        # Draw bounding box
        cv2.rectangle(output_image, (x, y), (x+w, y+h), (76, 175, 80), 3) # Green box (BGR)

        # LBP + KNN Prediction ---
        start_knn = time.time()
        
        # LBP Feature Extraction
        lbp = local_binary_pattern(face_resized, P=16, R=3, method="uniform")
        # Histogram calculation (Feature Vector)
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 16 + 3), range=(0, 16 + 2), density=True)
        hist = hist.reshape(1, -1)
        
        # Prediction
        pred_knn_idx = knn.predict(hist)[0]
        prob_knn = np.max(knn.predict_proba(hist))
        t_knn = (time.time() - start_knn) * 1000
        
        # Store result
        results.append(("LBP+KNN", LABEL_NAMES[pred_knn_idx], f"{prob_knn:.2f}", f"{t_knn:.1f} ms"))

        # HOG + SVM Prediction ---
        start_svm = time.time()
        
        # HOG Feature Extraction
        hog_feat = hog(face_resized, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), block_norm="L2-Hys")
        hog_feat = hog_feat.reshape(1, -1)
        
        # Prediction
        pred_svm_idx = svm.predict(hog_feat)[0]
        
        # Confidence (LinearSVC does not natively give probability, using decision function normalization)
        score = svm.decision_function(hog_feat)
        prob_svm = np.max(score) / (np.max(np.abs(score)) + 1e-6) if np.max(np.abs(score)) > 0 else 0.0
        
        t_svm = (time.time() - start_svm) * 1000
        
        # Store result
        results.append(("HOG+SVM", LABEL_NAMES[pred_svm_idx], f"{prob_svm:.2f}", f"{t_svm:.1f} ms"))

        # CNN (mini-Xception) Prediction ---
        start_cnn = time.time()
        
        # Prepare 4D tensor: Normalize and reshape (1, 48, 48, 1)
        face_cnn = face_resized.astype("float32") / 255.0
        face_cnn = face_cnn.reshape(1, 48, 48, 1)
        
        # Prediction
        pred_cnn = cnn.predict(face_cnn, verbose=0)
        cls_idx = np.argmax(pred_cnn)
        prob_cnn = np.max(pred_cnn)
        
        t_cnn = (time.time() - start_cnn) * 1000
        
        # Store result
        results.append(("CNN", LABEL_NAMES[cls_idx], f"{prob_cnn:.2f}", f"{t_cnn:.1f} ms"))

    else:
        results = [("Face Detection", "N/A", "N/A", "N/A")] * 3
        # Add text if no face is found
        cv2.putText(output_image, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    # HTML Table for Results (with basic styling)
    table_html = """
    <style>
        .result-table {
            width: 100%;
            border-collapse: collapse;
            font-family: Arial, sans-serif;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .result-table th, .result-table td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        .result-table th {
            background-color: #4CAF50; /* Green header */
            color: white;
            font-size: 1.05em;
        }
        .result-table tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .result-table tr:hover {
            background-color: #e9e9e9;
        }
        .ms-cell {
            font-weight: bold;
            color: #d32f2f; /* Red for latency */
        }
    </style>
    <table class='result-table'>
        <thead>
            <tr>
                <th>Model</th>
                <th>Prediction</th>
                <th>Confidence</th>
                <th>Latency (ms)</th>
            </tr>
        </thead>
        <tbody>
    """
    
    # Populate table rows
    for res in results:
        # Split latency string to remove " ms" for cleaner display
        latency_val = res[3].split(" ")[0]
        table_html += f"""
            <tr>
                <td>{res[0]}</td>
                <td><strong>{res[1]}</strong></td>
                <td>{res[2]}</td>
                <td class='ms-cell'>{latency_val}</td>
            </tr>
        """
    table_html += "</tbody></table>"

    # Convert output_image back to RGB for Gradio
    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

    return output_image, table_html

# ---------------- GRADIO INTERFACE ----------------
with gr.Blocks() as demo:
    gr.Markdown("# Comparison of Emotion Recognition Models (LBP+KNN vs HOG+SVM vs CNN)")
    gr.Markdown("The real-time interface uses a webcam to detect a face and shows three predictions side by side with confidence and latency.")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(
                label="Webcam (Input)", 
                sources=["webcam"], 
                type="numpy", 
                streaming=True, 
                show_label=True
            )
        with gr.Column(scale=1):
            output_img = gr.Image(label="Processed frame with detected face")
            output_html = gr.HTML(label="Prediction summary")
            
    # Use stream for continuous, real-time updates from the webcam
    input_img.stream(
        fn=process_frame,
        inputs=[input_img],
        outputs=[output_img, output_html],
        time_limit=None,
        stream_every=0.5 
    )

if __name__ == "__main__":
    demo.launch()