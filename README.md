**Implemented Models and Techniques**
This project features three distinct machine learning pipelines, allowing for a comprehensive analysis of feature-based vs. end-to-end learning approaches:
1. Classical Baseline: LBP + K-Nearest Neighbors (KNN)
2. Feature-Based Approach: HOG + Linear Support Vector Machine (Linear SVM)
3. Deep Learning Approach: mini-Xception Convolutional Neural Network (CNN)

**Real-Time Evaluation Pipeline**
The final deliverable is an interactive interface (e.g., using Gradio or Pygame) that visualizes predictions from all three models in real-time.
The core pipeline executed for every input frame is:
1. Face Detection: Identify the location of the face (e.g., using a lightweight Haar cascade or similar method).
2. Preprocessing: Crop, align, and resize the detected face to the required 48x48 pixel input size.
3. Multi-Model Inference: Pass the preprocessed image simultaneously through the LBP+KNN, HOG+SVM, and mini-Xception models.
4. Display: Present the prediction for each model, including its confidence score and the latency (in milliseconds) required for inference, allowing for a side-by-side performance comparison.

**Technologies Used**
1. Python
2. TensorFlow / Keras (for CNN implementation)
3. scikit-learn (for KNN and Linear SVM)
4. scikit-image (for LBP and HOG feature extraction)
5. OpenCV (for image handling and face detection)
6. Gradio (for the user interface)

**Project Results and Performance Metrics**

The project successfully implemented three distinct methodologies for Facial Expression Recognition (FER) on the challenging FER-2013 dataset. The results establish a clear performance hierarchy, validating the superiority of modern feature-learning and deep-learning approaches.

**Summary of Achieved Test Accuracy**

The project successfully benchmarked classical and deep learning approaches on the challenging **FER-2013** dataset. The results demonstrate the clear efficacy of feature-based and deep learning pipelines.

| Method | Feature Extraction / Architecture | Classifier | Achieved Test Accuracy | Key Performance Insight |
| :--- | :--- | :--- | :---: | :--- |
| **Deep Learning** | mini-Xception CNN (End-to-End) | Softmax | **65.1%** | Achieved the highest accuracy and stability, essential for real-world application. |
| **Feature-Based** | HOG (Histogram of Oriented Gradients) | Linear SVM | **62.3%** | Proved that robust feature engineering can successfully meet the high-performance threshold. |
| **Classical Baseline** | LBP (Local Binary Pattern) | K-Nearest Neighbors (KNN) | *<30%* | Confirmed the necessity of advanced methods, as performance was **highly unstable** and sensitive to image noise. |

**Performance Analysis**
1. Mini-Xception CNN and HOG+SVM
The mini-Xception CNN and the HOG+SVM models successfully met the high performance threshold, achieving accuracies well above the 60% target.
The CNN pipeline, leveraging extensive data augmentation and a specialized architecture, demonstrated the highest overall accuracy and the best generalization capability.
The HOG+SVM combination also performed exceptionally well, proving that carefully engineered features can still yield competitive results.
It is important to note that while these models are robust, they occasionally misclassify subtle or ambiguous emotional expressions—a known difficulty inherent in the FER-2013 dataset due to image quality and the subjective nature of human emotion labeling.

2. LBP+KNN (The Strict Baseline)
The LBP+KNN method clearly struggled with the complexity, noise, and intra-class variance of the FER-2013 images.
This model consistently provided unreliable and constantly fluctuating predictions, confirming its unsuitability for real-world application.
Its low performance effectively served its primary purpose: to strictly define the minimum viable performance boundary and highlight the necessity of advanced techniques like CNNs for this task.
