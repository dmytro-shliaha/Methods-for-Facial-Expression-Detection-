import cv2
import pygame
import time
import numpy as np
import joblib
import tensorflow as tf
from skimage.feature import local_binary_pattern, hog

# ---------------- LOAD MODELS ----------------
knn = joblib.load("models/lbp_knn.pkl")
svm = joblib.load("models/hog_svm.pkl")
cnn = tf.keras.models.load_model("models/mini_xception.keras")

label_names = ["angry","disgust","fear","happy","sad","surprise","neutral"]

# ---------------- SETTINGS ----------------
IMG_SIZE = (48, 48)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

pygame.init()
screen = pygame.display.set_mode((1000, 700))  # більше вікно
pygame.display.set_caption("Facial Expression Comparison")
font = pygame.font.SysFont("Arial", 26)

cap = cv2.VideoCapture(0)

running = True
last_results = []   # збережемо останні результати
last_time = 0

while running:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # -------- Оновлюємо результати не щокадру, а раз у 0.7 сек --------
    if time.time() - last_time > 0.7 and len(faces) > 0:
        (x, y, w, h) = faces[0]  # тільки перше обличчя
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, IMG_SIZE)

        new_results = []

        # --- LBP+KNN ---
        start = time.time()
        lbp = local_binary_pattern(face_resized, P=16, R=3, method="uniform")
        hist, _ = np.histogram(lbp.ravel(),
                               bins=np.arange(0, 16+3),
                               range=(0, 16+2),
                               density=True)
        hist = hist.reshape(1, -1)
        pred_knn = knn.predict(hist)[0]
        prob_knn = np.max(knn.predict_proba(hist))
        t_knn = (time.time() - start) * 1000
        new_results.append(("LBP+KNN", label_names[pred_knn], prob_knn, t_knn))

        # --- HOG+SVM ---
        start = time.time()
        hog_feat = hog(face_resized, orientations=9, pixels_per_cell=(8,8),
                       cells_per_block=(2,2), block_norm="L2-Hys")
        hog_feat = hog_feat.reshape(1, -1)
        pred_svm = svm.predict(hog_feat)[0]
        score = svm.decision_function(hog_feat)
        prob_svm = np.max(score) / (np.max(np.abs(score)) + 1e-6)  # нормалізація
        t_svm = (time.time() - start) * 1000
        new_results.append(("HOG+SVM", label_names[pred_svm], prob_svm, t_svm))

        # --- CNN ---
        start = time.time()
        face_cnn = face_resized.astype("float32")/255.0
        face_cnn = face_cnn.reshape(1,48,48,1)
        pred_cnn = cnn.predict(face_cnn, verbose=0)
        cls = np.argmax(pred_cnn)
        prob_cnn = np.max(pred_cnn)
        t_cnn = (time.time() - start) * 1000
        new_results.append(("CNN", label_names[cls], prob_cnn, t_cnn))

        last_results = new_results
        last_time = time.time()

    # ---------------- SHOW IN PYGAME ----------------
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_surface = pygame.surfarray.make_surface(np.rot90(frame_rgb))

    # Малюємо вебкамеру зліва
    screen.fill((0,0,0))
    screen.blit(pygame.transform.scale(frame_surface, (700, 700)), (0,0))

    # Панель справа
    panel_x = 720
    y_offset = 50
    title = font.render("Model Results", True, (255,255,0))
    screen.blit(title, (panel_x, 10))
    for (name, label, prob, t) in last_results:
        text = f"{name}: {label} ({prob:.2f}) [{t:.1f} ms]"
        surface = font.render(text, True, (255,255,255))
        screen.blit(surface, (panel_x, y_offset))
        y_offset += 60

    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

cap.release()
pygame.quit()
