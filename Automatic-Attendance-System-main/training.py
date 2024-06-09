import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
import imutils

curr_path = os.getcwd()

print("Loading face detection model")
proto_path = os.path.join(curr_path, 'deploy.prototxt')
model_path = os.path.join(curr_path, 'res10_300x300_ssd_iter_140000.caffemodel')
face_detector = cv2.dnn.readNetFromCaffe(prototxt=proto_path, caffeModel=model_path)

print("Loading face recognition model")
recognition_model = os.path.join(curr_path, 'openface_nn4.small2.v1.t7')
face_recognizer = cv2.dnn.readNetFromTorch(model=recognition_model)

data_base_path = 'sir_data'

face_embeddings = []
face_names = []

for epoch in range(5):
    for person_name in os.listdir(data_base_path):
        person_folder = os.path.join(data_base_path, person_name)
        count = 0
        count1 = 0
        for video_file in os.listdir(person_folder):
            video_path = os.path.join(person_folder, video_file)

            cap = cv2.VideoCapture(video_path)

            while True:
                count += 1
                ret, frame = cap.read()

                if not ret:
                    break

                frame = imutils.resize(frame, width=600)

                (h, w) = frame.shape[:2]

                image_blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0),
                                                   False, False)

                face_detector.setInput(image_blob)
                face_detections = face_detector.forward()

                i = np.argmax(face_detections[0, 0, :, 2])
                confidence = face_detections[0, 0, i, 2]

                if confidence >= 0.99:
                    count1 += 1
                    box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    face = frame[startY:endY, startX:endX]

                    if not face.size:
                        # Handle the case where the face is empty
                        print("Error: Face is empty.")
                    else:
                        # Perform resizing and other operations
                        face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0), True, False)

                    face_recognizer.setInput(face_blob)
                    face_recognitions = face_recognizer.forward()

                    face_embeddings.append(face_recognitions.flatten())
                    face_names.append(person_name)

            cap.release()
        print(f"Epoch: {epoch + 1}, Count: {count}, Detected: {count1}")

data = {"embeddings": face_embeddings, "names": face_names}

le = LabelEncoder()
labels = le.fit_transform(data["names"])

with open('recognizer_sir.pickle', "wb") as f:
    pickle.dump(data["embeddings"], f)


print("Training successful")
