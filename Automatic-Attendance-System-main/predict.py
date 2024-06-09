import numpy as np
import pickle
import os
import cv2
import time
import datetime
import imutils

curr_path = os.getcwd()

print("Loading face detection model")
proto_path = os.path.join(curr_path, 'deploy.prototxt')
model_path = os.path.join(curr_path, 'res10_300x300_ssd_iter_140000.caffemodel')
face_detector = cv2.dnn.readNetFromCaffe(prototxt=proto_path, caffeModel=model_path)

print("Loading face recognition model")
recognition_model = os.path.join(curr_path, 'openface_nn4.small2.v1.t7')
face_recognizer = cv2.dnn.readNetFromTorch(model=recognition_model)

recognizer = pickle.loads(open('recognizer.pickle', "rb").read())
le = pickle.loads(open('le.pickle', "rb").read())

print("Starting webcam")
vs = cv2.VideoCapture("D:\FaceSnap\Face_Data_III_AIML\21PA1A6101\21PA1A6101.mp4")  # Use 0 for the default camera
time.sleep(1)
fps_start = datetime.datetime.now()
fps = 0
total_frames = 0

unknown_faces = []  # List to store unknown faces
known_faces = set()  # Set to store known faces

# Initialize variables to keep track of the position of unknown faces
unknown_faces_x = 0
unknown_faces_y = 0
unknown_faces_margin = 10  # Margin between unknown faces
max_unknown_faces_per_row = 3  # Maximum number of unknown faces per row

while True:
    ret, frame = vs.read()
    frame = imutils.resize(frame, width=1400)
    total_frames = total_frames + 1

    fps_end = datetime.datetime.now()

    time_diff = fps_end - fps_start
    if time_diff.seconds == 0:
        fps = 0.0
    else:
        fps = (total_frames / time_diff.seconds)

    fps_text = "FPS : {:.2f}".format(fps)

    cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

    (h, w) = frame.shape[:2]

    image_blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)

    face_detector.setInput(image_blob)
    face_detections = face_detector.forward()

    for i in range(0, face_detections.shape[2]):
        confidence = face_detections[0, 0, i, 2]

        if confidence >= 0.98:
            box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]

            if face.size == 0 or face.shape[0] < 1 or face.shape[1] < 1:
                continue  # Skip this iteration and move to the next face

            (fH, fW) = face.shape[:2]

            face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), True, False)

            face_recognizer.setInput(face_blob)
            vec = face_recognizer.forward()

            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            if proba < 0.98:
                name = "Unknown"
                if len(unknown_faces) >= max_unknown_faces_per_row:
                    # Start a new row
                    unknown_faces_y += unknown_faces[0][0].shape[0] + unknown_faces_margin
                    unknown_faces_x = 0

                # Resize unknown face
                if len(unknown_faces) > 0:
                    resized_face = cv2.resize(face, (unknown_faces[0][0].shape[1], unknown_faces[0][0].shape[0]))
                else:
                    resized_face = cv2.resize(face, (100, 100))  # Default size if unknown_faces is empty

                # Display unknown face in a non-overlapping manner
                unknown_faces.append((resized_face, unknown_faces_x, unknown_faces_y))
                unknown_faces_x += resized_face.shape[1] + unknown_faces_margin  # Update x position

                text = "{}: {:.2f}".format(name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("Webcam Feed", frame)

    if len(unknown_faces) > 0:
        unknown_faces_display = np.zeros((unknown_faces[0][0].shape[0], unknown_faces_x, 3), dtype=np.uint8)

        for unknown_face, x, y in unknown_faces:
            h, w = unknown_face.shape[:2]
            unknown_faces_display[y:y + h, x:x + w] = unknown_face

        cv2.imshow("Unknown Faces", unknown_faces_display)
        unknown_faces = []  
        unknown_faces_x = 0  

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

vs.release()
cv2.destroyAllWindows()
