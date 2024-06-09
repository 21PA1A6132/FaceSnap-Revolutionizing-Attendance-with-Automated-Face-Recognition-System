import numpy as np
import pickle
import os
import cv2
import time
import datetime
import imutils
import json

curr_path = os.getcwd()

def is_exp(n):
    n = str(n)
    return 'e' in n


proto_path = os.path.join(curr_path, 'deploy.prototxt')
model_path = os.path.join(curr_path, 'res10_300x300_ssd_iter_140000.caffemodel')
face_detector = cv2.dnn.readNetFromCaffe(prototxt=proto_path, caffeModel=model_path)


recognition_model = os.path.join(curr_path, 'openface_nn4.small2.v1.t7')
face_recognizer = cv2.dnn.readNetFromTorch(model=recognition_model)

recognizer = pickle.loads(open('recognize 5.pickle', "rb").read())
le = pickle.loads(open('le 5.pickle', "rb").read())


vs = cv2.VideoCapture('uploads\\temp.mp4')  # Use 0 for the default camera
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
d = {}
for i in le.classes_:
    d[i]=-500
d['21PA1A6103']=0
d['21PA1A6132']=0
d['21PA1A6164']=0

try:
    while True:
        ret, frame = vs.read()
        if not ret:
            break

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

        image_blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 160.0, 186.0), False, False)

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

                newpreds = preds.tolist()

                j = newpreds.index(max(newpreds))
                name = le.classes_[j]

                if newpreds[j] > 0.98 and not(is_exp(newpreds[j])):
                    if(d[le.classes_[j]]!=-1):
                        d[le.classes_[j]]+=1

                    if(d[le.classes_[j]]>4):
                        d[le.classes_[j]]=-1
                        

                if preds[j] < 0.98:
                    name = "Unknown"
                    if len(unknown_faces) > 0:
                        # Resize unknown face
                        resized_face = cv2.resize(face, (unknown_faces[0][0].shape[1], unknown_faces[0][0].shape[0]))
                        # Display unknown face in a non-overlapping manner
                        unknown_faces.append((resized_face, unknown_faces_x, unknown_faces_y))
                        unknown_faces_x += resized_face.shape[1] + unknown_faces_margin  # Update x position

                    else:
                        unknown_faces.append((face, unknown_faces_x, unknown_faces_y))
                        unknown_faces_x += face.shape[1] + unknown_faces_margin  # Update x position

                    text = "{}: {:.2f}".format(name, preds[j] * 100)
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                    cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Display frame
        cv2.imshow("Webcam Feed", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
finally:
    json_string = json.dumps(d, indent=4)  # indent=4 for pretty printing

    # Save JSON string to a file
    with open("data.json", "w") as json_file:
        json_file.write(json_string)
    vs.release()
    cv2.destroyAllWindows()
