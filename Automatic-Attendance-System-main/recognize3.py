import numpy as np
import pickle
import os
import cv2
import time
import datetime
import imutils
import tkinter as tk

# Function to get screen width and height
def get_screen_resolution():
    root = tk.Tk()
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    root.destroy()
    return width, height

curr_path = os.getcwd()

def is_exp(n):
    n=str(n)
    return 'e' in n

print("Loading face detection model")
proto_path = os.path.join(curr_path, 'deploy.prototxt')
model_path = os.path.join(curr_path, 'res10_300x300_ssd_iter_140000.caffemodel')
face_detector = cv2.dnn.readNetFromCaffe(prototxt=proto_path, caffeModel=model_path)

print("Loading face recognition model")
recognition_model = os.path.join(curr_path, 'openface_nn4.small2.v1.t7')
face_recognizer = cv2.dnn.readNetFromTorch(model=recognition_model)

recognizer = pickle.loads(open('pickle\\recognize 5.pickle', "rb").read())
le = pickle.loads(open('pickle\le 5.pickle', "rb").read())

print("Starting webcam")
vs = cv2.VideoCapture('demotest3.mp4')  # Use 0 for the default camera
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
d={}
for i in le.classes_:
    d[i]=0
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
            
            newpreds = preds.tolist()
            
            j = newpreds.index(max(newpreds))
            name = le.classes_[j]
            
            if newpreds[j] > 0.98 and not(is_exp(newpreds[j])):
                if(d[le.classes_[j]]!=-1):
                    d[le.classes_[j]]+=1
                
                if(d[le.classes_[j]]>4):
                    d[le.classes_[j]]=-1
                    print("ANS ",name)

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

    # Get screen resolution
    screen_width, screen_height = get_screen_resolution()

    # Limiting the window size to the screen size
    if frame.shape[1] > screen_width:
        frame = imutils.resize(frame, width=screen_width)
    if frame.shape[0] > screen_height:
        frame = imutils.resize(frame, height=screen_height)

    cv2.imshow("Webcam Feed", frame)

    # Display unknown faces in a separate window
    if len(unknown_faces) > 0:
        unknown_faces_display = np.zeros((unknown_faces[0][0].shape[0], unknown_faces_x, 3), dtype=np.uint8)

        for unknown_face, x, y in unknown_faces:
            h, w = unknown_face.shape[:2]
            unknown_faces_display[y:y + h, x:x + w] = unknown_face

        cv2.imshow("Unknown Faces", unknown_faces_display)
        unknown_faces = []  # Clear the list after displaying
        unknown_faces_x = 0  # Reset x position for the next iteration

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
total=0
print("Total =",list(d.values()).count(-1))
vs.release()
cv2.destroyAllWindows()
