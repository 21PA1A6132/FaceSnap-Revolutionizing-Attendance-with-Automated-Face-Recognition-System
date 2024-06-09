# Automated Attendance Marking System using Facial Recognition

**Project Duration:** Dec 2023 - Feb 2024

**Technologies Used:** Python, Pytorch,OpenCV, DNN, SVM, Pickle, Imutils, Sklearn

## Description
Developed an automated attendance marking system that utilizes advanced facial recognition techniques to streamline and replace traditional manual attendance systems. This project aims to identify and mark attendance by analyzing short videos of the classroom, capturing multiple students simultaneously.

## Key Responsibilities

1. **Data Collection:**
   - Captured 1-minute videos of each student under different conditions (various angles, expressions, lighting).
   - Organized and stored videos in a structured folder hierarchy for efficient processing.

2. **Face Detection and Recognition:**
   - Implemented face detection using the `res10_300x300_ssd_iter_140000.caffemodel` pre-trained model.
   - Extracted facial embeddings with the `openface_nn4.small2.v1.t7` model for precise facial recognition.

3. **Training and Model Development:**
   - Extracted frames from videos and detected faces in each frame.
   - Generated and stored facial embeddings in pickle files (`le.pickle` for labels, `recognizer.pickle` for the model).
   - Trained an SVM classifier using the extracted embeddings to recognize and differentiate between students.

4. **Attendance Marking System:**
   - Captured and processed 10-second test videos of the classroom to identify faces.
   - Compared detected faces against pre-stored facial embeddings to mark attendance.
   - Implemented real-time display of recognized and unrecognized faces, ensuring accurate attendance tracking.

5. **Performance Optimization:**
   - Enhanced the system to handle various lighting conditions and facial expressions for reliable recognition.
   - Improved the efficiency of the system to process multiple faces simultaneously, reducing overall processing time.

## Achievements
- Successfully automated the attendance process, reducing manual effort and increasing accuracy.
- Developed a robust system capable of recognizing students even in varied conditions.
- Demonstrated the system's capability in real-time, showcasing its practical applicability in classroom settings.
- Won the Best Innovation Project Award in the Prajwalan-2k24 Hackathon Conducted by SRKR Engineering College.

## Technical Skills Gained
- Proficiency in using OpenCV for image and video processing.
- Deep understanding of DNN-based face detection and recognition models.
- Experience in training machine learning models (SVM) and managing data with Python.
- Practical knowledge of handling real-time video feeds and implementing efficient image processing techniques.

## Project Outcome
The automated attendance marking system was successfully implemented and tested, showcasing a significant improvement in attendance tracking efficiency and accuracy. The project highlights my capability to develop and integrate machine learning models into practical applications, demonstrating strong problem-solving and technical skills.
