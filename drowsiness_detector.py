import cv2
import winsound
import os
import urllib.request

FACE_XML = "haarcascade_frontalface_default.xml"
EYE_XML = "haarcascade_eye.xml"

FACE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
EYE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml"

if not os.path.exists(FACE_XML):
    urllib.request.urlretrieve(FACE_URL, FACE_XML)

if not os.path.exists(EYE_XML):
    urllib.request.urlretrieve(EYE_URL, EYE_XML)

face_cascade = cv2.CascadeClassifier(FACE_XML)
eye_cascade = cv2.CascadeClassifier(EYE_XML)

cap = cv2.VideoCapture(0)

closed_eyes_frames = 0
CLOSED_FRAMES_THRESHOLD = 12
beep_interval = 0

print("[INFO] Starting video stream... Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Could not access webcam.")
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    eyes_detected = False

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 140, 0), 2)

        roi_gray = gray[y : y + int(h * 0.60), x : x + w]
        roi_color = frame[y : y + int(h * 0.60), x : x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

        if len(eyes) > 0:
            eyes_detected = True
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    if len(faces) > 0:
        if not eyes_detected:
            closed_eyes_frames += 1
        else:
            closed_eyes_frames = 0

        if closed_eyes_frames >= CLOSED_FRAMES_THRESHOLD:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 12)
            cv2.rectangle(frame, (20, 20), (430, 110), (0, 0, 0), -1)
            cv2.putText(frame, "DROWSINESS ALERT!", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
            cv2.putText(frame, "STATUS: UNRESPONSIVE", (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            beep_interval += 1
            if beep_interval % 3 == 0:
                winsound.Beep(2500, 150)
        else:
            beep_interval = 0
            cv2.rectangle(frame, (20, 20), (320, 80), (0, 0, 0), -1)
            cv2.putText(frame, f"Closed Count: {closed_eyes_frames}/{CLOSED_FRAMES_THRESHOLD}", (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        closed_eyes_frames = 0
        beep_interval = 0
        cv2.rectangle(frame, (20, 20), (300, 80), (0, 0, 0), -1)
        cv2.putText(frame, "NO FACE DETECTED", (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    cv2.imshow("Driver Drowsiness Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()