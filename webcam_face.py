import cv2, os
cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face = cv2.CascadeClassifier(cascade)
cap = cv2.VideoCapture(0)
assert cap.isOpened(), "No webcam found."

while True:
    ret, frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray, 1.2, 5, minSize=(60,60))
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
    cv2.imshow("Face Detection (ESC to exit)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
