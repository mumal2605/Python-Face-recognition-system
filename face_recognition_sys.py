import numpy as np
import cv2
import cv2
face_cap=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

video_cap=cv2.VideoCapture(0)
if not video_cap.isOpened():
    print("Error: Could not open video.")
    exit()
while True:
    ret, video_data = video_cap.read()
    if not ret:
        print("Error: Could not read frame.")
        continue
    col = cv2.cvtColor(video_data,cv2.COLOR_BGR2GRAY)
    try:
        faces = face_cap.detectMultiScale(
            col,
            scaleFactor=1.1,  
            minNeighbors=7,   
            minSize=(20, 20), 
            flags=cv2.CASCADE_SCALE_IMAGE
        )
    except Exception as e:
        print("Error:", e)
        continue
    for (x,y,w,h) in faces:
        cv2.rectangle(video_data,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("video_live",video_data)
    if cv2.waitKey(10)==ord("a"):
        break
video_cap.release()
cv2.destroyAllWindows()

video_cap=cv2.VideoCapture(0)
while True:
   ret, video_data = video_cap.read()
   cv2.imshow("video_live",video_data)
   if cv2.waitKey(10)==ord("a"):
        break
video_cap.release()