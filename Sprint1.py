import numpy as np
import cv2

cap = cv2.VideoCapture(0)  #selects camera

while True: #continues until the conditions of pressing the 'q' key is met. 
    ret, frame = cap.read()

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release() #releases the camera
cv2.destroyAllWindows() #closes all windows
