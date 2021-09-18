import numpy as np
import cv2
import dlib

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    path = r'C:\Users\....png'      #Edit the path of image
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    image = cv2.imread(path, 0)

    faces = detector(gray)
    # print(faces)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        print(face)
        # position of face
        cv2.rectangle(image, (x1,y1), (x2,y2),(0,255,0),3)


        # cv2.circle(frame, (left_eye, y1), 60, (0, 0, 0), -1)
        # cv2.circle(frame, (right_eye, y1), 60, (0, 0, 0), -1)
        # cv2.circle(frame, (left_eye, y1), 20, (255, 255, 255), -1)
        # cv2.circle(frame, (right_eye, y1), 20, (255, 255, 255), -1)

        # Display the resulting frame
        cv2.imshow('frame', frame) # May be replaced with a background image rather than what webcam catches
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
