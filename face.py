import numpy as np
import cv2
import dlib
import math

cap = cv2.VideoCapture(0)
vid_width = cap.get(3)
vid_height = cap.get(4)

detector = dlib.get_frontal_face_detector()


def move_eyeball(x_center, y_center, eye_pos):
    return (eye_pos[0] + round((x_center/1034)*50) - 25, eye_pos[1] + round((y_center/979)*70) - 35)

while (True):
    path = '.\TERIble edit.png'  # the path of face image (people.jpg)
    image = cv2.imread(path, 0)

    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    # print(faces)
    for face in faces:
        x1 = round(1034 - (face.left()/vid_width)*1034)
        y1 = round((face.top()/vid_height)*979)
        x2 = round(1034 - (face.right()/vid_width)*1034)
        y2 = round((face.bottom()/vid_height)*979)

        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2

        #print(face)
        # position of face
        cv2.rectangle(image, (x1,y1), (x2,y2),(0,255,0),3)

        left_eye_pos = [350, 310]  # change the left_eye_pos to map the face position to eyeballs
        # (220, 250) is the coordinate of left eye(left-top corner)
        right_eye_pos = [515, 320]  # change the right_eye_pos to map the face position to eyeballs
        # (320, 250) is the coordinate of right eye(left-top corner)

        left_eye_pos = move_eyeball(x_center, y_center, left_eye_pos)
        right_eye_pos = move_eyeball(x_center, y_center, right_eye_pos)
        cv2.circle(image, left_eye_pos, 15, (0, 0, 0), -1)
        cv2.circle(image, right_eye_pos, 15, (0, 0, 0), -1)

        # Display the resulting frame
    cv2.imshow('Supervisor', image)  # May be replaced with a background image rather than what webcam catches

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
