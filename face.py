import numpy as np
import cv2
import dlib
import math

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()


# def move_eyeball(x_center, y_center, eye_pos):
#     theta = np.arctan([(y_center - eye_pos[1]) / (x_center - eye_pos[0])])
#     if (y_center - eye_pos[1] > 0) and (x_center - eye_pos[0] < 0):
#         theta = theta + math.pi
#     if (y_center - eye_pos[1] < 0) and (x_center - eye_pos[0] > 0):
#         theta = theta - math.pi
#     # 0->180 degree positive, -180->0 negative, 0 is the direction of right x-axis like in math
#     dis = 11  # distance from eyeball to the center of the eye
#     y_dis = math.sin(theta) * dis
#     x_dis = math.cos(theta) * dis
#     return (eye_pos[1] + y_dis, eye_pos[0] + x_dis)


while (True):
    path = r'C:\Users\zbh26\Desktop\face.jpg'  # the path of face image (people.jpg)
    image = cv2.imread(path, 0)

    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    # print(faces)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        print(face)
        # position of face
        # cv2.rectangle(image, (x1,y1), (x2,y2),(0,255,0),3)

        left_eye_pos = (225, 255)  # change the left_eye_pos to map the face position to eyeballs
        # (220, 250) is the coordinate of left eye(left-top corner)
        right_eye_pos = (325, 255)  # change the right_eye_pos to map the face position to eyeballs
        # (320, 250) is the coordinate of right eye(left-top corner)

        cv2.circle(image, left_eye_pos, 15, (0, 0, 0), -1)
        # left eyeballs
        cv2.circle(image, left_eye_pos, 3, (255, 255, 255), -1)

        cv2.circle(image, right_eye_pos, 15, (0, 0, 0), -1)
        # right eyeballs
        cv2.circle(image, right_eye_pos, 3, (255, 255, 255), -1)

        # Display the resulting frame
    cv2.imshow('Supervisor', image)  # May be replaced with a background image rather than what webcam catches

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
