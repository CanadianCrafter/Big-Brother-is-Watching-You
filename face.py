import numpy as np
import cv2
import dlib
import math

#----------------------------

#hard coded
eye_width = 80
eye_height = 80
your_image_path = "./tim.jpg"

#----------------------------

cap = cv2.VideoCapture(0)
vid_width = cap.get(3)
vid_height = cap.get(4)

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("./shape_predictor_5_face_landmarks.dat")
model = dlib.face_recognition_model_v1("./dlib_face_recognition_resnet_model_v1.dat")

path = '.\TERIble edit.png'  # the path of face image
image = cv2.imread(path)
img_width, img_height, channel = image.shape

def move_eyeball(x_center, y_center, eye_pos):
    return (eye_pos[0] + round((x_center/img_width)*eye_width) - round(eye_width/2), eye_pos[1] + round((y_center/img_height)*eye_height) - round(eye_height/2))

def parse_face(face):
    x1 = round(img_width - (face.left()/vid_width)*img_width)
    y1 = round((face.top()/vid_height)*img_height)-200
    x2 = round(img_width - (face.right()/vid_width)*img_width)
    y2 = round((face.bottom()/vid_height)*img_height)-200

    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2

    #print(face)
    # position of face
    # cv2.rectangle(image, (x1,y1), (x2,y2),(0,255,0),3)

    left_eye_pos = [350, 320]  # change the left_eye_pos to map the face position to eyeballs
    # (220, 250) is the coordinate of left eye(left-top corner)
    right_eye_pos = [515, 330]  # change the right_eye_pos to map the face position to eyeballs
    # (320, 250) is the coordinate of right eye(left-top corner)

    left_eye_pos = move_eyeball(x_center, y_center, left_eye_pos)
    right_eye_pos = move_eyeball(x_center, y_center, right_eye_pos)
    cv2.circle(image, left_eye_pos, 15, (0, 0, 0), -1)
    cv2.circle(image, right_eye_pos, 15, (0, 0, 0), -1)

def calculate_euclidean_distance(vector1, vector2):
    euclidean_distance = vector1 - vector2
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

only_detect_me = dlib.load_rgb_image(your_image_path)
detected = detector(only_detect_me, 1)
if len(detected) > 1:
    print("Error, too many faces detected")
    exit(1)
only_detect_me_shape = sp(only_detect_me, detected[0])
only_detect_me_face = dlib.get_face_chip(only_detect_me, only_detect_me_shape)
only_detect_me_vector = model.compute_face_descriptor(only_detect_me_face)
only_detect_me_vector = np.array(only_detect_me_vector)

cv2.namedWindow("Supervisor", cv2.WINDOW_NORMAL)
while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    face_tracked = False

    if len(faces) == 1:
        face = faces[0]
        parse_face(face)
        face_tracked = True
    else:
        for face in faces:
            face_shape = sp(image, face)
            face_detected = dlib.get_face_chip(image, face_shape)
            face_detected_vector = model.compute_face_descriptor(face_detected)
            face_detected_vector = np.array(face_detected_vector)
            
            dis = calculate_euclidean_distance(only_detect_me_vector, face_detected_vector)
            #print(dis)
            if dis < 0.7:
                parse_face(face)
                face_tracked = True
                break
    
    if face_tracked == False:
        left_eye_pos = [350, 320]  # change the left_eye_pos to map the face position to eyeballs
        # (220, 250) is the coordinate of left eye(left-top corner)
        right_eye_pos = [515, 330]  # change the right_eye_pos to map the face position to eyeballs
        # (320, 250) is the coordinate of right eye(left-top corner)
        cv2.circle(image, left_eye_pos, 15, (0, 0, 0), -1)
        cv2.circle(image, right_eye_pos, 15, (0, 0, 0), -1)

    
    cv2.imshow("Supervisor", image)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty("Supervisor", cv2.WND_PROP_VISIBLE) < 1:        
        break        

    image = cv2.imread(path)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
