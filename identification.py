import face_recognition
import cv2
import numpy as np
import dlib
import math

# hard coded
eye_width = 80
eye_height = 80



# Get a reference to webcam #0 (the default one)
cap = cv2.VideoCapture(0)
vid_width = cap.get(3)
vid_height = cap.get(4)

path = '/Users/sunny/PycharmProjects/hack_the_north_2021/TERIble edit.png'  # the path of face image
image = cv2.imread(path)
img_width, img_height, channel = image.shape



# move eyeball function
def move_eyeball(x_center, y_center, eye_pos):
    return (eye_pos[0] + round((x_center/img_width)*eye_width) - round(eye_width/2), eye_pos[1] + round((y_center/img_height)*eye_height) - round(eye_height/2))



# Load one user picture and learn how to recognize it.
user1_image = face_recognition.load_image_file("sunny.jpg")
user1_face_encoding = face_recognition.face_encodings(user1_image)[0]

# Load another use sample picture and learn how to recognize it.
# user2_image = face_recognition.load_image_file("biden.jpg")
# user2_face_encoding = face_recognition.face_encodings(user2_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    user1_face_encoding,
    # user2_face_encoding
]
known_face_names = [
    "User",
    # "User2"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    faces = detector(gray)




    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame) #  (top, right, bottom, left)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    # for (top, right, bottom, left), name in zip(face_locations, face_names):
    #     # Scale back up face locations since the frame we detected in was scaled to 1/4 size
    #     top *= 4
    #     right *= 4
    #     bottom *= 4
    #     left *= 4
    #
    #     # Draw a box around the face
    #     cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    #
    #     # Draw a label with a name below the face
    #     cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
    #     font = cv2.FONT_HERSHEY_DUPLEX
    #     cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    #
    # # Display the resulting image
    # cv2.imshow('Supervisor', frame)

    # If the face is user's face, eyeball will move
    if 'Unknown' not in face_names:
        print('yes')

        # face_top = face_locations[0][0]
        # face_right = face_locations[0][1]
        # face_bottom = face_locations[0][2]
        # face_left = face_locations[0][3]
        #
        #
        #
        # x1 = round(img_width - (face_left/vid_width)*img_width)
        # y1 = round((face_top/vid_height)*img_height)-200
        # x2 = round(img_width - (face_right/vid_width)*img_width)
        # y2 = round((face_bottom/vid_height)*img_height)-200

        for face in faces:
            x1 = round(img_width - (face.left() / vid_width) * img_width)
            y1 = round((face.top() / vid_height) * img_height) - 200
            x2 = round(img_width - (face.right() / vid_width) * img_width)
            y2 = round((face.bottom() / vid_height) * img_height) - 200

            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            # print(face)
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


        cv2.imshow('Supervisor', image)
    else:
        print('you are not the authorized user. ')




    # print(face_locations)
    # print(face_names)
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    image = cv2.imread(path)


# Release handle to the webcam
cap.release()
cv2.destroyAllWindows()
