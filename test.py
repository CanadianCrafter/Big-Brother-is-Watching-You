import numpy as np
import math
import numpy as np

def move_eyeball(x_center, y_center, eye_pos):
    theta = np.arctan([(y_center - eye_pos[1]) / (x_center - eye_pos[0])])[0]
    if (y_center > eye_pos[1]) and (x_center > eye_pos[0]):
        theta = theta + math.pi
    if (y_center < eye_pos[1]) and (x_center > eye_pos[0]):
        theta = theta - math.pi
    # 0->pi degree positive, -pi-> 0 negative, 0 is the direction of right x-axis like in math
    dis = 10  # distance from eyeball to the center of the eye
    y_dis = math.sin(theta) * dis
    x_dis = math.cos(theta) * dis
    return round(eye_pos[0] + x_dis), round(eye_pos[1] - y_dis)

print(move_eyeball(300,200,(225, 255)))
