import cv2
import numpy as np
from face_det_utils import detect_face
from djitellopy import Tello
import math


tello = Tello()
tello.connect()

tello.TAKEOFF_TIMEOUT = 50
tello.RESPONSE_TIMEOUT = 30
tello.FRAME_GRAB_TIMEOUT = 30
tello.for_back_velocity = 0
tello.left_right_velocity = 0
tello.up_down_velocity = 0
tello.yaw_velocity = 0
tello.speed = 0

tello.streamon()
camera = tello.get_frame_read()

is_first_frame = True

every_5_sec_command = 0

frame = None
tello.takeoff()
canvas = None

while True:
    frame = camera.frame

    every_5_sec_command += 1
    if every_5_sec_command > 400:
        print("Autocommand")
        every_5_sec_command = 0
        tello.move_up(20)
        tello.move_down(20)

    if frame is not None:
        canvas = frame.copy()
        boxes, labels, probs = detect_face(frame)

        for i in range(boxes.shape[0]):
            box = boxes[i, :]
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            offset = 100
            obj_center_x = (x1 + x2) // 2
            obj_center_y = (y1 + y2) // 2
            screen_center_x = frame.shape[1] // 2
            screen_center_y = frame.shape[0] // 4
            dist = math.sqrt(math.pow(obj_center_x - screen_center_x, 2) + math.pow(obj_center_y - screen_center_y, 2))
            cv2.arrowedLine(canvas, (obj_center_x, obj_center_y), (screen_center_x, screen_center_y), (0, 0, 255) if dist > offset else (0, 255, 0), 2)

            x_off = obj_center_x - screen_center_x
            y_off = obj_center_y - screen_center_y
            if area < 12000:
                every_5_sec_command = 0
                tello.move_forward(20)
            if area > 20000:
                every_5_sec_command = 0
                tello.move_back(20)
            if x_off < -offset:
                every_5_sec_command = 0
                tello.move_left(20)
            if x_off > offset:
                every_5_sec_command = 0
                tello.move_right(20)
            if y_off < -offset:
                every_5_sec_command = 0
                tello.move_up(20)
            if y_off > offset:
                every_5_sec_command = 0
                tello.move_down(20)

    if canvas is not None:
        cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
tello.land()
