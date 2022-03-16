from djitellopy import Tello
import cv2
import time


drone = Tello()
drone.connect(False)
drone.for_back_velocity = 0
drone.left_right_velocity = 0
drone.up_down_velocity = 0
drone.yaw_velocity = 0
drone.speed = 0

# print(drone.get_battery())

drone.streamoff()
drone.streamon()

while True:
    frame_read = drone.get_frame_read()
    frame = frame_read.frame
    # img = cv2.resize(myFrame, (width, height))

    cv2.imshow("Output", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
