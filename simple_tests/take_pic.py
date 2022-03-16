import cv2
import matplotlib.pyplot as plt
import djitellopy
from djitellopy import Tello
import time

tello = Tello()
tello.connect(True)


frame_read = tello.get_frame_read()
cv2.imshow('Live Video', frame_read.frame)
cv2.waitKey(1000)
