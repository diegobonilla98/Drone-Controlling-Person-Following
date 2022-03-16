from djitellopy import Tello

tello = Tello()

tello.connect(False)
tello.takeoff()

tello.move_left(100)
tello.rotate_counter_clockwise(90)
tello.move_forward(100)

tello.land()
