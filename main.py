from threading import Thread, Event
from time import sleep
import numpy as np
from pid import PID
from video import Video
from bluerov_interface import BlueROV
from pymavlink import mavutil
import video_maker

# TODO: import your processing functions
from pid_from_frame import *

print("Main started!")

# Create the video objectE
video = Video()
FPS = 2
RC_SLEEP = 0.1

# Create the PID object
PIDLateral = PID(50, 0, -6, 100)
PIDLongitudinal = PID(50, 0, 0, 100)
PIDYaw = PID(30, 0, 0, 100)
# Create the mavlink connection
mav_comn = mavutil.mavlink_connection("udpin:0.0.0.0:14550")
# Create the BlueROV object
bluerov = BlueROV(mav_connection=mav_comn)
# where to write frames to. If empty string, no photos are written!
output_path = "frames/frame"  # {n}.jpg

window_frame_count = 11  # the number of center lines to store for calculating median
max_misses = 6  # the number of frames without receiving a center line before the robot starts to spin

frame = None
frame_available = Event()
frame_available.set()

longitudinal_power = 0
lateral_power = 0
yaw_power = 0


def _get_frame():
    global frame
    global yaw_power
    global lateral_power
    global longitudinal_power
    count = 0
    center_lines = []

    while not video.frame_available():
        print("Waiting for frame...")
        sleep(0.01)

    try:
        while True:
            if video.frame_available():
                frame = video.frame()
                print("frame found")
                # cv2.imwrite("camera_stream.jpg", frame)
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if type(frame) == np.ndarray:
                    try:
                        center_line = line_from_frame(frame[20 : -1, 20 : -1], x_intercept_tolerance=25, lanes_x_tolerance = 300, lanes_y_tolerance = 100, lanes_darkness_threshold=10)
                        center_lines.append(center_line)
                        if len(center_lines) > window_frame_count:
                            center_lines.pop(0)

                        if center_lines.count(None) > 5:
                            print("No lines found in last 5 images. ")
                            yaw_power = 10  # %
                            lateral_power = 0
                            longitudinal_power = 0
                            # continue
                        else:
                            good_lines = list(
                                filter(lambda line: line is not None, center_lines)
                            )
                            if len(good_lines) > 1:
                                # print(good_lines)
                                # print(len(good_lines) // 2)
                                good_lines.sort(key=lambda x: x.slope)
                                middle_line = good_lines[len(good_lines) // 2]
                                (
                                    longitudinal_power,
                                    lateral_power,
                                    yaw_power,
                                ) = pid_from_line(
                                    middle_line,
                                    PIDLateral,
                                    PIDLongitudinal,
                                    PIDYaw,
                                    frame.shape[1],
                                )
                                if output_path != "":
                                    cv2.imwrite(
                                        f"{output_path}{count}.jpg",
                                        draw_frame(
                                            frame,
                                            middle_line,
                                            longitudinal_power,
                                            lateral_power,
                                            yaw_power,
                                        ),
                                    )
                        print(f"{yaw_power = }")
                        print(f"{longitudinal_power = }")
                        print(f"{lateral_power = }")
                        cv2.imwrite(f"frames/frame{count}.jpg", video_maker.render_frame(frame))
                    except Exception as e:
                        print(f"caught: {e}")
                        yaw_power = 0
                        lateral_power = 0
                        longitudinal_power = 0

                print(frame.shape)
                count += 1

                sleep(1 / FPS)
    except KeyboardInterrupt:
        return


def _send_rc():
    # on first startup, set everything to neutral
    # bluerov.set_rc_channels_to_neutral()
    bluerov.set_lateral_power(0)
    bluerov.set_vertical_power(0)
    bluerov.set_yaw_rate_power(0)
    bluerov.set_longitudinal_power(0)

    while True:
        bluerov.arm()

        mav_comn.set_mode(19)  # Remember to comment this out depending on the robot!
        # --- ^^^ CHANGE THIS ^^^ --- 19 for old, 0 for new

        bluerov.set_longitudinal_power(int(longitudinal_power))
        bluerov.set_lateral_power(int(lateral_power))
        bluerov.set_yaw_rate_power(int(yaw_power))
        sleep(RC_SLEEP)


def main():
    # Start the video thread
    video_thread = Thread(target=_get_frame)
    video_thread.start()

    # # Start the RC thread
    rc_thread = Thread(target=_send_rc)
    rc_thread.start()

    # Main loop
    try:
        while True:
            mav_comn.wait_heartbeat()
    except KeyboardInterrupt:
        video_thread.join()
        rc_thread.join()
        bluerov.set_lights(False)
        bluerov.disarm()
        print("Exiting...")


if __name__ == "__main__":
    main()
