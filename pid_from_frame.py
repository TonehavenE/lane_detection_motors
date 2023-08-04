from multiprocessing import Process

from lane_detection import *
from lane_following import *
from pid import *


def line_from_frame(
    frame: np.ndarray,
    kernel_size=7,
    bw_threshold=70,
    edges_t1=50,
    edges_t2=100,
    lines_threshold=50,
    min_line_length=100,
    max_line_gap=20,
    slope_tolerance=0.1,
    x_intercept_tolerance=50,
    lanes_x_tolerance=25,
    lanes_y_tolerance=25,
    lanes_darkness_threshold=10,
) -> Line:
    center_line = None

    # Process image
    sliced = split(frame)
    height = sliced.shape[0]
    width = sliced.shape[1]
    gray = to_gray(sliced)
    blurred = to_blurred(gray, kernel_size=kernel_size)
    bw = to_bw(blurred, t=bw_threshold)

    # Edge/line detection
    edges = find_edges(bw, t1=edges_t1, t2=edges_t2)
    lines = find_lines(edges, threshold=lines_threshold, min_line_length=min_line_length, max_line_gap=max_line_gap)
    if len(lines) > 1:
        grouped_lines = group_lines(
            lines,
            height,
            slope_tolerance=slope_tolerance,
            x_intercept_tolerance=x_intercept_tolerance,
        )  # group lines
        merged_lines = merge_lines(
            grouped_lines, height, width
        )  # merge groups of lines
        # Lane Detection
        lanes = detect_lanes(bw, merged_lines, lanes_x_tolerance, lanes_y_tolerance, lanes_darkness_threshold)
        # Lane picking
        center_lines = merge_lane_lines(lanes, height)  # find the center of each lane
        center_line = pick_center_line(center_lines, width)  # find the closest lane
        # print(f"{center_lines = }, {center_line = }")

        # cv2.imwrite("testing/bw.jpg", bw)
        # cv2.imwrite("testing/edges.jpg", edges)
        # cv2.imwrite("testing/lines.jpg", draw_lines(frame, lines, offset=True))
        # cv2.imwrite("testing/merged.jpg", draw_lines(frame, merged_lines, offset=True))
        # cv2.imwrite("testing/lanes.jpg", draw_lanes(frame, lanes, offset=True))
        # cv2.imwrite("testing/center.jpg", draw_lines(frame, center_lines, offset=True))

    return center_line


def pid_from_line(
    center_line, lateral_pid, longitudinal_pid, yaw_pid, width
):
    """Returns PID output from a center line.

    Args:
        center_line (Line): the center line
        lateral_pid (PID): the lateral PID
        longitudinal_pid (PID): the straight PID
        yaw_pid (PID): the yaw PID
        width (the image width): width

    Returns:
        (longitudinal, lateral, yaw): longitudinal, lateral, yaw values
    """
    longitudinal = 0
    lateral = 0
    yaw = 0
    if center_line is not None:
        (longitudinal_error, lateral_error, yaw_error) = error_from_line(
            center_line, width, angle_tol=2
        )

        longitudinal = longitudinal_pid.update(longitudinal_error)
        lateral = lateral_pid.update(lateral_error)
        yaw = yaw_pid.update(yaw_error)

    else:
        # we didn't find anything, so just turn to try and find something
        yaw = yaw_pid.update(np.pi / 4)  # turn 45 degrees

    return (longitudinal, lateral, yaw)


def draw_frame(frame, center_line, longitudinal, lateral, yaw):
    yaw_degs = np.rad2deg(yaw)
    if longitudinal == forward_power:
        text = f"Move forward: {longitudinal:.2f} | Turn: {yaw_degs:.2f}"
    elif lateral != 0:
        text = f"Move lateral: {lateral:.2f}% | Turn: {yaw_degs:.2f}"
    else:
        text = f"Don't move"

    # Drawing
    frame = draw_lines(frame, [center_line], (0, 0, 255), offset=True)
    frame = cv2.putText(
        frame,
        text,
        (0, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def process_frame(
    frame, lateral_pid, longitudinal_pid, yaw_pid, output_path=""
) -> tuple[float, float, float]:
    """Applies a sequence of image filtering and processing to suggest PID movements to center the lane.

    ### Parameters
        frame: the frame to process/render
        lateral_pid (PID): the horizontal PID control object
        longitudinal_pid (PID): the forward/backward PID control object
        yaw_pid (PID): the yaw PID control object

    ### Returns
        (float, float, float): the percent outputs for each of longitudinal, lateral, and yaw
    """
    lateral = 0
    longitudinal = 0
    yaw = 0
    # Process image
    sliced = split(frame)
    height = sliced.shape[0]
    width = sliced.shape[1]
    gray = to_gray(sliced)
    blurred = to_blurred(gray)
    bw = to_bw(blurred)

    # Edge/line detection
    edges = find_edges(bw)
    lines = find_lines(edges)
    if len(lines) > 1:
        grouped_lines = group_lines(
            lines, height, slope_tolerance=0.1, x_intercept_tolerance=50
        )  # group lines
        merged_lines = merge_lines(
            grouped_lines, height, width
        )  # merge groups of lines

        # Lane Detection
        lanes = detect_lanes(bw, merged_lines, 500, 200, 10)

        # Lane picking
        center_lines = merge_lane_lines(lanes, height)  # find the center of each lane
        center_line = pick_center_line(center_lines, width)  # find the closest lane
        (longitudinal_error, lateral_error, yaw_error) = error_from_line(
            center_line, width
        )

        longitudinal = longitudinal_pid(longitudinal_error)
        lateral = lateral_pid(lateral_error)
        yaw = yaw_pid(yaw_error)

    else:
        # we didn't find anything, so just turn to try and find something
        yaw = yaw_pid(np.pi / 4)  # turn 45 degrees
    if output_path != "":
        yaw_degs = np.rad2deg(yaw_error)
        if longitudinal == 100:
            text = f"Move forward: {longitudinal:.2f} | Turn: {yaw_degs:.2f}"
        elif lateral != 0:
            text = f"Move lateral: {lateral:.2f}% | Turn: {yaw_degs:.2f}"
        else:
            text = f"Don't move"

        # Drawing
        frame = draw_lanes(frame, lanes, offset=True)
        frame = draw_lines(frame, [center_line], (0, 0, 255), offset=True)
        frame = cv2.putText(
            frame,
            text,
            (0, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # image writing
        cv2.imwrite(output_path, frame)

    return (longitudinal, lateral, yaw)
