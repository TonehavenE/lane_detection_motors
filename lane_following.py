from math import isclose

import numpy as np

from Line import *


def merge_lane_lines(lanes: list[tuple[Line, Line]], height: int) -> list[Line]:
    """Combines the lines of each lane to produce a single center line for each.

    ### Parameters
    - lanes (list[tuple[Line, Line]]): the list of lanes
    - height (int): the height of the image

    ### Returns
    - list[Line]: the list of center lines
    """
    center_lines = []  # output list
    for lane in lanes:
        # the center line is defined as the line between the midpoint of the x-intercepts and the midpoint of the intercepts with the top of the image
        center_lines.append(
            Line(
                (lane[0].x_intercept + lane[1].x_intercept) / 2,
                height,
                (lane[0].x(0) + lane[1].x(0)) / 2,
                0,
            )
        )
    return center_lines


def pick_center_line(center_lines: list[Line], width: int) -> Line:
    """Picks the center line (line closest to center of image) from a list of lines

    ### Parameters
    - center_lines (list[Line]): the list of lines
    - width (int): width of the image

    ### Returns
    - Line: the line closest to center
    """

    def closest(lines: list[Line], k: int) -> Line:
        """The element in the list closest to `k`.

        ### Parameters
        - lines (list[Line]): the list of lines
        - k (int): the value to compare against

        ### Returns
        - Line: the Line with an x-intercept closest to `k`
        """
        if len(lines) > 0:
            x = np.asarray([line.x_intercept for line in lines])
            idx = (np.abs(x - k)).argmin()
            return lines[idx]

    closest_line = closest(center_lines, width / 2)
    return closest_line


def angle_from_line(line: Line, angle_tol: int = 5) -> float:
    if line:
        slope = line.slope
    else:
        return 0

    angle = np.arctan(-1 / slope)
    if isclose(np.rad2deg(angle), 0, abs_tol=angle_tol):
        angle = 0  # round to 0 if within 5 degrees

    return angle


def movement_from_line(
    line: Line, width: int, forward_tol: int = 50
) -> tuple[float, float]:
    """Returns the suggestion of movement from a line.

    Args:
        line (Line): the line the robot is trying to follow
        width (int): the width of the image
        forward_tol (int, optional): the tolerance around which you go forward. Defaults to 50.

    Returns:
        tuple[float, float]: the longitudinal and lateral error
    """
    mid = width / 2
    mid_left = mid - forward_tol
    mid_right = mid + forward_tol
    lateral = 0
    longitudinal = 0

    if line:
        x_intercept = np.clip(line.x_intercept, 0, width)
    else:
        return (lateral, longitudinal)

    if x_intercept > mid_right:
        # the lane center is right of the middle
        lateral = ((x_intercept - mid) / width) * 100
        longitudinal = 0

    elif x_intercept < mid_left:
        # the lane center is left of the middle
        lateral = ((x_intercept - mid) / width) * 100
        longitudinal = 0

    else:
        # the lane center is in the middle region
        lateral = 0
        longitudinal = 100

    return (longitudinal, lateral)


def error_from_line(
    line: Line, width: int, forward_tol: int = 50, angle_tol: int = 5
) -> tuple[str, str]:
    """Suggests which direction the AUV should move in based off a line.

    ### Parameters
    - line (Line): the center of the lane closest to the AUV
    - width (int): the width of the image
    - forward_tol (int, optional): the number of pixels around the middle where the AUV should continue straight. Defaults to 50.
    - angle_tol (int, optional): the range of angles that are considered straight.

    ### Returns
    - tuple[str, str]: (movement_direction, turn_direction)
    """
    yaw = angle_from_line(line, angle_tol)
    longitudinal, lateral = movement_from_line(line, width, forward_tol)
    return (longitudinal, lateral, yaw)
