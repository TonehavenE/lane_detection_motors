import math
from itertools import combinations
from random import randrange

import cv2
import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from typing import Union

from Line import *


# ================
# Helper Functions
# ================
def draw_lines(
    img: npt.NDArray[any],
    lines: list[Line],
    color: tuple[int, int, int] = (0, 255, 0),
    random=False,
    offset=False,
) -> npt.NDArray[any]:
    """Draws each line in lines on `img`

    ### Parameters
    - img (npt.NDArray[any]): the image to draw lines on (copied!)
    - lines (list[Line]): the list of lines to draw
    - color (tuple[int, int, int], optional): the color to draw the lines as. Defaults to (0, 255, 0).
    - random (bool, optional): whether or not to pick a random color for each line. Defaults to False.
    - offset (bool, optional): whether or not to offset each line by height/2. Defaults to False.

    ### Returns
    - npt.NDArray[any]: a copy of `img` with lines drawn on it
    """
    to_draw = img.copy()
    height = img.shape[0]
    for line in lines:
        if line:
            if random:
                color = (randrange(127, 255), randrange(127, 255), randrange(127, 255))
            if offset:
                line = Line(
                    line.x1, line.y1 + height / 2, line.x2, line.y2 + height / 2
                )
            cv2.line(to_draw, (line.x1, line.y1), (line.x2, line.y2), color, 3)
    return to_draw


def draw_lanes(
    img: npt.NDArray[any], lanes: list[tuple[Line, Line]], offset=False, random=False
) -> npt.NDArray[any]:
    """Draws lanes lines, each in a unique color, on img.

    ### Parameters
    - img (npt.NDArray[any]): the image to draw on
    - lanes (list[tuple[Line, Line]]): the list of lanes to draw
    - offset (bool, optional): whether or not to offset each line by height/2. Defaults to False.

    ### Returns
    - npt.NDArray[any]: the modified image
    """
    laned_img = img
    color = (0, 255, 0)
    for lane in lanes:
        if random:
            color = (randrange(255), randrange(255), randrange(255))
        laned_img = draw_lines(laned_img, lane, color=color, offset=offset)
    return laned_img


def group_data(labels: list[int], data: list[any]) -> dict[int, any]:
    """Group lines based on labels, which are expected to be the output of a DBSCAN fit.

    ### Parameters
    - labels (list[int]): the labels of the data, i.e. the clusters they should go to
    - data (list[any]): the data set

    ### Returns
    - dict[int, any]: the data set mapped to their labels
    """
    grouped_data = {}
    for index, element in enumerate(data):
        label = labels[index]
        if label == -1:
            continue
        if label not in grouped_data:
            grouped_data[label] = []
        grouped_data[label].append(element)

    return grouped_data


def dist(a: Union[float, int], b: Union[float, int]) -> Union[float, int]:
    """returns the distance between `a` and `b`

    ### Parameters
    - a (Union[float, int]): the first element
    - b (Union[float, int]): the second element

    ### Returns
    - Union[float, int]: the distance between the elements
    """
    return abs(a - b)


# ===============
# Image Filtering
# ===============
def to_gray(img: npt.NDArray[any]) -> npt.NDArray[any]:
    """the image converted to gray scale

    ### Parameters
    - img (npt.NDArray[any]): the image to convert

    ### Returns
    - npt.NDArray[any]: grayscale version of the image
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def split(img: npt.NDArray[any]) -> npt.NDArray[any]:
    """Splits or slices the image in half, leaving only the bottom half

    ### Parameters
    - img (npt.NDArray[any]): the image to split

    ### Returns
    - npt.NDArray[any]: the bottom half of the image
    """
    height = img.shape[0]
    return img[int(height / 2) : height]


def to_blurred(img: npt.NDArray[any], kernel_size: int = 19) -> npt.NDArray[any]:
    """Applies a Gaussian blur to the image.

    ### Parameters
    - img (npt.NDArray[any]): the image to blur
    - kernel_size (int, optional): the size of the kernel to convolve in blurring. Defaults to 19.

    ### Returns
    - npt.NDArray[any]: the blurred image
    """
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def to_bw(
    img: npt.NDArray[any], t: int = 90, white_value: int = 255
) -> npt.NDArray[any]:
    """Converts the image to black and white. Assumes that input is already in grayscale.

    ### Parameters
    - img (npt.NDArray[any]): the image to convert
    - t (int, optional): the threshold at which images are considered white. Defaults to 90.
    - white_value (int, optional): the value to set white pixels to. Defaults to 255.

    ### Returns
    - npt.NDArray[any]: the black and white image
    """
    _, bw_image = cv2.threshold(img, t, white_value, cv2.THRESH_BINARY)
    return bw_image


def find_edges(
    img: npt.NDArray[any], t1: int = 50, t2: int = 100, aperture: int = 3
) -> npt.NDArray[any]:
    """Detects the edges of the image, using the Canny algorithm.

    ### Parameters
    - img (npt.NDArray[any]): the image to detect edges on. Typically this is already in grayscale or black and white.
    - t1 (int, optional): the lower threshold for Canny algorithm. Defaults to 50.
    - t2 (int, optional): the upper threshold for Canny algorithm. Defaults to 100.
    - aperture (int, optional): the size of the edge detection aperture. Defaults to 3.

    ### Returns
    - npt.NDArray[any]: the image showing all detected edges
    """
    return cv2.Canny(img, threshold1=t1, threshold2=t2, apertureSize=aperture)


# ===================
# Edge/line Detection
# ===================
def find_lines(
    edges: npt.NDArray[any],
    rho=1,
    theta=np.pi / 180,
    threshold=100,
    min_line_length=100,
    max_line_gap=20,
) -> list[Line]:
    """Finds the line segments in the `edges` image. Assumes that `edges` has already been passed through a edge detection algorithm, such as canny.

    ### Parameters
    - edges (npt.NDArray[any]): the image to find lines in, e.g. the output of Canny
    - rho (int, optional): the resolution for Hough Lines. Defaults to 1.
    - theta (_type_, optional): the angle for Hough Lines. Defaults to np.pi/180.
    - threshold (int, optional): the number of votes which a Line must recieve to be counted. Defaults to 100.
    - min_line_length (int, optional): the lower threshold for Hough Lines. Defaults to 100.
    - max_line_gap (int, optional): the upper threshold for Hough Lines. Defaults to 20.

    ### Returns
    - list[Line]: a list of Line segments found in the image.
    """
    height = edges.shape[0]
    lines = cv2.HoughLinesP(
        edges,
        rho=rho,
        theta=theta,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    try:
        lines = [Line(line[0][0], line[0][1], line[0][2], line[0][3], image_height=height) for line in lines]
        return lines
    except TypeError:
        # could not iterate over lines, because none were detected
        return []

# ============
# Line Merging
# ============
def group_lines(
    lines: list[Line],
    height: int,
    slope_tolerance: float = 0.1,
    x_intercept_tolerance: int = 50,
):
    """Returns a dictionary containing lines that have been seperated into groups.

    ### Parameters
    - lines (list[Line]): the list of lines that need to be grouped
    - height (int): the height of the image. Required for calculating x-intercepts.
    - slope_tolerance (float, optional): the tolerance with which to group lines by slope. Defaults to 0.1.
    - x_intercept_tolerance (int, optional): the tolerance with which to group lines by x-intercept. Defaults to 50.

    ### Returns
    - dict[int: dict[int: list[Line]]] | None: the grouped lines, or None if len(lines) < 1
    """
    if len(lines) < 1:
        return None
    # Step 1. Group lines by slope
    slopes = [line.slope for line in lines]
    slopes = np.array(slopes).reshape(-1, 1)  # convert slopes to a 2d array

    dbscan = DBSCAN(eps=slope_tolerance, min_samples=1)
    labels = dbscan.fit_predict(slopes)  # labels is a list of clusters, basically
    grouped_lines = group_data(labels, lines)

    # Step 2. Seperate each slope-group into x-intercept groupings
    dbscan = DBSCAN(eps=x_intercept_tolerance, min_samples=1)
    for label, lines in grouped_lines.items():
        x_intercepts = []
        for line in lines:
            if line.x_intercept == NO_X_INTERCEPT:
                x_intercepts.append(
                    line.y_intercept
                )  # we want to group horizontal lines by y-intercept instead
            else:
                x_intercepts.append(line.x(height / 2))
        x_intercepts = np.array(x_intercepts).reshape(-1, 1)
        labels = dbscan.fit_predict(x_intercepts)
        grouped_lines[label] = group_data(labels, lines)

    return grouped_lines


def merge_lines(
    grouped_lines, height: int, width: int
) -> list[Line]:
    """Merges groups of lines into individual lines.

    ### Parameters
    - grouped_lines (dict[int : dict[int : list[Line]]]): the list of grouped lines, typically the output of `group_lines`
    - height (int): the height of the image, required for clipping the line segments
    - width (int): the width of the image, required for clipping the line segments

    ### Returns
    - list[Line]: the list of merged lines
    """
    merged_lines = []
    for _, slope_group in grouped_lines.items():
        for _, x_group in slope_group.items():
            # for each group, reset the x and y list
            x_list = []
            y_list = []
            # create a list of all the x and y coordinates in the group
            for line in x_group:
                x_list.extend([line.x1, line.x2])
                y_list.extend([line.y1, line.y2])
            m, b = np.polyfit(x_list, y_list, 1)  # line of best fit for the points

            # y = mx + b
            # (y - b) / m = x
            y_list = [height, 0]
            x_list = [
                (y_list[0] - b) / m,
                (y_list[1] - b) / m,
            ]  # [intercept with bottom of image, intercept with top]

            # Sometimes the intercepts will have negative coordinates. The following if block fixes that.
            for i in range(len(x_list)):
                if x_list[i] < 0:
                    x_list[i] = 0
                    y_list[i] = b
                elif x_list[i] > width:
                    x_list[i] = width
                    y_list[i] = m * width + b

            # Finally, create and append the line segments.
            merged_line = Line(x_list[0], y_list[0], x_list[1], y_list[1])
            merged_lines.append(merged_line)

    merged_lines.sort(key=lambda x: x.x_intercept)  # sort by x_intercept
    return merged_lines


# ==============
# Lane Detection
# ==============
def pixels_between(
    img: npt.NDArray[any], p1: tuple[int, int], p2: tuple[int, int]
) -> float:
    """Returns the average color of the pixels between point 1 and point 2.

    ### Parameters
    - img (npt.NDArray[any]): the image, assumed to be single channel (black and white)
    - p1 (tuple[int, int]): the first point
    - p2 (tuple[int, int]): the second point

    ### Returns
    - float: the average color value between the two points
    """
    # we will index by pixels, so make sure they are within the confines of the image
    # print(p1)
    # print(p2)
    x_list = np.clip([p1[0], p2[0]], 0, img.shape[1] - 1)
    y_list = np.clip([p1[1], p2[1]], 0, img.shape[0] - 1)
    x_list.sort()
    y_list.sort()
    line = Line(x_list[0], y_list[0], x_list[1], y_list[1])
    if line.length() > 5:
        rr, cc = line.pixels_between()
        average_value = np.average(img[cc, rr])
        return average_value
    return 255 # if the gap 

def detect_lanes(
    img: npt.NDArray[any],
    lines: list[Line],
    x_tolerance: int = 300,
    y_tolerance: int = 300,
    darkness_threshold: float = 10.0,
) -> list[tuple[Line, Line]]:
    """Detects the lanes, or pairs of lines, within the given set of lines. Requires the image in order to check for pixels between potential pairs.

    ### Parameters
    - img (npt.NDArray[any]): the image which lines are being detected on
    - lines (list[Line]): the list of lines to pair up
    - x_tolerance (int): the maximum difference in x-intercepts in which two lines could be a pair. Defaults to 300 pixels.
    - y_tolerance (int): the maximum difference in y-intercepts in which two lines could be a pair. Defaults to 300 pixels.
    - darkness_threshold (float): the maximum average value of pixels between the two lines for them to be considered a lane. Defaults to 10.0.

    ### Returns
    - list[tuple[Line, Line]]: the list of paired lines, or lanes
    """
    lanes = []  # the return list
    lines.sort(key=lambda x: x.x_intercept)

    for line1, line2 in combinations(lines, 2):
        # for each unique pair of lines
        if line1.is_paired() or line2.is_paired():
            # If either line has a pair, we can't pair it again
            continue
        if (
            dist(line1.x_intercept, line2.x_intercept) < x_tolerance
            or dist(line1.y_intercept, line2.y_intercept) < y_tolerance
        ):
            # (line1, line2) is a fair candidate for a lane.
            # Next, check the pixels between the lines to see if it is dark.
            if (
                pixels_between(img, (line1.x1, line1.y1), (line2.x1, line2.y1))
                < darkness_threshold
            ):
                # The pixels between the lines are dark, so it is a lane.
                line1.paired = True
                line2.paired = True
                lanes.append((line1, line2))

    return lanes
