from multiprocessing import Process

from lane_detection import *
from lane_following import *


def render_frame(frame):
    """Applies a sequence of image filtering and processing to find the center lane of a frame. Outputs the frame with the center lane drawn and a text overlay suggesting which direction to move/turn.
    
    ### Parameters
        frame: the frame to process/render

    ### Returns
        image: the post-processed image.
    """
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
        grouped_lines = group_lines(lines, height, slope_tolerance=0.1, x_intercept_tolerance=50) # group lines
        merged_lines = merge_lines(grouped_lines, height, width) # merge groups of lines

        # Lane Detection
        lanes = detect_lanes(bw, merged_lines, 500, 200, 10)

        # Lane picking
        center_lines = merge_lane_lines(lanes, height) # find the center of each lane
        center_line = pick_center_line(center_lines, width) # find the closest lane
        (longitudinal, lateral, turn) = error_from_line(center_line, width) # textual suggestion of how to move
        # print(f"{longitudinal = }, {lateral = }, {turn = }")
        turn = np.rad2deg(turn)
        if longitudinal == 100:
            text = f"Move forward: {longitudinal:.2f} | Turn: {turn:.2f}"
        elif lateral != 0:
            text = f"Move lateral: {lateral:.2f}% | Turn: {turn:.2f}"
        else:
            text = f"Don't move"

        # Drawing
        # frame = draw_lanes(frame, lanes, offset=True)
        frame = draw_lines(frame, [center_line], (0, 0, 255), offset=True)
        frame = cv2.putText(frame, text, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
    return frame

if __name__ == "__main__":
    cap = cv2.VideoCapture('AUV_Vid.mkv')
    ret, frame1 = cap.read()
    height, width, layers = frame1.shape
    size = (width, height)
    out = cv2.VideoWriter("rendered_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, size)

    count = 0 # the number of frames since the last    
    while ret:
        ret, frame = cap.read()
        if not ret:
            break

        print(f"now on frame {count}...")
        frame = render_frame(frame)
            
        out.write(frame)

        count += 1

    cap.release()
    out.release()
    print("Finished rendering the video.")