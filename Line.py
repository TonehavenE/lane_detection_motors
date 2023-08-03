from numpy import power, sqrt
from skimage.draw import line

# CONSTANTS
VERTICAL_SLOPE = power(10, 10)
HORIZONTAL_SLOPE = 0.00000000000001
NO_X_INTERCEPT = power(10, 10)
IMAGE_HEIGHT = int(1080/2)

class Line:
    """Represents a line segment given two points. Has methods for calculating slope, x-intercept, and the x or y coordinate given the other.
    """
    def __init__(self, x1: int, y1: int, x2: int, y2: int, image_height: int = IMAGE_HEIGHT):
        """Constructs a Line object given two points, (x1, y1) and (x2, y2)

        ### Parameters
        - x1 (int): the x coordinate of the first point
        - y1 (int): the y coordinate of the first point
        - x2 (int): the x coordinate of the second point
        - y2 (int): the y coordinate of the second point
        ### Notes
        x-intercept in this class refers to the intercept with the bottom of the image. The image height is declared as a class constant, so make sure you change it if the image size changes!
        """
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)
        self.image_height = image_height
        self.slope = self.calculate_slope()
        self.x_intercept = self.calculate_x_intercept()
        # self.x_of = np.vectorize(self.x)
        # self.y_of = np.vectorize(self.y)
        self.y_intercept = self.y(0)
        self.paired = False

    def calculate_slope(self) -> float:
        """Calculates the slope of the line.

        ### Returns
        - float: the slope
        """
        if self.x1 == self.x2:
            return VERTICAL_SLOPE
        elif self.y1 == self.y2:
            return HORIZONTAL_SLOPE
        else:
            return (self.y2 - self.y1) / (self.x2 - self.x1)

    def calculate_x_intercept(self) -> int:
        """Calculates the intercept of the line segment with the bottom of the image. This is referred to as the x-intercept in all of the code.

        Returns:
        - int: the point (x, 0) of the intercept
        """
        if self.y1 == self.y2:
            return NO_X_INTERCEPT
        else:
            # return ((self.slope * self.x1) - self.y1) / self.slope
            return round(((self.image_height - self.y1) / self.slope) + self.x1, 0)

    def get_points(self) -> list[int, int, int, int]:
        """Returns the points of the line.

        ### Returns
        - list[int, int, int, int]: [x1, y1, x2, y2]
        """
        return [self.x1, self.y1, self.x2, self.y2]
    
    def pixels_between(self) -> (list[int], list[int]):
        """The discrete points between the two points of the line

        ### Returns
        - tuple[list[int], list[int]]: the x and y coordinates of the points between (x1, y1) and (x2, y2)
        """
        return line(self.x1, self.y1, self.x2, self.y2)
    
    def y(self, x: float) -> float:
        """Returns the y-coordinate of a point on the line, given x.

        ### Parameters
        - x (float): the x-coordinate

        ### Returns
        - float: the y-coordinate

        ### Notes
        y - y1 = m(x - x1)
        y = m(x - x1) + y1 <- this is the formula used
        """
        return self.slope * (x - self.x1) + self.y1
    
    def x(self, y: float) -> float:
        """Returns the x-coordinate of a point on the line, given y.

        ### Parameters
        - y (float): the y-coordinate

        ### Returns
        - float: the x-coordinate

        ### Notes
        y - y1 = mx - mx1
        mx = y - y1 + mx1
        x = ((y - y1) / m) + x1
        """
        return ((y - self.y1) / self.slope) + self.x1

    def is_paired(self) -> bool:
        """Returns whether the line is already paired with another line.

        ### Returns
        - bool: if the line is paired
        """
        return self.paired
    
    def length(self) -> float:
        """The length of the line.

        ### Returns
        - float: the length of the line
        """
        return sqrt(power(self.x2 - self.x1, 2) + power(self.y2 - self.y1, 2))

    def __str__(self) -> str:
        """The string representation of the line.

        ### Returns
        - str: "p1: (x1, y1), p2: (x2, y2), slope: m, x-intercept: x, y-intercept: b"
        """
        return f"p1: ({self.x1}, {self.y1}), p2: ({self.x2}, {self.y2}), slope: {self.slope:.2f}, x-intercept: {self.x_intercept}, y-intercept: {self.y_intercept}"