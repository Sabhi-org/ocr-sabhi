import cv2
import numpy as np
from ocr.processors import Closer, EdgeDetector
from sklearn.cluster import KMeans
from collections import defaultdict
import itertools


class HoughLineCornerDetector:
    def __init__(
        self,
        rho_acc=2,
        theta_acc=360,
        thresh=100,
        output_process=True
    ):

        self.rho_acc = rho_acc
        self.theta_acc = theta_acc
        self.thresh = thresh
        self.output_process = output_process
        self._preprocessor = [
            Closer(output_process=output_process),
            EdgeDetector(output_process=output_process),
        ]

    def __call__(self, image):
        # Step 1: Process for edge detection
        self._image = image
        self._n = max(self._image.shape[0], self._image.shape[1])

        for processor in self._preprocessor:
            self._image = processor(self._image)

        # Step 2: Get hough lines
        self._lines = self._get_hough_lines()
        self._segmented_lines = self._segment_by_angle_kmeans()

        # Step 3: Get intersection points
        self._intersections = self._get_intersections()

        # Step 4: Get Quadrilaterals
        return self._find_quadrilaterals()

    def _get_hough_lines(self):
        """
        Extract straight lines from image using Hough Transform.

        Returns
        ----------
        array containing rho and theta of lines (Hess Norm formulation)
        """
        lines = cv2.HoughLines(
            self._image, self.rho_acc, np.pi / self.theta_acc, self.thresh
        )
        if self.output_process:
            self._draw_hough_lines(lines)

        return lines

    def _draw_hough_lines(self, lines):
        hough_line_output = self._get_color_image()
        n = self._n

        for line in lines:
            rho, theta = line[0]
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a * rho, b * rho
            x1 = int(x0 + n * (-b))
            y1 = int(y0 + n * (a))
            x2 = int(x0 - n * (-b))
            y2 = int(y0 - n * (a))

            cv2.line(hough_line_output, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imwrite("output/hough_line.jpg", hough_line_output)

    def _segment_by_angle_kmeans(self, k=2, **kwargs):
        """Groups lines based on angle with k-means.

        Uses k-means on the coordinates of the angle on the unit circle
        to segment `k` angles inside `lines`.
        """

        lines = self._lines

        # Define criteria = (type, max_iter, epsilon)
        default_criteria_type = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
        )

        criteria = kwargs.get("criteria", (default_criteria_type, 10, 1.0))
        flags = kwargs.get("flags", cv2.KMEANS_RANDOM_CENTERS)
        attempts = kwargs.get("attempts", 10)

        # returns angles in [0, pi] in radians
        angles = np.array([line[0][1] for line in lines])
        # multiply the angles by two and find coordinates of that angle
        pts = np.array(
            [[np.cos(2 * angle), np.sin(2 * angle)] for angle in angles],
            dtype=np.float32,
        )

        # run kmeans on the coords
        labels, centers = cv2.kmeans(
            pts, k, None, criteria, attempts, flags)[1:]
        labels = labels.reshape(-1)  # transpose to row vec

        # segment lines based on their kmeans label
        segmented = defaultdict(list)
        for i, line in zip(range(len(lines)), lines):
            segmented[labels[i]].append(line)
        segmented = list(segmented.values())
        return segmented

    def _get_intersections(self, k=2):
        """Finds the intersections between groups of lines."""
        segmented_lines = self._segmented_lines
        intersections = []

        group_lines = list(itertools.product(
            segmented_lines[0], segmented_lines[1]))

        for line_i, line_j in group_lines:
            int_point = self._intersection(line_i, line_j)
            intersections.append(int_point)

        if self.output_process:
            self._draw_intersections(intersections)

        return intersections

    def _find_quadrilaterals(self):
        X = np.array([[point[0][0], point[0][1]]
                      for point in self._intersections])

        kmeans = KMeans(
            n_clusters=4,
            init="k-means++",
            max_iter=100,
            n_init=10,
            random_state=0
        ).fit(X)

        if self.output_process:
            self._draw_quadrilaterals(self._lines, kmeans)

        return [[center.tolist()] for center in kmeans.cluster_centers_]

    def _draw_quadrilaterals(self, lines, kmeans):
        grouped_output = self._get_color_image()
        n = self._n

        for idx, line in enumerate(lines):
            rho, theta = line[0]
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a * rho, b * rho
            x1 = int(x0 + n * (-b))
            y1 = int(y0 + n * (a))
            x2 = int(x0 - n * (-b))
            y2 = int(y0 - n * (a))

            cv2.line(grouped_output, (x1, y1), (x2, y2), (0, 0, 255), 2)

        for point in kmeans.cluster_centers_:
            x, y = point

            cv2.circle(grouped_output, (int(x), int(y)), 5, (255, 255, 255), 5)

        cv2.imwrite("output/grouped_quads.jpg", grouped_output)

    def _get_angle_between_lines(self, line_1, line_2):
        rho1, theta1 = line_1[0]
        rho2, theta2 = line_2[0]
        # x * cos(theta) + y * sin(theta) = rho
        # y * sin(theta) = x * (- cos(theta)) + rho
        # y = x * (-cos(theta) / sin(theta)) + rho
        m1 = -(np.cos(theta1) / np.sin(theta1))
        m2 = -(np.cos(theta2) / np.sin(theta2))
        return abs(np.arctan(abs(m2 - m1) / (1 + m2 * m1))) * (180 / np.pi)

    def _intersection(self, line1, line2):
        """Finds the intersection of two lines given in Hesse normal form.

        Returns closest integer pixel locations.
        See https://stackoverflow.com/a/383527/5087436
        """
        rho1, theta1 = line1[0]
        rho2, theta2 = line2[0]

        A = np.array(
            [[np.cos(theta1), np.sin(theta1)], [
                np.cos(theta2), np.sin(theta2)]]
        )

        b = np.array([[rho1], [rho2]])
        x0, y0 = np.linalg.solve(A, b)
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        return [[x0, y0]]

    def _draw_intersections(self, intersections):
        intersection_point_output = self._get_color_image()
        n = self._n

        for line in self._lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + n * (-b))
            y1 = int(y0 + n * (a))
            x2 = int(x0 - n * (-b))
            y2 = int(y0 - n * (a))

            cv2.line(intersection_point_output,
                     (x1, y1), (x2, y2), (0, 0, 255), 2)

        for point in intersections:
            x, y = point[0]

            cv2.circle(intersection_point_output,
                       (x, y), 5, (255, 255, 127), 5)

        cv2.imwrite("output/intersection_point_output.jpg",
                    intersection_point_output)

    def _get_color_image(self):
        return cv2.cvtColor(self._image.copy(), cv2.COLOR_GRAY2RGB)
