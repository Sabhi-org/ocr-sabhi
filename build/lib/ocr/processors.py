import cv2
import math
from scipy import ndimage
import numpy as np
import pytesseract
import re


class RotationCorrector:
    def __init__(self, output_process=False):
        self.output_process = output_process

    def __call__(self, image):
        img_before = image.copy()

        img_edges = cv2.Canny(img_before, 50, 150, apertureSize=3)

        lines = cv2.HoughLinesP(
            img_edges, 1, math.pi / 90.0, 100, minLineLength=100, maxLineGap=5
        )
        print("Number of lines found:", len(lines))

        def get_angle(line):
            x1, y1, x2, y2 = line[0]
            return math.degrees(math.atan2(y2 - y1, x2 - x1))

        median_angle = np.median(
            np.array([get_angle(line) for line in lines])
        )
        img_rotated = ndimage.rotate(
            img_before, median_angle, cval=255, reshape=False)

        print("Angle is {}".format(median_angle))

        if self.output_process:
            cv2.imwrite("output/10. tab_extract rotated.jpg", img_rotated)

        return img_rotated


class Resizer:
    """Resizes image.

    Params
    ------
    image   is the image to be resized
    height  is the height the resized image should have.
            Width is changed by similar ratio.

    Returns
    -------
    Resized image
    """

    def __init__(self, height=1280, output_process=False):
        self._height = height
        self.output_process = output_process

    def __call__(self, image):
        #        if image.shape[0] <= self._height:
        #            return image
        ratio = round(self._height / image.shape[0], 3)
        width = int(image.shape[1] * ratio)
        dim = (width, self._height)
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        if self.output_process:
            cv2.imwrite("output/resized.jpg", resized)
        return resized


class OtsuThresholder:
    """Thresholds image by using the otsu method

    Params
    ------
    image   is the image to be Thresholded

    Returns
    -------
    Thresholded image
    """

    def __init__(self, thresh1=0, thresh2=255, output_process=False):
        self.output_process = output_process
        self.thresh1 = thresh1
        self.thresh2 = thresh2

    def __call__(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(
            image,
            self.thresh1,
            self.thresh2,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        if self.output_process:
            cv2.imwrite("output/thresholded.jpg", thresholded)
        return thresholded


class AdaptiveThresholder:
    """Thresholds image by using the Gaussian  method

    Params
    ------
    image   is the image to be Thresholded

    Returns
    -------
    Thresholded image
    """

    def __init__(self, thresh1=255, block_size=11, constant=2, output_process=False):
        self.output_process = output_process
        self.thresh1 = thresh1
        self.block_size = block_size
        self.constant = constant

    def __call__(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresholded = cv2.adaptiveThreshold(
            image,
            self.thresh1,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.block_size,
            self.constant
        )

        if self.output_process:
            cv2.imwrite("output/thresholded.jpg", thresholded)
        return thresholded


class FastDenoiser:
    """Denoises image by using the fastNlMeansDenoising method

    Params
    ------
    image       is the image to be Thresholded
    strength    the amount of denoising to apply

    Returns
    -------
    Denoised image
    """

    def __init__(self, strength=7, output_process=False):
        self._strength = strength
        self.output_process = output_process

    def __call__(self, image):
        temp = cv2.fastNlMeansDenoising(image, h=self._strength)
        if self.output_process:
            cv2.imwrite("output/denoised.jpg", temp)
        return temp


class Closer:
    def __init__(self, kernel_size=3, iterations=10, output_process=False):
        self._kernel_size = kernel_size
        self._iterations = iterations
        self.output_process = output_process

    def __call__(self, image):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self._kernel_size, self._kernel_size)
        )
        closed = cv2.morphologyEx(
            image, cv2.MORPH_CLOSE, kernel, iterations=self._iterations
        )

        if self.output_process:
            cv2.imwrite("output/closed.jpg", closed)
        return closed


class Opener:
    def __init__(self, kernel_size=3, iterations=25, output_process=False):
        self._kernel_size = kernel_size
        self._iterations = iterations
        self.output_process = output_process

    def __call__(self, image):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self._kernel_size, self._kernel_size)
        )
        opened = cv2.morphologyEx(
            image, cv2.MORPH_OPEN, kernel, iterations=self._iterations
        )

        if self.output_process:
            cv2.imwrite("output/opened.jpg", opened)
        return opened


class EdgeDetector:
    def __init__(self, output_process=False):
        self.output_process = output_process

    def __call__(self, image, thresh1=50, thresh2=150, apertureSize=3):
        edges = cv2.Canny(image, thresh1, thresh2, apertureSize=apertureSize)
        if self.output_process:
            cv2.imwrite("output/edges.jpg", edges)
        return edges


class SharpenAndDilate:
    def __init__(
        self,
        kernel=(7, 7),
        normalized=False,
        output_process=False
    ):
        self._kernel = kernel
        self._normalized = normalized
        self.output_process = output_process

    def __call__(
        self,
        image,
        debug=False
    ):

        self._image = image
        self._debug = debug

        return self._sharpened_and_dialated()

    def _sharpened_and_dialated(self):
        rgb_planes = cv2.split(self._image)
        result = []
        for plane in rgb_planes:
            image_result = cv2.dilate(plane,
                                      np.ones(self._kernel, np.uint8))
            bg_img = cv2.medianBlur(image_result, 21)

            if self._debug:
                cv2.imshow("bg_img", bg_img)
                cv2.waitKey(0)
            image_result = 255 - cv2.absdiff(plane, bg_img)

            if self._normalized:
                norm_img = cv2.normalize(
                    image_result,
                    None,
                    alpha=0,
                    beta=255,
                    norm_type=cv2.NORM_MINMAX,
                    dtype=cv2.CV_8UC1)

                result.append(norm_img)
            else:
                result.append(image_result)
        image = cv2.merge(result)
        return image


class RotationDetector:
    def __init__(self):
        self._init = None

    def __call__(
        self,
        image,
        debug=False
    ):

        self._image = image
        # Save image to avoid leptonica error
        temp_file_name = "rotation_detector_image.png"
        cv2.imwrite(temp_file_name, self._image)
        self._image = cv2.imread(temp_file_name)
        # Step 1: Get Document HoughLines

        self._angle_hough = self._get_angle_hough()

        self._angle_tesseract = self._get_angle_tesseract()
        print(self._angle_tesseract)
        self._angle = np.round(
            np.degrees(self._angle_hough)
        )

        return self._angle - 90.0

    def _get_angle_tesseract(
        self,
        config="--psm 12 --oem 3",
    ):
        image_osd = pytesseract.image_to_osd(self._image, config=config)
        rotation_angle = re.search('(?<=Rotate: )\d+', image_osd).group(0)
        return rotation_angle

    def _get_angle_hough(self):
        self._hough_lines = self._get_hough_lines()
        return np.median(
            np.array([self._get_line(line) for line in self._hough_lines])
        )

    def _get_line(self, line):
        _, theta = line[0]

        return theta

    def _get_hough_lines(
        self,
        rho_acc=1,
        theta_acc=90,
        thresh=100,
        preprocessor=[
            Closer(),
            EdgeDetector(),
        ]
    ):
        """
        Extract straight lines from image using Hough Transform.

        Returns
        ----------
        array containing rho and theta of lines (Hess Norm formulation)
        """

        for processor in preprocessor:
            image = processor(self._image.copy())

        lines = cv2.HoughLines(
            image, rho_acc, np.pi / theta_acc, thresh
        )

        return lines
