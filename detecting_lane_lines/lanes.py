import cv2
import numpy as np
# import matplotlib.pyplot as plt


def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    print(image.shape)
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    # print(left_fit_average, 'left')
    # print(right_fit_average, 'right')
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])


def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


image = cv2.imread('./test_image.jpg')
lane_image = np.copy(image)
canny_image = canny(lane_image)
cropped_image = region_of_interest(canny_image)
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100,
                        np.array([]), minLineLength=40, maxLineGap=5)
averaged_lines = average_slope_intercept(lane_image, lines)
line_image = display_lines(lane_image, averaged_lines)
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
# cv2.imshow('result', line_image)
cv2.imshow('result', combo_image)
cv2.waitKey(0)

# Edge detection
#  Identifying sharp changes in intensity in adjacent pixels

# An image can be expressed as a matrix of pixel
# A pixel is a dot in a picture. It contains
# the light intensity at some location in the image
# It is denoted by numbers in the 8-bit space where
# the lowest is black and the highest is white

# Gradient
# Measure of change in brightness over adjacent pixels
# Strong gradient means sharp change. In every gradient,
# there is a corresponding brightness

# Grayscale
# Unlike a colored picture with 3 channels each with 256
# possible intensities (each pixel a combination of 3 values),
# Any pixel in a grayscale image has only 1 channel
# with 56 possible intensities. Processing a single channel is
# cheaper and faster thatn processing 3 channels

# Noise Reduction
# While it is important to catch as many edges as possible
# it is quite imperative that we reduce noise. Image noise can
# create false edges. We can filter out noise by smoothening the image.
# We do that with the Gaussian filter
