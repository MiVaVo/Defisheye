import cv2
import numpy as np


def return_n_of_section(img, threshold=600, angle_th=10):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = 255 - gray
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(image=edges,
                            rho=1,
                            theta=np.pi / 180,
                            threshold=threshold,
                            minLineLength=minLineLength,
                            maxLineGap=maxLineGap)
    lines = lines[[decide_on_line(i, angle_th) for i in lines]]
    print(lines.shape[0], "N OF PART LINES")
    img_plot = img.copy()

    for line in lines:

        for x1, y1, x2, y2 in line:
            cv2.line(img_plot, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return lines, img_plot


def return_n_of_lines(img, threshold):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(image=edges,
                           rho=1,
                           theta=np.pi / 180,
                           threshold=threshold)
    print(lines.shape[0], "N OF STRAIGH LINES")
    img_plot = img.copy()
    for line in lines:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(img_plot, (x1, y1), (x2, y2), (255, 255, 255), 2)
    return lines, img_plot


def calc_angle(line):
    x1, y1, x2, y2 = line[0]
    xd = x2 - x1
    yd = y2 - y1

    if xd == 0:
        return 90
    tg_alpha = yd / xd
    return abs(np.degrees(np.arctan(tg_alpha)))


def decide_on_line(line, degree_th):
    degree = calc_angle(line)
    if degree > degree_th:
        return False
    else:
        return True
