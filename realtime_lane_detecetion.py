# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import os
import numpy as np
import cv2
from sklearn import linear_model


def find_lines(lines):
    slope_min = 0.3
    slope_max = 2
    x_right = []
    y_right = []

    x_left = []
    y_left = []

    for line in lines:
        for x1,y1,x2,y2 in line:
            m = (y2-y1)/(x2-x1)
            if m < -slope_min and m > -slope_max: #left
                if x1 > imshape[1]*0.5 or x2 > imshape[1]*0.5:
                    continue
                else:
                    x_left.append([x1])
                    x_left.append([x2])
                    y_left.append(y1)
                    y_left.append(y2)
            elif m > slope_min and m < slope_max: #right
                if x1 < imshape[1]*0.5 or x2 < imshape[1]*0.5:
                    continue
                else:
                    x_right.append([x1])
                    x_right.append([x2])
                    y_right.append(y1)
                    y_right.append(y2)

    x_right = np.array(x_right)
    y_right = np.array(y_right)
    if not x_right.any() or not y_right.any():
        x1, y1, x2, y2 = int(imshape[1]), int(imshape[0]), int(imshape[1]-0.000001), int(imshape[0]-0.000001)
    else:
        model_ransac_right = linear_model.RANSACRegressor(linear_model.LinearRegression())
        model_ransac_right.fit(x_right, y_right)
        line_x_right = np.arange(imshape[1]*0.55, imshape[1]*0.9)
        line_y_right = model_ransac_right.predict(line_x_right[:, np.newaxis])
        x1, y1, x2, y2 = [int(line_x_right[0]), int(line_y_right[0]), int(line_x_right[-1]), int(line_y_right[-1])]

    x_left = np.array(x_left)
    y_left = np.array(y_left)
    if not x_left.any() or not y_left.any():
        x1N, y1N, x2N, y2N = 0, imshape[0], 1, imshape[0]-1
    else:
        model_ransac_left = linear_model.RANSACRegressor(linear_model.LinearRegression())
        model_ransac_left.fit(x_left, y_left)
        line_x_left = np.arange(imshape[1]*0.1, imshape[1]*0.45)
        line_y_left = model_ransac_left.predict(line_x_left[:, np.newaxis])
        x1N, y1N, x2N, y2N = [int(line_x_left[0]), int(line_y_left[0]), int(line_x_left[-1]), int(line_y_left[-1])]

    left = [x1N, y1N, x2N, y2N]
    right = [x1, y1, x2, y2]

    return left, right

capture = cv2.VideoCapture(0)
capture.set(3, 640)
capture.set(4, 480)

if capture.isOpened:
    file_path = './result.mp4'
    fps = 25.4
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    size = (int(width),int(height))

    output = cv2.VideoWriter(file_path, fourcc, fps, size)

    while True:
        ret, img = capture.read()

        if not ret:
            print('Failure: loading video')
            break
        if cv2.waitKey(1) & 0xFF == 27:
            break

        #----------------GET IMAGE-----------------
        imshape = img.shape

        # -------------GREYSCALE IMAGE---------------
        # Grayscale one color channel
        grayIm = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #------------GAUSSIAN SMOOTHING-----------------
        kernel_size = 9
        smoothedIm = cv2.GaussianBlur(grayIm, (kernel_size, kernel_size), 0)

        #-------------EDGE DETECTION---------------------
        minVal = 60
        maxVal = 150
        edgesIm = cv2.Canny(smoothedIm, minVal, maxVal)

        #-------------------------CREATE MASK (Trizoidal) ----------------------------
        vertices = np.array([[(0,imshape[0]),(465, 320), (475, 320), (imshape[1],imshape[0])]], dtype=np.int32)
        mask = np.zeros_like(edgesIm)
        color = 255
        cv2.fillPoly(mask, vertices, color)

        #----------------------APPLY MASK TO IMAGE-------------------------------
        maskedIm = cv2.bitwise_and(edgesIm, mask)

        #-----------------------HOUGH LINES------------------------------------
        rho = 2 # distance resolution in pixels of the Hough grid
        theta = np.pi/180 # angular resolution in radians of the Hough grid
        threshold = 45     # minimum number of votes (intersections in Hough grid cell)
        min_line_len = 20  #40 # minimum number of pixels making up a line
        max_line_gap = 100    # maximum gap in pixels between connectable line segments
        lines = cv2.HoughLinesP(maskedIm, rho, theta, threshold, np.array([]),
                                minLineLength=min_line_len, maxLineGap=max_line_gap)

        """ To do: plot hough lines """
        # houghlines = np.zeros_like(img)
        # if lines is not None:
        #     print(f'lines len/shape {len(lines)} {lines.shape}')
        #     for i in range(len(lines)):
        #         for rho, theta in lines[i]:
        #             a = np.cos(theta)
        #             b = np.sin(theta)
        #             x0 = a*rho
        #             y0 = b*rho
        #             x1 = int(x0 + 1000*(-b))
        #             y1 = int(y0+1000*(a))
        #             x2 = int(x0 - 1000*(-b))
        #             y2 = int(y0 -1000*(a))

        #             cv2.line(houghlines,(x1,y1),(x2,y2),(0,0,255),2)

    # res = np.vstack((img_original,img))
    # cv2.imshow('img',res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

        #------------------------Ransac Regressor--------------------
        if lines is not None:
            left, right = find_lines(lines)
            x1,y1,x2,y2 = right
            x1N, y1N, x2N, y2N = left
        else:
            x1, y1, x2, y2 = int(imshape[1]), int(imshape[0]), int(imshape[1]-0.000001), int(imshape[0]-0.000001)
            x1N, y1N, x2N, y2N = 0, imshape[0], 1, imshape[0]-1


        #-----------------------Blend Image------------------------------
        laneFill = img.copy()
        vertices = np.array([[(x1,y1),(x2,y2), (x1N,y1N), (x2N,y2N)]], dtype=np.int32)
        color = [241,255,1]
        cv2.fillPoly(laneFill, vertices, color)
        opacity = .25
        blendedImg = cv2.addWeighted(laneFill,opacity,img,1-opacity,0,img)
        cv2.line(blendedImg,(x1,y1),(x2,y2),(0,255,0),4) # plot line on color image
        cv2.line(blendedImg,(x1N,y1N),(x2N,y2N),(0,255,0),4) # plot line on color image

        # res = np.vstack(blendedImg, houghlines)
        cv2.imshow("Video", blendedImg)
        output.write(blendedImg)
    output.release()

else:
    print("CAN'T OPEN CAMERA!")

capture.release()
cv2.destroyAllWindows()
