#importing some useful packages
import numpy as np
import cv2
import math
import os
from sklearn import linear_model, datasets


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    #return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[0,0,255], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)



def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)


    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)

    return line_img, lines


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

# Here is my code !!!

def increaseYellow(img):

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # Now we have HSV values
    channels = cv2.split(hsv_img )

    # define range of blue color in HSV
    lower_yellow = np.array([20, 140, 50])
    upper_yellow = np.array([40, 255, 255])

    yellows = cv2.inRange(hsv_img, lower_yellow, upper_yellow)


    out = np.copy(channels[1])
    cv2.add(channels[1], -200, out, mask = yellows )
    channels[1] = out

    out = np.copy(channels[2])
    cv2.add(channels[2], 90, out, mask = yellows )
    channels[2] = out

    new_hsv = cv2.merge(channels)

    new_rgb = cv2.cvtColor(new_hsv, cv2.COLOR_HSV2BGR)

    return new_rgb, yellows


# Checks if line is more than 45º. Criterium may be changed depending the camera focal length. This one is easy

def lineOk(l):

    points = l[0]

    x0 = points[0]
    y0 = points[1]
    x1 = points[2]
    y1 = points[3]

    if abs(y1-y0) > abs(x1 - x0)*0.4:
        return 1

    else:
        return 0


def houghLineOk(l):
    points = l

    x0 = points[0]
    y0 = points[1]
    x1 = points[2]
    y1 = points[3]

    #return y1-y0 != 0 and x0-x1 != 0

    # compute length of segment
    l = math.sqrt((x1-x0)*(x1-x0) + (y1-y0)*(y1-y0))


    return abs(y1-y0) > abs(x1 - x0)*0.3 and l > 50

# Builds an image from the polygons

def polyImage(polygons, sh):

    img = np.zeros((sh[0], sh[1], 3), dtype=np.uint8)

    for poly in polygons:
        cv2.fillPoly(img, poly, color=[0, 0, 255])

    return img


# Extends a line to the bottom or border of the image and to the horizon in top direction

def extend(l, shape, horizon):

    # if line is quite horizontal kill it

    points = l

    #print(points, shape, horizon)

    if points[1] < points[3]:
        x0 = points[0]
        y0 = points[1]
        x1 = points[2]
        y1 = points[3]
    else:
        x1 = points[0]
        y1 = points[1]
        x0 = points[2]
        y0 = points[3]

    if x0 == x1: # Vertical line - easy
        xe = horizon
        ye = shape[0]
        return np.array([[x0, y0, xe, ye]],  dtype=np.int32)

    elif y0 == y1: # Horizontal line. Just output it
        return l


    # Now x1 y1 have the bottom end of the line - Update extending

    ye1 = shape[0]
    xe1 = x0 + ((x1 - x0) / (y1-y0) * (ye1 - y0))

    # Now try to extend line to horizon

    ye = horizon
    xe = x0 + ((x1 - x0)/(y1-y0)*(horizon - y0))

    # check lateral borders. JUst to be OK

    if xe1 < 0:
        xe1 = 0
        ye1 = y0 + ((y1-y0)/(x1-x0)) * (xe1 - x0)

    if xe1 >= shape[1]:
        xe1 = shape[1]-1
        ye1 = y0 + ((y1 - y0) / (x1 - x0)) * (xe1 - x0)

    if xe < 0:
        xe = 0
        ye = y0 + ((y1 - y0) / (x1 - x0)) * (xe - x0)

    if xe >= shape[1]:
        xe = shape[1] - 1
        ye = y0 + ((y1 - y0) / (x1 - x0)) * (xe - x0)

    return np.array([[xe, ye, xe1, ye1]], dtype=np.int32)


def canonicalLine(line):
    points = line[0]

    x0 = points[0]
    y0 = points[1]
    x1 = points[2]
    y1 = points[3]

    if x1 == x0 :
        return [0.0, 0.0]

    if y1 == y0:
        return [0.0, 0.0]

    m = float(y1 - y0) / float(x1 - x0)
    b = float(y0) - m * float(x0)

    return[b, m]


def vanishingPoint(lines):

    hough_points = []

    print('Lines ', lines.shape[0])
    for l in lines:     # Clear bad lines
        if lineOk(l):
            hough_points.append(canonicalLine(l))

    hough_points = np.array(hough_points)
    # Now we make the regression

    print("HP",hough_points.shape[0])

    if hough_points.shape[0] < 2:
        return 100, 100

    X = np.array(hough_points[:,1])   # m
    Y = np.array(hough_points[:,0])   # b

    X = X.reshape(-1, 1)

    model = linear_model.RANSACRegressor(linear_model.LinearRegression())
    model.fit(X, Y)

    inlier_mask = model.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)


    m1 = 3.0
    m2 = 175.0

    XM = np.array([m1, m2]).reshape(-1, 1)
    YM = model.predict(XM)

    b1 = YM[0]
    b2 = YM[1]

    x = np.int64(round((b2 - b1)/(m1 - m2), 0))
    y = np.int64(round(b1 + m1 * x, 0))


    return x, y

def drawPoint(img, x, y, color):


    lines = np.array([[[x-10, y, x+10, y]], [[x, y-10, x, y+10]]])

    draw_lines(img, lines, color=color, thickness=2)

# Basic image processing

def process(img, hor, hor_width):

    yimg, ylsmask = increaseYellow(img)

    # Convert to gray
    gray = grayscale(yimg)


    # Some values to adjust

    kernel_size = 7
    high_threshold = 150
    low_threshold = 50

    blurred = gaussian_blur(gray, kernel_size)
    edges = canny(blurred, low_threshold, high_threshold)

    # My contrinution is to use a simple horizon and the width which we cut it (explain)

    imshape = img.shape

    vertices = np.array([[(0, imshape[0]), (imshape[1] / 2 - hor_width, hor), (imshape[1] / 2 + hor_width, hor),
                          (imshape[1], imshape[0])]], dtype=np.int32)

    roi = region_of_interest(edges, vertices)

    # Now apply Hough Transform to get line segments

    rho = 1
    theta = np.pi/180
    threshold = 10
    min_line_length = 70
    max_line_gap = 70

    draw, lines = hough_lines(roi, rho, theta, threshold, min_line_length, max_line_gap)

    mask   = np.apply_along_axis(houghLineOk, 2, lines)
    masked = lines[mask]
    clean_lines = masked.reshape(masked.shape[0], 1, 4)

    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, clean_lines)




    return line_img, clean_lines, blurred, edges, ylsmask



# Funcio que fa el pipeline de una sola imatge. Retorna una llista de polygons. Idealment 2

def pipeline(image):
    sh = image.shape
    hor = sh[0] * 0.45
    hor_width = sh[1] * 0.15

    result, lines, blurred, edges, ylsmask = process(image, hor, hor_width)

    # Find vanishing point


    xv, yv = vanishingPoint(lines)


    print("Vanishing", xv, yv)

    # First process, lines must not be very horizontal. It is important
    # As lanes are not perpendicular. that way we kill most of cars lines
    # To be done. Probably a select on the array

    # print(lines.shape)

    #if yv != 100:
    #    hor = yv

    extended_lines = np.apply_along_axis(extend, 2, lines, sh, hor)
    #    for i in range(lines.shape[0]):
    #       print(lines[i], "->", extended_lines[i])

    # Now we may build polygons for the real lines that we want to get
    # To do it we use the fact that all interesting lines touch the bottom (?)
    # and the horizon so just get xmin, xmax at top and bottom
    # Also we group lines nearer than 30 pixels in the bottom.
    #
    #  May do some more intelligent system as classifiying but seems enough

    # print(extended_lines[:, 0, 2])
    sorted_lines = extended_lines[np.argsort(extended_lines[:, 0, 2])]

    # Polygon computation

    xh0 = sorted_lines[0, 0, 0]
    xh1 = sorted_lines[0, 0, 0]
    xb0 = sorted_lines[0, 0, 2]
    xb1 = sorted_lines[0, 0, 2]
    yh0 = sorted_lines[0, 0, 1]
    yb0 = sorted_lines[0, 0, 3]

    acxb = 0
    acxh = 0

    polygons = []
    clines = []
    nlines = 0

    for l in sorted_lines:
        x0 = l[0, 0]
        y0 = l[0, 1]
        x1 = l[0, 2]
        y1 = l[0, 3]

        #abs(y1-y0) > (abs(x1 - x0)*0.4) and

        if  abs(x0 - sh[1]/2) < sh[1]/20:  # Kill not vertical lines

            if abs(nlines == 0 or (acxb/nlines) - x1) < 50 :  # Consolidation of lines!!!

                xh0 = min(xh0, x0)
                xh1 = max(xh1, x0)
                xb0 = min(xb0, x1)
                xb1 = max(xb1, x1)
                yh0 = y0
                yb0 = y1
                acxb = acxb + x1
                acxh = acxh + x0
                nlines = nlines + 1

            else:
                # Build new polygon. They are a list of vertexes

                # Check if is a true line. Width > 15

                if nlines > 0 and abs(xb0 - xb1) > 10 and abs(xb0 - xb1) < 150 and abs(xh0-xh1) < 150 :

                    xbm = acxb / nlines
                    xhm = acxh / nlines

                    clines.append(canonicalLine([[xhm, hor, xbm, sh[0]]]))
                    d = abs(xb0-xb1)/2

                    poly = np.array(
                        [[(xbm-d, yb0), (xhm, yh0),
                          (xbm+d, yb0)]], dtype=np.int32)
                    polygons.append(poly)

                xh0 = x0
                xh1 = x0
                xb0 = x1
                xb1 = x1
                yh0 = y0
                yb0 = y1
                acxb = x1
                acxh = x0
                nlines = 1

    if nlines > 0 and abs(xb0 - xb1) > 10 and abs(xb0 - xb1) < 150 and abs(xh0-xh1) < 150 :
        xbm = acxb / nlines
        xhm = acxh / nlines
        d = abs(xb0 - xb1) / 2
        clines.append(canonicalLine([[xhm, hor, xbm, sh[0]]]))
        poly = np.array(
            [[(xbm - d, yb0), (xhm, yh0),
              (xbm + d, yb0)]], dtype=np.int32)
        polygons.append(poly)

    return np.array(polygons), clines, result, xv, yv, blurred, edges, sorted_lines, ylsmask

def simplify( clines, xv, yv, ybot):

    dleft = -10000
    dright = 10000

    left = clines[0]
    right = clines[len(clines)-1]

    for cline in clines:
        xhor = (yv - cline[0]) / cline[1]
        xbot = (ybot - cline[0]) / cline[1]
        d = abs(xhor - xv)
        dbot = xbot - xv

        if dbot < xv:   #left lane
            if d < dleft:
                left = cline

        else:
            if d < dright:
                right = cline


    return left, right

def cannonical2endpoint(left, top, bot):
        return [[np.int32((top - left[0])/left[1]), np.int32(top),np.int32((bot - left[0])/left[1]), np.int32(bot) ]]

    #list images
for file in os.listdir("lost_frames"):

    if file == "308.jpg":

        image = cv2.imread("lost_frames"+"/"+file)
        print("Shape : ",image.shape)

        lane_lines, clines, resultado, xv, yv, blurred, edges, sorted, ylsmask = pipeline(image)


        left, right = simplify(clines, xv, yv, image.shape[0])
        line_array = np.array(
            [cannonical2endpoint(left, yv, image.shape[0]), cannonical2endpoint(right, yv, image.shape[0])])

        print("Lane Lines ",lane_lines.shape[0])
        # Now
        # Build a composite image

        cv2.imshow("original", image)
        cv2.imshow("blurred", blurred)
        cv2.imshow("edges", edges)
        composite = weighted_img(resultado, image)
        cv2.imshow("hough", composite)

        extended = np.copy(image)
        draw_lines(extended, sorted, thickness=1)
        cv2.imshow("extended", extended)
        poly_image = polyImage(lane_lines, image.shape)
        poly_composite = weighted_img(poly_image, image)

        draw_lines( poly_composite,line_array, color=[255,0,0])

        drawPoint(poly_composite, xv, yv, [0, 255, 0])
        cv2.imshow("end detection", poly_composite)

        cv2.waitKey(0)

cv2.destroyAllWindows()
#cap.release()
