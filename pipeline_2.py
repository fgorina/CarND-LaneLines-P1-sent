#importing some useful packages
import numpy as np
import cv2
import math
import os
from sklearn import linear_model, datasets
from sys import argv


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

# Increase yellow levels as they are not enough
# Specially designed for the challenge video
#   - Converts image to HSV. I am reading with OpenCV so my original format is BGR
#       if using matplotlib then change BGR to RGB
#   - Split into channels
#   - Yellows are between Hue 20 and 40
#   - Other parameters select saturated yellows
#   - So select yellow points
#   - Reduce their saturation so are more "white"
#   - Increase the Brightness
#   - Merge the channels
#   - Convert to BGR/RGB format
#   - Return the image and the mask for debugging

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


# Checks if line is in a correct angle
#   Just remove more or less horizontal lines
#   And vertical lines

def lineOk(l):

    points = l[0]

    x0 = points[0]
    y0 = points[1]
    x1 = points[2]
    y1 = points[3]

    if abs(y1-y0) > abs(x1 - x0)*0.4 and (x1-x0) != 0 :
        return 1
    else:
        return 0

# Similar function for selecting segments
# Includes the posibility of selecting by length of segments
#  Not used now

def houghLineOk(l, xv):
    points = l

    x0 = points[0]
    y0 = points[1]
    x1 = points[2]
    y1 = points[3]


    if y0 > y1: # swap so x1 y1 is bottom
        t = x1
        x1 = x0
        x0 = t
        t = y1
        y1 = y0
        y0 = t

    if x1 < xv:
        dir_ok = x0 > x1
    else:
        dir_ok  = x0 < x1

    l = math.sqrt((x1-x0)*(x1-x0) + (y1-y0)*(y1-y0))

    return abs(y1-y0) > abs(x1 - x0)*0.3 and l > 0 and  dir_ok


# Builds an image from the polygons

def polyImage(polygons, sh):

    img = np.zeros((sh[0], sh[1], 3), dtype=np.uint8)

    for poly in polygons:
        cv2.fillPoly(img, poly, color=[0, 0, 255])

    return img


# Extends a line to the bottom or border of the image and to the horizon in top direction
# Returns a lines correspondig to the original line but extended

def extend(l, shape, horizon):

    # if line is quite horizontal kill it

    points = l

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

# returns the b and m of the equation y = b + mx giving the two endpoints
# Probably would be better to use equation x = b + my because it allows vertical lines
# will change in ext version

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

# Compute vanishing point of lines. Really the intersecting point
#
#   - Transforms the lines to b,m  (line set -> point set)
#   - Applies a regression to the points (line in b,m)
#   - Computes 2 points of the regression line
#   - Those 2 points allows us to get the original real space point that
#       corresponds to the regression line,that is common to al original lines
#       which is the vanishing point

def vanishingPoint(lines):

    hough_points = []

    #print('Lines ', lines.shape[0])

    # First remove vertical and horizontal lines
    for l in lines:     # Clear bad lines
        if lineOk(l):
            hough_points.append(canonicalLine(l))

    hough_points = np.array(hough_points)
    # Now we make the regression

    if hough_points.shape[0] < 2:
        return 100, 100

    X = np.array(hough_points[:,1])   # m
    Y = np.array(hough_points[:,0])   # b

    X = X.reshape(-1, 1)

    model = linear_model.RANSACRegressor(linear_model.LinearRegression())
    model.fit(X, Y)

    # For a future use.

    inlier_mask = model.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    # Use two arbitrary values of m for computing the real space point
    m1 = 3.0
    m2 = 175.0

    XM = np.array([m1, m2]).reshape(-1, 1)
    YM = model.predict(XM)

    b1 = YM[0]
    b2 = YM[1]

    x = np.int32(round((b2 - b1)/(m1 - m2), 0))
    y = np.int32(round(b1 + m1 * x, 0))

    return x, y

# Draws a cross at a point with a color. Just a utility for
# marking center of image and vanishing point

def drawPoint(img, x, y, color):

    lines = np.array([[[x-10, y, x+10, y]], [[x, y-10, x, y+10]]])

    draw_lines(img, lines, color=color, thickness=2)

# From a canonical line (b, m pair) compute endpoints with y = top and bottom
# and return with a [[xtop, top, xbot, bot ]] format

def cannonical2endpoint(cline, top, bot):
    b = cline[0]
    m = cline[1]
    xtop = np.int32((np.float32(top) - b) / m)
    xbot = np.int32((np.float32(bot) - b) / m)

    return np.array([[xtop, top, xbot, bot ]])

# Basic image processing.
#
#   - Applies an increaseYellow to get more contrast in yellow lines
#   - Converts to grayscale
#   - Applies gaussian blur and canny algorithm
#   - Then apllies Hough Lines Detection just to the region of interesting
#   - roi is a trapez wit top at horizon and top width hor width and bottom the bottom corners of the image
#   - Resulting lines are filtered with houghLineOk
#   - Returns the array of lines

def imageProcessing(img, hor, hor_width):

    yimg, ylsmask = increaseYellow(img)

    # Convert to gray
    gray = grayscale(yimg)

    # Some values to adjust

    kernel_size = 9
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
    min_line_length = 50
    max_line_gap = 70

    draw, lines = hough_lines(roi, rho, theta, threshold, min_line_length, max_line_gap)

    xv, yv = vanishingPoint(lines)

    mask   = np.apply_along_axis(houghLineOk, 2, lines, xv)
    masked = lines[mask]
    clean_lines = masked.reshape(masked.shape[0], 1, 4)

    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, clean_lines)

    #return line_img, clean_lines, blurred, edges, ylsmask # just when debugging
    return clean_lines, xv, yv



# Given the lines from image processing processes the lines
#   - Computes vanishing point. Used sometimes in tests.
#   - Horizontal distance from vanishing point to center of image
#       may be used to get the attitude of the car with respect to the lanes
#       If they are centered we are going straight. If there is son curve
#       or the car is not aligned they will differ
#       Be careful with camera and car alignment
#
#   - Then extend lines so go from horixon to bottom
#   - The idea is to group them by the bottom x position
#   - Sort lines by x bottom position
#   - Compute groups, the enclosing polygon and the centered average line
#   - Now changed the polygon computation so they enclose the bottom but
#   - go to 1 point at horizon/vanishing point. Betten perspective image when displayed
#   - Return polygons, center of polygon lines and vanishing point

def lineProcessing(lines, hor, hor_width, sh):
    # Find vanishing point


    #print("Vanishing", xv, yv)
    extended_lines = np.apply_along_axis(extend, 2, lines, sh, hor)

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

            if nlines == 0 or abs((acxb/nlines) - x1) < 50 :  # Consolidation of lines!!!

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

                    clines.append([canonicalLine([[xhm, hor, xbm, sh[0]]]), nlines])
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
        clines.append([canonicalLine([[xhm, hor, xbm, sh[0]]]), nlines])
        poly = np.array(
            [[(xbm - d, yb0), (xhm, yh0),
              (xbm + d, yb0)]], dtype=np.int32)
        polygons.append(poly)

    return np.array(polygons), clines

#   Given a set of center lines select two, one for left lane line and another
#       for right lane line
#   Actual algorith select the most centered of them.
#   Sometimes is not accurate but it will be safer in any case
#   Video has a different line selection algorithm
#
def laneLineSelectionPhoto(clines, polys , xv, yv, ybot):

    dleft = -10000
    dright = 10000

    lpos = -1000
    rpos = 10000

    nleft = 0
    nright = 0

    cline1 = clines[0]

    left = cline1[0]
    left_poly = polys[0]

    cline1 = clines[len(clines)-1]
    right = cline1[0]
    right_poly = polys[len(clines)-1]

    for i in range(0, len(clines)):
        clinet = clines[i]
        cline = clinet[0]

        xhor = (yv - cline[0]) / cline[1]
        xbot = (ybot - cline[0]) / cline[1]



        d = abs(xhor - xv)
        dbot = xbot - xv

        #if dbot < xv:   #left lane
        if cline[1] <  0:
            #if d < dleft:
            if xbot > lpos:
                lpos = xbot
                left = cline
                dleft = d
                left_poly = polys[i]

        else:
           # if d < dright:
            if xbot < rpos:
                rpos = xbot
                right = cline
                dright = d # d
                right_poly = polys[i]

    return left, right, np.array([left_poly, right_poly])

# Lane line selection for Video
# Given 2 lines from horizon to bottom we define a distance as
#
#   abs(xhor1-xhor0) + abs(abot1 - xbot0)
#
#   We select the two centered lines that are neares to the corresponding
#   last frame left and right lines
#
def laneLineSelectionVideo( clines, polys, xv, yv, ybot, old_left, old_right):

    # compute xvalues for old ones

    hor_old_left = (yv - old_left[0]) / old_left[1]
    hor_old_right = (yv - old_right[0]) / old_right[1]
    bot_old_left = (ybot - old_left[0]) / old_left[1]
    bot_old_right = (ybot - old_right[0]) / old_right[1]

    cline1 = clines[0]
    left = cline1[0]
    delta_left = 1000000000
    left_poly = polys[0]

    cline1 = clines[len(clines)-1]
    right = cline1[0]
    delta_right = 1000000000
    right_poly = polys[len(clines)-1]

    for i in range(0, len(clines)):
        clinet = clines[i]
        cline = clinet[0]

        xhor = (yv - cline[0]) / cline[1]
        xbot = (ybot - cline[0]) / cline[1]
        d = abs(xhor - xv)
        dbot = xbot - xv

        if cline[1] <  0:
            err = abs(xhor - hor_old_left) + abs(xbot - bot_old_left)

            if err < delta_left:
                delta_left = err
                left = cline
                left_poly = polys[i]

        else:
            err = abs(xhor - hor_old_right) + abs(xbot - bot_old_right)
            if err < delta_right:
                delta_right = err
                right = cline
                right_poly = polys[i]

    return left, right, np.array([left_poly, right_poly])


# Main Program

# python pipeline_2.py foto ./test_images   [test_images_output]
# python pipeline_2.py foto ./test_videos/challenge.mp4   [test_images_output]
if len(argv) >= 2:
    modus = argv[1]
else:
    modus = "foto"

if len(argv) >= 3:
    indir = argv[2]
elif modus == "foto":
    indir = "./test_images"
else:
    indir = "./test_videos/challenge.mp4"

if len(argv) >= 4:
    outdir = argv[3]
else:
    outdir = ""

if modus == "foto":
    #list images
    for file in os.listdir(indir):

        if file.lower().endswith(".jpg") or file.lower().endswith(".png") or file.lower().endswith(".jpeg"):

            image = cv2.imread(indir+"/"+file)      # get the image
            hor = np.int32(image.shape[0] * 0.6)    # 0.6 Magic number for this cameras. Tested with a different one and should be changed with focal
            hor_width = np.int32(image.shape[1] * 0.06)   # 0.06Magic number for this cameras. Tested with a different one and should be changed with focal

            lines , xv, yv = imageProcessing(image, hor, hor_width)  # Process image and get the line segments
            lane_lines, clines  = lineProcessing(lines, hor, hor_width, image.shape) # Process line segments and get consolidated lines
            left, right, lane_lines = laneLineSelectionPhoto(clines, lane_lines, xv, yv, image.shape[0])  # Select left and right lines

            # Result image construction

            left_line = cannonical2endpoint(left, hor, np.int32(image.shape[0]))    # COmpute endpoits of lane lines
            right_line = cannonical2endpoint(right, hor, np.int32(image.shape[0]))
            line_array = np.array([left_line, right_line])

            poly_image = polyImage(lane_lines, image.shape)     # Build an image with the polygons corresponding to let/roght lines
            poly_composite = weighted_img(poly_image, image)
            draw_lines( poly_composite, line_array, color=[255,0,0], thickness=2)    # Put blue lines over them
            drawPoint(poly_composite, xv, yv, [0,255,0])    # Draw vanishing point

            if outdir != "":
                cv2.imwrite(outdir + "/" + file, poly_composite)    # Write to output directory

            cv2.imshow(file, poly_composite)    # Show in a window. Loop for every image just pressing a key
            cv2.waitKey(0)
else:

    cap = cv2.VideoCapture(indir)
    ret, old_frame = cap.read() # Get first frame to get sizing
    frame = 0
    lost_frame = 0
    hor = np.int32(old_frame.shape[0] * 0.40) # Magic number for this cameras. Tested with a different one and should be changed with focal
    hor_width = np.int32(old_frame.shape[1] * 0.06) # Magic number for this cameras. Tested with a different one and should be changed with focal

    while True: # Loops till have a good frame.

        lines, old_xv, old_yv = imageProcessing(old_frame, hor, hor_width)  # Process image and get the line segments
        old_lane_lines, old_clines   = lineProcessing(lines, hor, hor_width, old_frame.shape) # Process line segments and get consolidated lines

        if len(old_clines) > 1:
            old_left, old_right, old_lane_lines = laneLineSelectionPhoto(old_clines, old_lane_lines, old_xv, old_yv, old_frame.shape[0])  # Select left and right lines
            break
        else:
            ret, old_frame = cap.read()

 # Now loop over all frames

    while (ret):
        ret, b_frame = cap.read()
        if ret:
            frame = frame + 1
            lines, xv, yv = imageProcessing(b_frame, hor, hor_width)  # Process image and get the line segments
            p_lane_lines, clines  = lineProcessing(lines, hor, hor_width, b_frame.shape) # Process line segments and get consolidated lines

            # Only bif difference with Photo is laneLineSelectionVideo trying to get
            # consistency between different frames
            if len(clines) > 1:  # If not we may not process clines
                left, right, lane_lines = laneLineSelectionVideo(clines, p_lane_lines, xv, yv, b_frame.shape[0], old_left, old_right)
                lost_frame = 0

            # Glups, some frame is really difficult. Try to use last frame up to 5 test_images
            elif p_lane_lines.shape[0] < 2 and lost_frame < 5:
                # When debugging is interesting to write frames where no line recognition
                #cv2.imwrite("lost_frames/" + str(frame) + ".jpg", b_frame)
                #print("Frame ", frame , " only ", lane_lines.shape[0] , "lines detected")

                lane_lines = old_lane_lines
                left = old_left
                right = old_right
                clines = old_clines
                xv = old_xv
                yv = old_yv
                lost_frame = lost_frame + 1

            # Store current values in old variables

            old_left = left
            old_right = right
            old_clines = clines
            old_lane_lines = lane_lines
            old_xv = xv
            old_yv = yv

            # Now convert from cline to normal line format and create an array for drawlines
            left_line = cannonical2endpoint(left, hor, np.int32(b_frame.shape[0]))
            right_line = cannonical2endpoint(right, hor, np.int32(b_frame.shape[0]))
            line_array = np.array([left_line, right_line])

            # Build the new imabe
            poly_image = polyImage(lane_lines, b_frame.shape)
            poly_composite = weighted_img(poly_image, b_frame)

            draw_lines( poly_composite, line_array, color=[255,0,0], thickness=4)

            drawPoint(poly_composite, np.int64(b_frame.shape[1]/2), np.int64(b_frame.shape[0]/2), [0, 0, 255]) # Center of frame
            drawPoint(poly_composite, xv, yv, [0, 255, 0]) # vanishing point
            cv2.imshow('Video', poly_composite )
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                cv2.imwrite("./lost_frames/"+str(frame)+".jpg", b_frame)
                continue
            else:
                continue


cv2.destroyAllWindows()
#cap.release()
