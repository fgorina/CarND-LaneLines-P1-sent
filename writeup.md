# **Finding Lane Lines on the Road**



**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"
[image2]: ./test_images_output/solidWhiteRight.jpg "Vanishing"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline uses 3 main steps

- Image Processing
- Line Processing
- Lane line selection

### 1.1 Image Processing

First the image must be enhanced for yellows. This is really important in the challenge video.
To do it, the program converts the image to HSV, extracts the channels, finds all pixels with
a Hue between 20 and 40 and over those pixels reduces saturation and increases value.

Channels are once again merged together and converted to RGB.

Then image goes through the sequence of
- Convert to grayscale
- Apply Gauss kernel
- Apply Canny algorithm for edge detection
- Apply Hough transformation to get line segments

These segments are returned not only as a picture but also as geometric lines which
will be used by the rest of the program.

Then it computes the vanishing point and filters the segments acording its position and direction. Segments to the right of the vp must point left and viceversa.

![Vanishing Point][image2]

### 1.2 Line Processing

Lines from the last step are extended from horizon level to bottom and are grouped according x position at bottom. For each group the central line and a enclosing polygon are computed.

### 1.3 Lane line selection

We take the set of lines and select the ones that will be the left and right lane. There are 2 algorithms used, one for fixed images and one for video.

#### 1.3.1 Fixed images

Select the two central sets. It works quite well.

#### 1.3.2 Video

Video line selection uses the fact that there is some coherence from one frame to the other, so the nearest lines to the old frame lines are selected.

To get the concept of _nearest_ line a _distance_ between lines is defined as the sum of the absolute value of the difference of x positions at bottom and at horizon level.

So we select, for the left and right lane lines the two ones with minimum distance to the ones in the last frame.

It has been a bit hard to implement in the notebook as I had to put some globals to maintain the state. A suggestion is to have a state variable in the **process_image** that may be passed between invocations.

Finally, using the same time coherence between frames, if there is a frame where no lines are detected the ones from the last frame are used.

The algorithm works quite well and has been tested with a somewhat less straight road. Except in short radius bends it works OK.


### 2. Identify potential shortcomings with your current pipeline

The algorithms are very sensitive to the region of interest used. It has to be tunned for different cameras. It is reasonable as when we change orientation or focal length of the camera the lanes change position.

I believe this could be a problem if the camera moves. One posibility would be to use the vanishing point to get the horizon level.

Also problems in curvy roads.

### 3. Suggest possible improvements to your pipeline

Better image processing would mean having some type of equalization so we adapt to different light conditions and road colors.

For curves one posibility is to break the image horizontally in bands. Each band may be treated independently and use some selection criteria for linking lines in every band. That vould acomodate curves much better.

Also the pooling of lines together may be much better. Perhaps using some type of clasiffier or just considering all possible lane lines and voting according the segments and distance between them and the line.

Get some way to know how good is a line estimation. Also most of the functions just work but don't fail gracefully.
