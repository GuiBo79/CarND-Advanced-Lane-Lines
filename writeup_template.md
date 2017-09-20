## Writeup - Advanced Lane Lines Projects 

### This writeup contain a description about how was foud the solution for , as well, how the pipeline was coded, advanced lane lines. 

---

The goals of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/calibration5.jpg "Chess Original"
[image2]: ./examples/calibration5_undist.jpg "Chess Undistorced"
[image3]: ./examples/test1_undist.jpg "Test Image Undistorced"
[image4]: ./examples/original_image.png "OriginalXBinary Image"
[image5]: ./examples/bird_view.png "Bird View Image"
[image6]: ./examples/bird_view_binary.png "Bird View Binary"
[image7]: ./examples/histogram.png "Histogram"
[image8]: ./examples/find_lanes.png "Histogram"
[image9]: ./examples/find_next_lane.png "Histogram"
[video1]: ./examples/histogram.png "Histogram"


### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### 1. Camera Calibration and Image Undistortion 

The code for this step is contained in the first code cell of the IPython notebook located in "adv_lanes.ipynb" , in the function calibrate_camera().

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. 

The function calibrate_camera() returns the 'mtx' and 'dist' coeficients who are used in a second function called undist() (also in the first cell of the jupyter notebook). 

I applied this distortion correction to the test image using the undist() , who takes as arguments an image, mxt and dist coeficients, and will be used in the entire pipeline.

Chessboard Original 
![alt text][image1]
Chessboard Undistorced 
![alt text][image2]
A test image Undistorced
![alt text][image3]


### 2. Color and Gradient pipeline for binary images generation 

I used a combination of color and gradient thresholds to generate a binary image in the function color_gradient() , in the cell number 6 , in the jupyter notebook adv_lanes.  Tho process the images I implemented all the techniques that could be used , and than I selected the ones who gaves me the best result. Above, all the image processing functions.

abs_sobel_thresh() - Calculate the Sobel transform
mag_thresh() - Magnitude Sobel Transform
dir_threshold - Direction Sobel Transform
RGB_Split() - Split RGB Channels
HLS_Split() - Split HLS Channels 
thresh_color_channel() - Apply Threshold to Color Channels 
gaussian_blur() - Apply the Gaussian Filter (Just for the Challenge Pipeline challenge.ipynb)

The color_image() funtion:

def color_gradient (img):  
    
    img = undist(img, mtx, dist)
    _,_,s_channel = HLS_Split(img)
    s_binary = thresh_color_channel(s_channel, thresh_min=100, thresh_max=255)
      
    
    sobel_binary = abs_sobel_thresh(img, orient='x', thresh_min=20, thresh_max=150)
    mag_binary = mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255))  
    
    combined_binary = (sobel_binary | s_binary)
    
    return img,combined_binary
    
    
 Above is possible to see that the binary image is composed by a Sobel transform related to "x" and the "s" color channel  thresholded . The composition is made using a bit wise "OR" operation.
 
 Below, an examples of the output of the color_gradient() funtion.
 
 
![alt text][image4]



 
### 3. "Bird View" function and the Perpective Trasform

The code for my perspective transform includes a function called bird_view(), which appears in first cell of the adv_lanes.ipynb file.  The bird_view() function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

src_pts = np.float32([[245,720],[590,450],[685,450],[1060,720]])
dst_pts = np.float32([[245,720],[170,0],[1060,0],[1040,720]])


This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 245, 720      | 245, 720        | 
| 590, 450      | 170, 0      |
| 685, 450     | 1060, 0      |
| 1060, 720      | 1040, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

#### 4. Finding Lanes and Polynomial Fitting

Here the approach was different for the project video and the challenge video. For both solutions were used two functions , find_lanes() and find_next_lane(). For the Project Video, the entire pipeline is pretty much simplier than for the challenge video. To organise better the code development, first , with a simple approach I solved the project video and the I implement some more sophisticate techniques to the challenge video(challenge.ipynb) 

In the project video, first the find_lane() function is called and then find_next_lane() function assumes the job, without lane detection.
In the challenge video, in the first iteration find_lane() is called and if find_next_lane() does not detect the lane in the next frame , find_lane() is called again. Another difference from both codes are the way X points are passed to the draw_lanes() function. In the case of the challenge video is passed and average of the last measurements. All theses differences between the two codes approach made the chalenge.ipynb consume much more computer resources , and was pretty hard to fit the model and process the entire video in my simple I5 Asus Notebook with just 6GB of memory. 
In the challenge pipeline, to simplify arguments and returns of functions was implemented the class Line() (Line.py) , imported in the first lines of the code.

Going deeper in the finding lanes problem , the principle is:
        a.detect the pikes of the histogram of the binary output of the bird_view() function as below ![alt text][image6]![alt text][image7]
        
        
        
        




#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
