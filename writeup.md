## Writeup - Advanced Lane Lines Projects - REVIEWED

### REVIEWED ITEMS

* Insertion o RED channel in color_gradient() function
* Insertion of the class Line for entire project to manage data
* Inclusion of lane.detected instance to control Blind search and Window search
* Correction of bird_view() function to get more pixels and improve pikes detection in the histogram
* "X" values passed as average of last iterations
* Correction of relative center position calculation
* Insertion of SANITY check 


### This writeup contain a description about how was found the solution for advanced lane lines detection as well an explanation about the code.
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
[image5]: ./examples/bird_view_straight.png "Bird View Image"
[image6]: ./examples/bird_view_binary.png "Bird View Binary"
[image7]: ./examples/histogram.png "Histogram"
[image8]: ./examples/find_lanes.png "Histogram"
[image9]: ./examples/find_next_lane.png "Histogram"
[image10]: ./examples/draw_color_lanes.png "Histogram"
[image11]: ./examples/reviewed_bird_view.png "Histogram"
[image12]: ./examples/reviewed_histogram.png "Histogram"
[video1]: ./Project_ADVLANES.mp4 "Project Video"
[video2]: ./Project-CHALLENGE.mp4 "Challenge Video"
[video3]: ./Project-Review.mp4 "Reviewed Video"


### Development Environment
    Intel® Core™ i5-5200U CPU @ 2.20GHz × 4 - 6Gb RAM
    GeForce 930M/PCIe/SSE2 - 2Gb RAM
    Ubuntu 17.04 - 64bits
    Jupyter NoteBook - Python 3.6

### Project Files

1. writeup.md (This WriteUp)
2. adv_lanes.ipynb (Jupyter Notebook pipeline to Project Video)
3. challenge.ipynb (Jupyter Notebook pipeline to Challenge Video)
4. Line.py (Line class file)
5. Project_ADVLANES.mp4 (Project Video)
6. Project-CHALLENGE.mp4 (Challenge Video)
7. calibration.p (Calibration parameters)
8. Project_Review.mp4 (Video of REVIEWED code)

### 1. Camera Calibration and Image Undistortion 

The code for this step is contained in the first code cell of the IPython notebook located in "adv_lanes.ipynb" , in the function calibrate_camera().

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. 

The function calibrate_camera() returns the 'mtx' and 'dist' coefficients who are used in a second function called undist() (also in the first cell of the jupyter notebook). 

I applied this distortion correction to the test image using the undist() , who takes as arguments an image, mxt and dist coefficients, and will be used in the entire pipeline.

Chessboard Original 
![alt text][image1]
Chessboard Undistorced 
![alt text][image2]
A test image Undistorced
![alt text][image3]


### 2. Color and Gradient pipeline for binary images generation 

I used a combination of color and gradient thresholds to generate a binary image in the function color_gradient() , in the cell number 6 , in the jupyter notebook adv_lanes.  Tho process the images I implemented all the techniques that could be used , and than I selected the ones who gave me the best result. Above, all the image processing functions.

    abs_sobel_thresh() - Calculate the Sobel transform
    mag_thresh() - Magnitude Sobel Transform
    dir_threshold - Direction Sobel Transform
    RGB_Split() - Split RGB Channels
    HLS_Split() - Split HLS Channels 
    thresh_color_channel() - Apply Threshold to Color Channels 
    gaussian_blur() - Apply the Gaussian Filter (Just for the Challenge Pipeline challenge.ipynb)


The color_gradient() funtion:

    def color_gradient (img):  
    img = undist(img, mtx, dist)
    _,_,s_channel = HLS_Split(img)
    s_binary = thresh_color_channel(s_channel, thresh_min=100, thresh_max=255)
    sobel_binary = abs_sobel_thresh(img, orient='x', thresh_min=20, thresh_max=150)
    mag_binary = mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255))  
    
    combined_binary = (sobel_binary | s_binary)
    
    return img,combined_binary
    
 REVIEWED color_gradient() Funtion with insertion of the RED channel:
 
       def color_gradient (img):  
    
        img = undist(img, mtx, dist)
         _,_,s_channel = HLS_Split(img)
         r_channel,_,_ = RGB_Split(img)
         s_binary = thresh_color_channel(s_channel, thresh_min=100, thresh_max=255)
         r_binary = thresh_color_channel(s_channel, thresh_min=150, thresh_max=255)
         sobel_binary = abs_sobel_thresh(img, orient='x', thresh_min=20, thresh_max=150)
         mag_binary = mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255))  
    
         combined_binary = (sobel_binary | s_binary) | r_binary
    
         return img,combined_binary
    
    

    
    
 Above is possible to see that the binary image is composed by a Sobel transform related to "x" , the "s" color channel  thresholded and the "r" color channel  thresholded . The composition is made using a bit wise "OR" operation.
 
 Below, an examples of the output of the color_gradient() funtion.
 
 
![alt text][image4]



 
### 3. "Bird View" function and the Perspective Transform

The code for my perspective transform includes a function called bird_view(), which appears in first cell of the adv_lanes.ipynb file.  The bird_view() function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

    src_pts = np.float32([[245,720],[590,450],[685,450],[1060,720]])

    dst_pts = np.float32([[245,720],[170,0],[1060,0],[1040,720]])
    
REVIWED SRC and DIST points:

    src_pts = np.float32([[300,720],[590,450],[685,450],[1000,720]])
    dst_pts = np.float32([[300,720],[300,0],[1000,0],[1000,720]])


This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 300, 720      | 300, 720        | 
| 590, 450      | 300, 0      |
| 685, 450     | 1000, 0      |
| 1000, 720      | 1000, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

### 4. Finding Lanes and Polynomial Fitting

Here the approach was different for the project video and the challenge video. For both solutions were used two functions , find_lanes() and find_next_lane(). For the Project Video, the entire pipeline is pretty much simpler than for the challenge video. To organise better the code development, first , with a simple approach I solved the project video and then I implement some more sophisticate techniques to the challenge video(challenge.ipynb) 

In the project video, first the find_lane() function is called and then find_next_lane() function assumes the job, without lane detection.
In the challenge video, in the first iteration find_lane() is called and if find_next_lane() does not detect the lane in the next frame , find_lane() is called again. Another difference from both codes are the way X points are passed to the draw_lanes() function. In the case of the challenge video is passed and average of the last measurements. All theses differences between the two codes approach made the chalenge.ipynb consume much more computer resources , and was pretty hard to fit the model and process the entire video in my simple I5 Asus Notebook with just 6GB of memory. 
In the challenge pipeline, to simplify arguments and returns of functions was implemented the class Line() (Line.py) , imported in the first lines of the code.

Going deeper in the finding lanes problem , the principle is:

a.detect the pikes of the histogram of the binary output of the bird_view() function as below. 

![alt text][image6]


![HISTOGRAM][image7]

a*.REVIEWED - Is very clear the improvement reached changing the bird view points as well the RED Channel 

![alt text][image11]


![HISTOGRAM][image12]

b.Fit the X and Y points to a second degree polynomial. 

![alt text][image8]
![alt text][image9]



### 5. Curvature radius calculation and center position (REVIEWED)

In the second cell of the jupyter notebook is located the function lanes_curvature() , were the radius and center position are  calculated as follow



    def lanes_curvature (ploty, left_fit, right_fit):
    y_eval = np.max(ploty)
    left.radius_of_curvature = int((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right.radius_of_curvature = int((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    lane.pix_to_meters =  '{0:.3g}'.format(((lane.carpos-lane.midpos)*3.7)/(right.allx[719] - left.allx[719]))
    
    return 

 
    

The values as printed inside the draw_lanes() function using the method cv2.putText()

 
    



### 6. Drawing Lanes.(REVIEWED)

The draw_lanes() function is responsible to draw the lane in the original image.


    def draw_lines(undist,warped, Minv, left_fitx, right_fitx, ploty, show = False):

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    #result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    cv2.putText(result,"Left Rad: " + str(int(left.radius_of_curvature)) + "m", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 5)
    cv2.putText(result,"Right Rad: " + str(int(right.radius_of_curvature)) + "m", (100,200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 5)
    cv2.putText(result,"Center Error: " + str(lane.pix_to_meters) + "m", (100,300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 5)
    cv2.putText(result,"Lane Detect: " + str(lane.detected) , (100,400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 5)
    
    if show == True:
        plt.imshow(result)
        
    return result
    

### 7. Lane Detection and SANITY Check.(REVIEWED)

All the core code to detect the lanes , call the right function to find the lanes, as well the counter of iterations and use of averages to output "X" values to the draw_lanes() functions were coded inside the function.
As can be see below, when the instance of the class Line lane.detected are false, the code calls the function to find blindly the lanes (find_lanes()). Was determined that an error greater the 2% in the lane width will be considered as not detected , for the opposite situation the lane is considered detected .

The SANITY check was implemented inside the functions find_lanes() and find_next_lane() as folow:

    ##Sanity Check
    if right.allx[719] - left.allx[719] < 0.98*lane.width or right.allx[719] - left.allx[719] > 1.02*lane.width:
        lane.detected = False
               
    else:
        lane.detected = True

The global counter "i" for iterations is used to prevent memory overflow, stopping and reseting all the "appended" values.


    def advanced_lane_lines(image):
    global i
    i=i+1
    
    
    undistorced,result = color_gradient(image)

    result, Minv  = bird_view(result)
    
    if lane.detected == False:
        find_lanes(result, show = False)
    
    else:
        find_next_lane(result, left.current_fit, right.current_fit, show = False)
        
    left.recent_xfitted.append(left.allx)
    right.recent_xfitted.append(right.allx)
    left.bestx = np.average(left.recent_xfitted[-40:],axis=0)
    right.bestx = np.average(right.recent_xfitted[-40:],axis=0)

    lanes_curvature (lane.ally, left.current_fit, right.current_fit)
    result = draw_lines(undistorced,result, Minv, left.bestx , right.bestx , lane.ally,show=True)
    
    if i > 100:
        left.recent_xfitted = []
        right.recent_xfitted = []
        i=0
          
    
    return result
    
    

    
 
### Pipeline (video)

#### 1. Project Video

[Link to project video](./Project_ADVLANES.mp4)
[Link to REVIEWED video](./Project_Review.mp4)


#### 2. Challenge Video

[Link to challenge video](./Project-CHALLENGE.mp4)

### Discussion

This project was very challenger but in the same way very pleasure. See our skills improving, as well being tested are a very good manner to know our strengths and weakness. Unfortunately I could not test the harder challenge entire video in my computer due memory limitations, but one aspect of the challenge for me became very clear. 
In some way, is not hard to treat different problems with different codes , the problem is to create a robust and generic code who performs reasonable in all situations, for example, for a winding road maybe a second degree polynomial is not enough , maybe have to use 3th or 4th, and makes the code to adapt to different situations , using as base the knowledge I have now, is a tricky job.
Another concern I had is about the complexity of all operation and how to make the code runs live in realtime situation. To process the entire code takes much more time than the video has, so I tend to think Python is not the best solution for embedded solutions, maybe I wrong, but is a question I still have.

About the Review: To further my skills to have this project done the review was indispensable. My code became shorter and pretty much faster, doing the job in a better way. I tried to do my best to implement all the modifications requested , and for sure, even far way from perfection, my project now is better, thank you very much.



