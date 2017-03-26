This is a report of Vehicle Detection and Tracking project.

Video Output: https://www.youtube.com/watch?v=a_QMvUkus6o

Part 1: Explain how (and identify where in your code) you extracted HOG features from the training images

After many trials and errors, I found YCrCb color space leads to the least false positives. (HSV didn't work as well after I experimented with it). Here is the visualization: 

Car image sample 
![screen shot 2017-03-19 at 1 24 26 pm](https://cloud.githubusercontent.com/assets/11469505/24084444/b05fcdb2-0ca7-11e7-997d-40570d043b01.png)

Car Image YCrCb visualization 
![car_plot](https://cloud.githubusercontent.com/assets/11469505/24084447/c45e4cd0-0ca7-11e7-894f-845790fdef5b.png)

Non car image sample
![screen shot 2017-03-19 at 1 25 44 pm](https://cloud.githubusercontent.com/assets/11469505/24084449/d266d91e-0ca7-11e7-9465-c3885e435b21.png)

Non car image YCrCb visualization
![noncar_plot](https://cloud.githubusercontent.com/assets/11469505/24084451/de148554-0ca7-11e7-8259-f6bd7c6facfa.png)

Part 2. Train the classifier 

I got inspirations from the following: 

1. Udacity Self Driving Car Vehicle Detection Class
2. Small U-Net for vehicle detection
https://chatbotslife.com/small-u-net-for-vehicle-detection-9eec216f9fd6#.8ypkhmhb0

I chose SVC since it's fast to train and works well with HOG features. In order to decrease false positives, here are the HOG parameters I picked: 

color_space = 'YCrCb'  
orient = 9
pix_per_cell = 8
cell_per_block = 2
spatial_size = (32, 32)
hist_bins = 32

Here are the steps:

1. Set the parameters above
2. Create car_features and non_car_features
3. Apply np.vstack
4. Fit a scaler X_scaler
5. Apply the scaler to X
6. Define the labels vector
7. Split up data into training and testing (randomized) 
8. Use Linear SVC

Here is the training result: 

It uses 9 orientations,  8 pixels per cell,  2 cells per block,  32 histogram bins, and  (32, 32) spatial sampling. The test accuracy is around 0.98.

Part 3: Sliding window search

To minimize false positives and false negatives, I used y_start_stop = (400, 650), scales = [1.5, 2]. I tried many combinations and this one seems to work the best. Scale larger than 1.5 doesn't seem to work well. 

Here is the original test1 image 

![screen shot 2017-03-19 at 10 12 53 pm](https://cloud.githubusercontent.com/assets/11469505/24089204/40fa49d0-0cf1-11e7-9382-2f43eb52230e.png)

Here is the sliding window on test1 image

![screen shot 2017-03-19 at 10 14 18 pm](https://cloud.githubusercontent.com/assets/11469505/24089209/6be60792-0cf1-11e7-9710-e0d7623b319e.png)

Here is sliding window cluster on test1 image 

![screen shot 2017-03-19 at 10 15 22 pm](https://cloud.githubusercontent.com/assets/11469505/24089220/92603d98-0cf1-11e7-814b-5cc0b262ed7f.png)

Here is the heat map on test1 image

![screen shot 2017-03-19 at 10 15 50 pm](https://cloud.githubusercontent.com/assets/11469505/24089227/a615aab2-0cf1-11e7-82e2-3627ca14f44f.png)

Part 4: Video processing

Here are the steps I took: 

1. Pass in image
2. Use y_start_stop = (400, 650), scales = [1.5, 2]
3. Call find_cars function
4. Apply heat map 
5. Append images with sliding window
6. Append images with sliding window cluster
7. Append images with heat map

Output video sample images:

One white car big
![screen shot 2017-03-19 at 11 27 44 pm](https://cloud.githubusercontent.com/assets/11469505/24090276/e87e6876-0cfb-11e7-94bd-08c8834d8f60.png)

One white car small
![screen shot 2017-03-19 at 11 28 01 pm](https://cloud.githubusercontent.com/assets/11469505/24090278/ee472590-0cfb-11e7-8a1d-8d0f0aeee386.png)

One black car and one white car close 
![screen shot 2017-03-19 at 11 28 12 pm](https://cloud.githubusercontent.com/assets/11469505/24090283/fc037878-0cfb-11e7-99d8-06b8a959dbc6.png)

One black car and one white car further
![screen shot 2017-03-19 at 11 28 23 pm](https://cloud.githubusercontent.com/assets/11469505/24090285/08020d24-0cfc-11e7-94f1-e69b13f3b6d0.png)


Video project_output_mar20.mp4 is submitted along with jupyter notebook


Part 5: Discussion

I learned a lot about multi-scale windows and heatmap in this project. The final output is good in general but there are couple false positives where there is no car. I tried different y_start_stop and scales however still couldn't fully avoid false positives. I would like to try U-net as mentioned in this blog post: https://chatbotslife.com/small-u-net-for-vehicle-detection-9eec216f9fd6#.8ypkhmhb0. The training will take longer (about 2 hours) but it will identify cars correctly in the driving frames it did not see before and has minimum false positives.

