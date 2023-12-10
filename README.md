# lane_detection_oneFrame

## Lane Detection using OpenCV and Python

This README file provides an overview of the Python script for lane detection using OpenCV.

### Functionality

This script implements a lane detection algorithm for real-time video processing using OpenCV. It includes the following functionalities:

* **Color and Edge Detection:**
    * Converts the input image to HLS color space for enhanced lane visibility.
    * Applies thresholding and Gaussian blur to extract lane edge information.
* **Region of Interest (ROI) Masking:**
    * Focuses processing on relevant areas of the image, such as the bottom half of the road.
* **Perspective Transformation (Bird's Eye View):**
    * Warps the image to a bird's eye view for improved lane line detection.
* **Hough Lines with RANSAC Regression:**
    * Identifies lane lines based on pixel intensity within sliding windows using Hough lines.
    * Applies RANSAC regression to refine the detected lines and remove outliers.
* **Lane Filling and Visualization:**
    * Fills the detected lane area with color and overlays it on the original image.
    * Plots the final output with lane lines highlighted.

### Prerequisites

This script requires the following:

* Python 3.8
* OpenCV library
* numpy library
* matplotlib library (optional, for visualization)

### Usage

1. Install the required libraries.
2. Download the Python script and video file.
3. Run the script using the following command:

```python
python lane_detection.py
```

4. The script will process the video and generate a new video with overlaid lane lines and curvature information.

### Script Breakdown

The script performs the following steps:

1. Reads the input image.
2. Converts the image to HLS color space.
3. Applies Gaussian blur and Canny edge detection.
4. Defines a mask for the ROI.
5. Applies the mask to the edge image.
6. Uses Hough lines transformation to detect lane lines.
7. Applies RANSAC regression to refine the detected lines.
8. Fills the detected lane area with color.
9. Overlays the lane fill on the original image.
10. Saves the final output as a video.

### Notes

* This script is for educational purposes only and should not be used in real-world self-driving applications.
* The script may not be robust to various lighting conditions, road markings, and other environmental factors.

### Further Development

* Implement lane curvature estimation and vehicle position calculation.
* Improve lane detection accuracy by incorporating additional techniques.
* Adapt the script to handle various road environments and weather conditions.

## License

This script is provided under an open-source license. You can freely use, modify, and distribute it under the terms of the license.

We hope this README file provides a comprehensive understanding of the lane detection script and its functionality. Please feel free to contact us if you have any questions or suggestions.
