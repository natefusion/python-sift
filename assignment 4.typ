#align(center+horizon)[
    #text(size:16pt)[*Assignment 4*]
    
    *Nathan Piel*

    CAP 5410: Advanced Computer Vision
    
    #linebreak()
    #linebreak()

    Department of Computer Science

    Florida Polytechnic University, Florida, USA

    11/09/25
]

#pagebreak()
#outline()

#set heading(numbering: "1.")
#pagebreak()

= Introduction
SIFT stands for Scale Invariant Feature Transform. It is an algorithm "for extracting distinctive invariant features from images that can be used to perform reliable matching between different views of an object or scene @sift[p.~1]. Here, I will implement a portion of the SIFT algorithm as described in the paper @sift, including scale space extrema detection, key point localization, and orientation assignment. I will walk through the implementation of these parts and display the outputs generated using rotated and scaled variations of a test image.

= Scale Space Extrema Detection @sift[ch. 3]
The scale space extrema are key points in an image that are invariant across scales. This works by looking for stable key points across many different image scales. The algorithm starts by converting the image to grayscale (You can follow along in `sift.py`, start at `sift_featuress`). This is because this algorithm works best on seeing the changes in brightness in the image.

Next, the gaussian pyramid is built. It is a data structure with progressively blurred and scaled copies of the input image. The octaves represent the image scale. The first octave starts with the original image size and the next octaves progressively divide the image size by two until the image dimensions are about 4x4 pixels. Each octave is divided into intervals. Each interval progressively applies a gaussian blur to the image such that the amount of blurring is separated by a constant factor. The correct blurring for each image is achieved by creating an array of differential #sym.sigma values such that blurring the previous interval with the appropriate #sym.sigma value will result in a correctly blurred image. The #sym.sigma values are calculated like so:
```python
k = 2**(1/intervals)
sig[0] = sigma
sig[1] = sigma * np.sqrt(k**2 - 1)
for i in range(2, intervals + 3):
    sig[i] = sig[i-1] * k
```
Also, each octave represents a doubling in the amount of blurring. This is implemented in `build_gaussian_pyramid`.

Now, the difference of gaussians pyramid is built. This is done by simply subtracting adjacent intervals in each octave of the gaussian pyramid. This is implemented in `build_difference_of_gaussians_pyramid`.

To find the extremum, the algorithm looks at each interval and its two adjacent intervals (the first and last interval are ignored because they only have one adjacent interval). For each pixel, the algorithm compares 26 of its neighbors. One 3x3 set from the left and right intervals and the eight adjacent pixels from the center interval. If this pixel is greater than every neighbor or less than every neighbor it is counted as an extremum. This is implemented in `is_extremum`.

The whole scale space extrema calculation is performed in `scale_space_extrema`.

#figure(image("scale_space_extrema_REGULAR.png"),
    caption : [
        The original image. There are 58 keypoints.
    ])

#figure(image("scale_space_extrema_FLIPPED.png"),
    caption : [
        The image rotated by 90#sym.degree. There are 56 key points.
    ])

#figure(image("scale_space_extrema_BIG.png"),
    caption : [
        The image resized to 300x300 pixels. There are 267 keypoints.
    ])

The key points for the original and rotated image appear to be in the same places relative to the object in the image. The scaled image is very messy, but appears to have key points in the same spots as the other two images. This will improve in the next filtering steps.

= Key Point Localization @sift[ch. 4]
Key point localization builds upon the scale space extrema detection by filtering the initial key points.

The first step is to find the extemum of each key point, (x, y, #sym.sigma) where #sym.sigma is a particular interval within an octave, of a taylor series expansion of the difference of gaussians function. This means finding the first order partial derivatives (1PD) and second order partial derivatives (2PD) (which would be finite differences since the image is discrete). The 1PD is multiplied with the inverse of the 2PD to give the extremum. If any dimension of the extremum has a magnitude greater than 0.5, it lies closer to a different sample point and should be discarded.

Next, the value of the extrema (using the taylor series expansion) is found. This is calculated by adding the pixel value with half of the product of PD1 and the extremum. If this value is less than 0.03, then the contrast of that pixel is too low and should be discarded. This is calculated like so:
```python
H = dog_ord2_partial_derivs(dog, octave, interval, row, col)
H_inv = np.linalg.inv(H)
deriv = dog_ord1_partial_derivs(dog, octave, interval, row, col)
x_hat = -H_inv @ deriv
#...
D = dog[octave][interval][row][col]    
D_of_x_hat = D + 0.5 * deriv.T @ x_hat
good_value = np.linalg.norm(D_of_x_hat) >= 0.03
```
Where `dog` is the difference of gaussians pyramid.

Finally, edge responses are eliminated. This is calculated like so:
```python
r = 10
Tr_of_H = H[0][0] + H[1][1]
Det_of_H = H[0][0] * H[1][1] - H[0][1]**2
good_edge = Tr_of_H**2 / Det_of_H  < (r + 1)**2 / r
```

Where `H` is 2PD matrix (The Hessian Matrix). When `good_edge` is false, the key point should be discarded.

The key point localization code is in `keypoint_localization`.

#figure(image("localization_REGULAR.png"),
    caption : [
        The original image. There are 22 keypoints.
    ])

#figure(image("localization_FLIPPED.png"),
    caption : [
        The image rotated by 90#sym.degree. There are 22 key points.
    ])

#figure(image("localization_BIG.png"),
    caption : [
        The image resized to 300x300 pixels. There are 33 keypoints.
    ])

The number of key points has been greatly reduced compared to the previous step. Now, the scaled image has only ten more key points compared to the other two images and are close to the same place. There is likely a way to improve this further with a more correct and complete implementation.

= Orientation Assignment @sift[ch. 5]
This step assigns one or more orientations to each key point.

First, image gradients are calculated within a region around the key point. The gradient magnitudes are distributed into a histogram with 36 bins, covering the 360 degree range. The magnitude is multiplied "by a Gaussian-weighted circular window with a #sym.sigma that is 1.5 times the scale of that keypoint." @sift[p.~13]

Next, the bins with a value that is at least 80% of the dominant bin are chosen. Of the chosen bins, if it is greater than the left and right adjacent bins (with circular wrapping), that bin is added as a feature. Specifically, the values of the adjacent bins are interpolated, which is used to adjust which bin should be chosen as the direction (as a part of the feature). This is implemented like so:
```python
num_bins = 36
# ...
left = num_bins - 1 if b == 0 else b - 1
right = (b + 1) % num_bins

if hist[b] > hist[left] and hist[b] > hist[right] and hist[b] >= dom * 0.8:
    # "... a parabola is fit to the 3 histogram values closest to each peak ..."
    bin_val = b + (0.5 * (hist[left] - hist[right]) / (hist[left] - 2.0*hist[b] + hist[right]))
    if bin_val < 0:
        bin_val += num_bins
    elif bin_val >= num_bins:
        bin_val -= num_bins
    d = 2*np.pi * bin_val / num_bins - np.pi
```
Where `b` is the bin index and `d` is resulting direction to be added to the feature at that particular pixel.

The orientation assignment code is in `orientation_assignment`.

#figure(image("orientation_REGULAR.png"),
    caption : [
        The original image. There are 27 keypoints.
    ])

#figure(image("orientation_FLIPPED.png"),
    caption : [
        The image rotated by 90#sym.degree. There are 31 key points.
    ])

#figure(image("orientation_BIG.png"),
    caption : [
        The image resized to 300x300 pixels. There are 42 keypoints.
    ])

Some of the key points have the same position but have different directions. Comparing the original image with the flipped image, the directions of the vectors on each key point are the same relative to the object in the image.

= Code
The code is found in `sift.py`. I left some comments. Some are quotes from @sift. Others reference a particular line from a reference SIFT implementation @sift_impl usually when I copied some constant that I didn't know how else to derive.

= Discussion
This was not easy to implement. The paper is very hard to read and is either missing lots of important information or is so terse I miss them. I found a reference SIFT implementation @sift_impl that I used very heavily when writing this code. There probably bugs in this implementation, but it seems to work. Rotating the image results in nearly the same output. The scaled up image resulted in more key points but are clustered around the same spots as in the original. Although, I imagine this can be improved with a better implementation. I don't know exactly why the key points are more scattered with the scaled image. The key point orientation also seems to be working. The directions of all the vectors are in the same spots relative to the object in all three image variations.

= Conclusion
The SIFT algorithm implemented in Python was able to successfully compute image features that are scale and rotation invariant. I learned how to implement most of the SIFT algorithm, including finding the scale space extrema, performing key point localization, and orientation assignment. This involved reading through the SIFT paper @sift and a reference implementation @sift_impl. I demonstrated that the algorithm was working by displaying the key points on an image, a scaled image, and a rotated image.

#bibliography("refs.bib")

