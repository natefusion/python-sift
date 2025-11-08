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
The scale space extrema are key points in an image that are invariant across scales. This works by looking for stable key points across many different image scales. The algorithm starts by converting the image to grayscale (You can follow along in `sift.py`, start at `sift_featuress`). This is because works best on seeing the changes in brightness in the image.

Next, the gaussian pyramid is built. It has octaves, which determine the image scale. The first octave starts with the original image size and the next octaves progressively divide the image size by two until the image dimensions are about 4x4 pixels. Each octave is divided into intervals. Each interval progressively applies a gaussian blur to the image such that the amount of blurring is separated by a constant factor. Also, each octave represents a doubling in the amount of blurring. This is implemented in `build_gaussian_pyramid`.

Now, the difference of gaussians pyramid is built. This is done by simply subtracting adjacent intervals in each octave of the gaussian pyramid. This is implemented in `build_difference_of_gaussians_pyramid`.

To find the extrema, the algorithm looks at each interval and its two adjacent intervals (the first and last interval are ignored because they only have one adjacent interval). For each pixel, the algorithm compares 26 of its neighbors. One 3x3 set from the left and right intervals and the eight adjacent pixels from the center interval. If this pixel is greater than every neighbor or less than every neighbor it is counted as an extrema. This is implemented in `is_extremum`.

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

= Key Point Localization @sift[ch. 4]
Key point localization builds upon the scale space extrema detection by filtering the initial key points.

The first step is to find the extemum of each key point, (x, y, #sym.sigma) where #sym.sigma is a particular interval within an octave, of a taylor series expansion of the difference of gaussians function. This means finding the first order partial derivatives (1PD) and second order partial derivatives (2PD) (which would be finite differences since the image is discrete). The 1PD is multiplied with the inverse of the 2PD to give the extremum. If any dimension of the extremum has a magnitude greater than 0.5, it lies closer to a different sample point and should be discarded.

Next, the value of the extrema (using the taylor series expansion) is found. This is calculated by adding the pixel value with half of the product of PD1 and the extremum. If this value is less than 0.03, then the contrast of that pixel is too low and should be discarded. This is calculated like so:
```python
H = dog_ord2_partial_derivs(dog, octave, interval, row, col)
H_inv = np.linalg.inv(H)
deriv = dog_ord1_partial_derivs(dog, octave, interval, row, col)
x_hat = -H_inv @ deriv
# ...
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

Where `H` is 2PD matrix. Wheen `good_edge` is false, the key point should be discarded.

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

The number of key points has been greatly reduced compared to the previous step.

= Orientation Assignment @sift[ch. 5]
This step assigns one or more orientations to each key point. The gradients are calculated within a region around the key point. The gradient magnitudes are distributed into a histogram with 36 bins, covering the 360 degree range. The magnitude is multiplied "by a Gaussian-weighted circular window with a #sym.sigma that is 1.5 times the scale of that keypoint." @sift[p.~13]

Next, the bins with a value that is at least 80% of the dominant bin are chosen. Of the chosen bins, if it is greater than the left and right adjacent bins, that bin is added as a feature. Specifically, the values of the adjacent bins are interpolated, which is used to adjust which bin should be chosen as the direction (as a part of the feature).

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

= Conclusion

#bibliography("refs.bib")

