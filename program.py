import cv2
import numpy as np
import torch
import torch.nn.functional as F

image_filename = './blocks_L-150x150.png'
image_8bit = cv2.imread(image_filename)
image = np.array(image_8bit / 255.0, dtype=np.float32)


def build_gaussian_pyramid(image, octaves, intervals, sigma):
    # "... We must produce s + 3 images in the stack of blurred images for each octave ..."
    sig = np.zeros(intervals + 3)
    pyramid = []

    # partial sigma values
    # the gaussian blur will blur a previously blurred image, so only perform an incremental blur to get to the correct sigma
    k = 2**(1/intervals)
    sig[0] = sigma
    sig[1] = sigma * np.sqrt(k*k - 1)
    for i in range(2, intervals + 3):
        sig[i] = sig[i-1] * k

    for o in range(0, octaves):
        pyramid.append([])
        for i in range(0, intervals + 3):
            if (i == 0 and o == 0):
                pyramid[o].append(image)
            elif i == 0:
                # "... it will be 2 images from the top of the stack ..."
                src = pyramid[o-1][((intervals + 3) - 1) - 2]
                pyramid[o].append(cv2.resize(src=src, dsize=(src.shape[0]//2, src.shape[1]//2), interpolation=cv2.INTER_NEAREST))
            else:
                pyramid[o].append(cv2.GaussianBlur(pyramid[o][i-1], (0,0), sig[i]))
                
    return pyramid


def build_difference_of_gaussians_pyramid(gaussian_pyramid, octaves, intervals):
    pyramid = []
    for o in range(0, octaves):
        pyramid.append([])
        for i in range(0, intervals + 3 - 1):
            g1 = gaussian_pyramid[o][i]
            g2 = gaussian_pyramid[o][i+1]
            pyramid[o].append(g2 - g1)
    return pyramid


def dog_ord1_partial_derivs(dog, octave, interval, row, col):
    dx = (dog[octave][interval][row+1][col] - dog[octave][interval][row-1][col]) / 2.0
    dy = (dog[octave][interval][row][col+1] - dog[octave][interval][row][col-1]) / 2.0
    ds = (dog[octave][interval+1][row][col] - dog[octave][interval-1][row][col]) / 2.0
    return np.array([[dx, dy, ds]]).T


# https://en.wikipedia.org/wiki/Finite_difference#Generalizations
def dog_ord2_partial_derivs(dog, octave, interval, row, col):
    o = octave
    i = interval
    r = row
    c = col
    f = dog[o]
    v = f[i][r][c]
    
    dxx = (f[i][r+1][c] - 2*v + f[i][r-1][c])
    dyy = (f[i][r][c+1] - 2*v + f[i][r][c-1])
    dss = (f[i+1][r][c] - 2*v + f[i-1][r][c])
    dxy = (f[i][r+1][c+1] - f[i][r+1][c-1] - f[i][r-1][c+1] + f[i][r-1][c-1]) / 4.0
    dxs = (f[i+1][r+1][c] - f[i-1][r+1][c] - f[i+1][r-1][c] + f[i-1][r-1][c]) / 4.0
    dys = (f[i+1][r][c+1] - f[i-1][r][c+1] - f[i+1][r][c-1] + f[i-1][r][c-1]) / 4.0

    return np.array([[dxx, dxy, dxs],
                     [dxy, dyy, dys],
                     [dxs, dys, dss]])


def should_accept_pixel(dog, octave, interval, row, col):
    # section 4
    H = dog_ord2_partial_derivs(dog, octave, interval, row, col)
    H_inv = np.linalg.inv(H)
    deriv = dog_ord1_partial_derivs(dog, octave, interval, row, col)
    x_hat = -H_inv @ deriv

    # " ... If the offset [x_hat] is larger than 0.5 in any dimension, then it means that the ex-
    #       tremum lies closer to a different sample point ... "
    if np.any(np.abs(x_hat) > 0.5):
        return False

    D = dog[octave][interval][row][col]    
    D_of_x_hat = D + 0.5 * deriv.T @ x_hat
    good_value = np.linalg.norm(D_of_x_hat) >= 0.02

    # section 4.1
    r = 10
    Tr_of_H = H[0][0] + H[1][1]
    Det_of_H = H[0][0] * H[1][1] - H[0][1]**2
    good_edge = Tr_of_H**2 / Det_of_H  < (r + 1)**2 / r

    return good_value and good_edge


def is_extremum(difference_of_gaussians_pyramid, octave, interval, row, col):
    # " ... each sample point is compared to its eight neighbors
    #       in the current image and nine neighbors in the scale above and below
    #   ... It is selected only if it is larger than all of these
    #       neighbors or smaller than all of them ... "
    dog = difference_of_gaussians_pyramid
    center = dog[octave][interval][row][col]
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                neighbor = dog[octave][interval+i][row+j][col+k]
                if center >= 0:
                    if center < neighbor:
                        return False
                else:
                    if center > neighbor:
                        return False
    return True


def scale_space_extrema(difference_of_gaussians_pyramid, octaves, intervals):
    features = []
    for o in range(0, octaves):
        print("Octave", o)
        for i in range(1, intervals):
            print("Interval", i)
            height, width = difference_of_gaussians_pyramid[o][0].shape
            for r in range(1, height-1):
                for c in range(1, width-1):
                    if is_extremum(difference_of_gaussians_pyramid, o, i, r, c):
                        if should_accept_pixel(difference_of_gaussians_pyramid, o, i, r, c):
                            features.append((o, i, r, c))
    return features


def calc_grad_mag_ori(gaussian_pyramid, o, i, r, c):
    # " ... For each image sample, L(x, y), at this
    #       scale, the gradient magnitude, m(x, y),
    #       and orientation, θ(x, y), is precomputed
    #       using pixel differences ... "
    f = gaussian_pyramid[o][i]
    if r > 0 and r < f.shape[0] - 1 and c > 0 and c < f.shape[1] - 1:
        dx = f[r+1][c] - f[r-1][c]
        dy = f[r][c+1] - f[r][c-1]
        magnitude = np.sqrt(dx**2 + dy**2)
        direction = np.atan2(dy, dx)
        return magnitude, direction
    return 0, 0


def orientation_assignment(gaussian_pyramid, features, intervals, sigma_init):
    new_features = []
    num_bins = 36
    
    for o, i, r, c in features:
        scale = sigma_init * (2.0 ** (o + i / intervals))
        sigma = sigma_init * scale
        rad = int(np.round(3.0 * sigma))

        hist = np.zeros((num_bins,))
        for k in range(-rad, rad+1):
            for l in range(-rad, rad+1):
                magnitude, direction = calc_grad_mag_ori(gaussian_pyramid, o, i, r+k, c+l)
                # " ... Each sample added to the histogram is weighted by its gradient magni-
                #       tude and by a Gaussian-weighted circular window with a σ that is
                #       1.5 times that of the scale of the keypoint. ... "
                w = np.exp(-(k*k + l*l) / (2 * sigma**2))
                bin_idx = int(np.round(36 * (direction + np.pi) / (2 * np.pi))) % 36
                hist[bin_idx] += magnitude * w

        dom = np.max(hist)
        for b in range(0, num_bins):
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
                new_features.append((o, i, r, c, d))
    return new_features


def compute_descriptors(features, gaussian_pyramid, d, n, sigma_init):
    new_features = []
    for o, i, r, c, ori in features:
        hist = np.zeros((d, n))
        cos_t = np.cos(ori)
        sin_t = np.sin(ori)
        bin_per_rad = n / (2 * np.pi)
        exp_denom = d * d * 0.5
        scale = sigma_init * (2.0 ** (o + i / intervals))
        hist_width = 3 * scale
        radius = int(hist_width * np.sqrt(2) * (d + 1.0) * 0.5 + 0.5)
        for k in range(-radius, radius+1):
            for l in range(-radius, radius+1):
                c_rot = (l * cos_t - i * sin_t) / hist_width
                r_rot = (l * sin_t + i * cos_t) / hist_width
                rbin = r_rot + d / 2 - 0.5
                cbin = c_rot + d / 2 - 0.5

                if rbin > -1 and rbin < d and cbin > -1 and cbin < d:
                    magnitude, direction = calc_grad_mag_ori(gaussain_pyramid[o][i], r+k, c+l)
                    grad_ori -= ori
                    while grad_ori < 0:
                        grad_ori += np.pi * 2
                    while grad_ori >= np.pi * 2:
                        grad_ori -= np.pi * 2

                    obin = grad_ori * bins_per_rad
                    w = np.exp(-(c_rot**2 + r_rot**2) / exp_denom)
                    # interp_hist_entry
        #hist_to_descr
    return new_features

def sift_features(image, intervals, sigma):
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    sigma_gaussian = np.sqrt(max(sigma*sigma - 0.5*0.5, 0.1))
    image_smooth = cv2.GaussianBlur(image_gray, (0,0), sigma_gaussian)
    octaves = round(np.log(np.min(image_smooth.shape)) / np.log(2) - 2)
    gaussian_pyramid = build_gaussian_pyramid(image_smooth, octaves, intervals, sigma)
    difference_of_gaussians_pyramid = build_difference_of_gaussians_pyramid(gaussian_pyramid, octaves, intervals)
    features = scale_space_extrema(difference_of_gaussians_pyramid, octaves, intervals)
    features_with_ori = orientation_assignment(gaussian_pyramid, features, intervals, sigma)
    return gaussian_pyramid, difference_of_gaussians_pyramid, features, features_with_ori
    

def hstack_resize(array_of_images):
    output = [array_of_images[0]]
    s = array_of_images[0].shape
    for i in range(1, len(array_of_images)):
        si = array_of_images[i].shape
        output.append(np.pad(array_of_images[i], ((0, s[0]-si[0]), (0,0))))
        
    return np.hstack(output)


def organize_scale_space(scale_space):
    vstacked = []
    for i in scale_space:
        images = []
        for j in i:
            images.append(np.pad(j, ((4,4), (4,4))))
        vstacked.append(np.vstack(images))

    return hstack_resize(vstacked)


def opencv_sift():
    gray= cv2.cvtColor(image_8bit,cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp = sift.detect(gray,None)

    print(f'There are {len(kp)} REAL keypoints')
 
    s=cv2.drawKeypoints(gray,kp,gray)
    return s


if __name__ == '__main__':
    cv2.namedWindow("Display window", cv2.WINDOW_NORMAL)
    gaussian_pyramid, difference_of_gaussians_pyramid, features, features_with_ori = sift_features(image, 5, np.sqrt(2)/2)
    
    features_image = np.zeros((image.shape[0], image.shape[1],))
    for o, i, r, c in features:
        rs = r * 2**o
        cs = c * 2**o
        features_image[rs][cs] = 1.0

    features_with_ori_image = np.zeros((image.shape[0], image.shape[1],))
    for o, i, r, c, d in features_with_ori:
        rs = r * 2**o
        cs = c * 2**o
        features_with_ori_image[rs][cs] = 1.0

    image_with_keypoints = np.copy(image)
    for o, i, r, c, d in features_with_ori:
        rs = r * 2**o
        cs = c * 2**o
        cv2.circle(img=image_with_keypoints, center=(cs, rs), radius=3, color=(255, 0, 0), thickness=1)
            
    print(f'There are {len(features)} key points')
    print(f'There are {len(features_with_ori)} key points (after orientation assignment)')
    blurred = organize_scale_space(gaussian_pyramid)
    dog = cv2.convertScaleAbs(organize_scale_space(difference_of_gaussians_pyramid), alpha=0.5, beta=0.5)

    # cv2.imshow("Display window", hstack_resize([blurred, dog, features_image, features_with_ori_image]))
    cv2.imshow("Display window", image_with_keypoints)
    while cv2.waitKey(0) & 0xFF != ord('q'):
        pass
    cv2.destroyAllWindows()
