import os.path
import numpy as np
import pandas as pd
import tsfel
import pre_processing as pre
import io
import zipfile
from skimage import io as skio, color
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from skimage.color import label2rgb


def feature_extraction_phy(X):
    # X_train = pre.generate()
    name = ['BP Dia_mmHg', 'BP_mmHg', 'EDA_microsiemens', 'LA Mean BP_mmHg', 'LA Systolic BP_mmHg',
            'Pulse Rate_BPM',
            'Resp_Volts', 'Respiration Rate_BPM']
    cfg_file = tsfel.get_features_by_domain('spectral')
    features = tsfel.time_series_features_extractor(cfg_file, X, header_names=name)
    print(features.columns, features)


# X_train = pre.generate()
# feature_extraction_phy(X_train)


def calculate_min_max_with_increase(multidimensional_list, increase_percent=10):
    array = np.array(multidimensional_list)

    x_min, y_min = np.min(array, axis=0)
    x_max, y_max = np.max(array, axis=0)

    increase_factor = 1 + increase_percent / 100
    x_min_with_increase = x_min / increase_factor
    x_max_with_increase = x_max
    y_min_with_increase = y_min / increase_factor
    y_max_with_increase = y_max * increase_factor

    return x_min_with_increase, x_max_with_increase, y_min_with_increase, y_max_with_increase


def image_process(subject, task, i=107):
    methods = pre.PreProcessing(f'{subject}', f'{task}')
    path = r'X:\BP4D+_v0.2\2D+3D'
    end = f'{subject}.zip'
    image_names = methods.select_image_files()
    feature_positions = methods.read_feeature_head_positions()

    # Path to ZIP file
    zip_file_path = os.path.join(path, end)

    # Specific path to the JPEG file inside the ZIP
    jpeg_file_path = image_names[1]
    fp = feature_positions[i][1]
    crop_bounds = calculate_min_max_with_increase(fp)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Check if the desired file exists in the ZIP
        if jpeg_file_path in zip_ref.namelist():
            with zip_ref.open(jpeg_file_path) as file:
                img = skio.imread(file)
                min_x, max_x, min_y, max_y = map(int, crop_bounds)
                cropped_image = img[min_y:max_y, min_x:max_x]
                input_image = resize(img, (128 * 4, 128 * 3))
                resized_image = resize(cropped_image, (128 * 4, 128 * 3))

    return img, input_image, resized_image


def feature_extraction_hog(subject, task):
    _, input_img, resized_img = image_process(subject, task)

    numbins = 9
    pix_per_cell = (8, 8)
    cell_per_block = (2, 2)

    fd, hog_img = hog(resized_img, orientations=numbins, pixels_per_cell=pix_per_cell,
                      cells_per_block=cell_per_block, visualize=True, channel_axis=2)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 6), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(input_img, cmap=plt.cm.gray)
    ax1.set_title('Input image')

    ax2.axis('off')
    ax2.imshow(resized_img, cmap=plt.cm.gray)
    ax2.set_title('Cropped image')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_img, in_range=(0, 10))

    ax3.axis('off')
    ax3.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax3.set_title('Histogram of Oriented Gradients')

    plt.show()


#local binary patterns
def feature_extraction_lbp(subject, task):
    image, input_img, resized_img = image_process(subject, task)
    image_gray = color.rgb2gray(resized_img)
    METHOD = 'uniform'


    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(image_gray, n_points, radius, METHOD)

    def hist(ax, lbp):
        n_bins = int(lbp.max() + 1)
        return ax.hist(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins),
                       facecolor='0.5')

    # plot histograms of LBP of textures
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))
    plt.gray()
    ax1.imshow(resized_img, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')

    ax2.imshow(lbp, cmap='gray')
    ax2.set_title('LBP Image')
    ax2.axis('off')

    hist(ax3, lbp)
    ax3.set_ylabel('Percentage')

    plt.show()


feature_extraction_lbp('M001', 'T1')
