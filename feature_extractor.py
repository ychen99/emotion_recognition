import os.path
import numpy as np
import pandas as pd
import tsfel
import pre_processing as pre
import cv2
import io
import zipfile
from PIL import Image
from skimage import io as skio
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt


def feature_extraction_phy(X):
    # X_train = pre.generate()
    name = ['BP Dia_mmHg', 'BP_mmHg', 'EDA_microsiemens', 'LA Mean BP_mmHg', 'LA Systolic BP_mmHg',
            'Pulse Rate_BPM',
            'Resp_Volts', 'Respiration Rate_BPM']
    cfg_file = tsfel.get_features_by_domain('temporal')
    features = tsfel.time_series_features_extractor(cfg_file, X, header_names=name)
    print(features.columns, features)


def feature_extraction_image(subject, task):
    X = 0
    methods = pre.PreProcessing(f'{subject}', f'{task}')
    path = r'X:\BP4D+_v0.2\2D+3D'
    end = f'{subject}.zip'
    image_names = methods.select_image_files(subject, task)

    # Path to ZIP file
    zip_file_path = os.path.join(path, end)

    # Specific path to the JPEG file inside the ZIP
    jpeg_file_path = image_names[0]

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Check if the desired file exists in the ZIP
        if jpeg_file_path in zip_ref.namelist():
            jpeg_data = zip_ref.read(jpeg_file_path)

            image_stream = io.BytesIO(jpeg_data)
            image_stream = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
            img = cv2.imdecode(image_stream, cv2.IMREAD_COLOR)

            # Convert the original image to gray scale
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_gray_resized = cv2.resize(img_gray, (128, 128))

            # Specify the parameters for our HOG descriptor
            win_size = img_gray_resized.shape
            cell_size = (8, 8)
            block_size = (16, 16)
            block_stride = (8, 8)
            num_bins = 9

            # Set the parameters of the HOG descriptor using the variables defined above
            hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)

            # Compute the HOG Descriptor for the gray scale image
            hog_descriptor = hog.compute(img_gray)

            print('HOG Descriptor:', hog_descriptor)
            print('HOG Descriptor has shape:', hog_descriptor.shape)


def feature_extraction_image1(subject, task):
    methods = pre.PreProcessing(f'{subject}', f'{task}')
    path = r'X:\BP4D+_v0.2\2D+3D'
    end = f'{subject}.zip'
    image_names = methods.select_image_files(subject, task)

    # Path to ZIP file
    zip_file_path = os.path.join(path, end)

    # Specific path to the JPEG file inside the ZIP
    jpeg_file_path = image_names[40]

    numbins = 9
    pix_per_cell = (8,8)
    cell_per_block = (2,2)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Check if the desired file exists in the ZIP
        if jpeg_file_path in zip_ref.namelist():
            with zip_ref.open(jpeg_file_path) as file:
                img = skio.imread(file)
                resized_img = resize(img, (128 * 4, 128 * 3))
                print(resized_img.shape)
                fd, hog_img = hog(resized_img, orientations=numbins, pixels_per_cell=pix_per_cell,
                                    cells_per_block=cell_per_block, visualize=True, channel_axis=2)

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

                ax1.axis('off')
                ax1.imshow(resized_img, cmap=plt.cm.gray)
                ax1.set_title('Input image')

                # Rescale histogram for better display
                hog_image_rescaled = exposure.rescale_intensity(hog_img, in_range=(0, 10))

                ax2.axis('off')
                ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
                ax2.set_title('Histogram of Oriented Gradients')
                plt.show()




feature_extraction_image1('F045', 'T1')

# Open the ZIP file
