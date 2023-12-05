import os.path
import cv2
import math
import numpy as np
import pandas as pd
import tsfel
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python import solutions
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
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


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


def plot_face_blendshapes_bar_graph(face_blendshapes):
    # Extract the face blendshapes category names and scores.
    face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
    face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
    # The blendshapes are ordered in decreasing score value.
    face_blendshapes_ranks = range(len(face_blendshapes_names))

    fig, ax = plt.subplots(figsize=(12, 12))
    bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
    ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
    ax.invert_yaxis()

    # Label each bar with values
    for score, patch in zip(face_blendshapes_scores, bar.patches):
        plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")
    # print(face_blendshapes_names)
    ax.set_xlabel('Score')
    ax.set_title("Face Blendshapes")
    plt.tight_layout()
    plt.show()


def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)
    landmark_indics = face_landmarks_list[0]
    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_iris_connections_style())

    return annotated_image


def image_process(subject, task, i=107):
    methods = pre.PreProcessing(f'{subject}', f'{task}')
    path = r'X:\BP4D+_v0.2\2D+3D'
    end = f'{subject}.zip'
    image_names = methods.select_image_files()

    # Path to ZIP file
    zip_file_path = os.path.join(path, end)

    # Specific path to the JPEG file inside the ZIP
    jpeg_file_path = image_names[109]

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Check if the desired file exists in the ZIP
        if jpeg_file_path in zip_ref.namelist():
            with zip_ref.open(jpeg_file_path) as file:
                img = skio.imread(file)

                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

                base_options = python.BaseOptions(model_asset_path='face_landmarker.task')

                options = vision.FaceLandmarkerOptions(base_options=base_options,
                                                       output_face_blendshapes=True,
                                                       output_facial_transformation_matrixes=True,
                                                       num_faces=1)
                detector = vision.FaceLandmarker.create_from_options(options)

                detection_result = detector.detect(image)
                face_landmarks_list = detection_result.face_landmarks
                landmark_indics = face_landmarks_list[0]
                annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

                # plot_face_blendshapes_bar_graph(detection_result.face_blendshapes[0])

                #plt.imshow(annotated_image)
                #plt.show()

    return img, landmark_indics, annotated_image


def image_segmentation(image, land_index):
    # y-axis
    eyebrowUpper = [21, 68, 104, 108, 69, 151, 337, 299, 333, 384, 54, 284]  # min
    eyeLower = [111, 117, 118, 119, 120, 121, 350, 349, 357, 348, 347, 346, 340]  # max
    noseCentral = [2]
    lipLower = [176, 148, 152, 377, 400]  # max

    # x-axis
    leftEye = [21, 162, 127]  # min
    rightEye = [234, 251, 389, 264, 356]  # max
    leftCheek = [127, 234, 93]  # min
    rightCheek = [454, 323, 366]  # max
    leftLip = [132, 58, 172, 136, 150, 149]  # min
    rightLip = [361, 288, 397, 365, 379, 378]  # max

    def calculate_min_max(list_index, axis, operation):
        coordinates = [land_index[i].y if axis == 'y-axis' else land_index[i].x for i in list_index]
        return np.max(coordinates) if operation == 'max' else np.min(coordinates)

    def crop_image(img, bounds):
        img_height, img_width = img.shape[:2]

        start_x = int(bounds[0] * img_width)
        end_x = int(bounds[1] * img_width)
        start_y = int(bounds[2] * img_height)
        end_y = int(bounds[3] * img_height)
        return img[start_y:end_y, start_x:end_x]

    cropped_images = []

    first_bound = [calculate_min_max(leftEye, 'x-axis', 'min'),
                   calculate_min_max(rightEye, 'x-axis', 'max'),
                   calculate_min_max(eyebrowUpper, 'y-axis', 'min'),
                   calculate_min_max(eyeLower, 'y-axis', 'max')]

    second_bound = [calculate_min_max(leftCheek, 'x-axis', 'min'),
                    calculate_min_max(rightCheek, 'x-axis', 'max'),
                    calculate_min_max(eyeLower, 'y-axis', 'max'),
                    calculate_min_max(noseCentral, 'y-axis', 'max')]

    third_bound = [calculate_min_max(leftLip, 'x-axis', 'min'),
                   calculate_min_max(rightLip, 'x-axis', 'max'),
                   calculate_min_max(noseCentral, 'y-axis', 'max'),
                   calculate_min_max(lipLower, 'y-axis', 'max')]

    cropped_images.append(crop_image(image, first_bound))
    cropped_images.append(crop_image(image, second_bound))
    cropped_images.append(crop_image(image, third_bound))
    print(cropped_images)
    return cropped_images


def plot_image(subject, task):
    original_image, landmark_indics, annoted_img = image_process(subject, task)
    cropped_images = image_segmentation(original_image, landmark_indics)
    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(1, 3, 1)

    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(3, 3, 3)
    ax4 = fig.add_subplot(3, 3, 6)
    ax5 = fig.add_subplot(3, 3, 9)

    ax1.imshow(original_image, cmap='gray')
    ax1.axis('off')  # Turn off axis
    ax1.set_title('Original Image')

    ax2.imshow(annoted_img, cmap='gray')
    ax2.axis('off')  # Turn off axis
    ax2.set_title('Image with landmark')

    ax3.imshow(cropped_images[0], cmap='gray')
    ax3.axis('off')
    ax3.set_title('Cropped Image 1')

    ax4.imshow(cropped_images[1], cmap='gray')
    ax4.axis('off')
    ax4.set_title('Cropped Image 2')

    ax5.imshow(cropped_images[2], cmap='gray')
    ax5.axis('off')
    ax5.set_title('Cropped Image 3')

    plt.tight_layout()
    plt.show()





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
    print(fd, fd.shape)


# local binary patterns
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

# feature_extraction_hog('F001', 'T1')
