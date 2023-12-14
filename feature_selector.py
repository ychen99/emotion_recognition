import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import feature_extractor as fe
import pre_processing as pre
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def feature_standarlization(feature_arr):
    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(feature_arr)
    return standardized_features


feature_names = ['mean','max','min', 'median','variance', 'std']


def feature_combine(subject, task, idx):
    hog_features = fe.feature_extraction_hog(subject, task, idx)
    lbp_features = fe.feature_extraction_lbp(subject, task, idx)
    combined_feature = [sublist1 + sublist2 for sublist1, sublist2 in zip(hog_features, lbp_features)]
    return combined_feature


def pad_features(features, pad_value=0):
    max_length = max(len(sublist) for sublist in features)
    padded_features = [sublist + [pad_value] * (max_length - len(sublist)) for sublist in features]
    return padded_features


def generate_features():
    tasks = ['T1','T6']
    file_start = r"X:\BP4D+_v0.2\Physiology"
    subject_list = os.listdir(file_start)
    combined_features = []
    labels = []
    subject = 'F001'
    for task in tasks:
        methods = pre.PreProcessing(f'{subject}', f'{task}')
        image_names = methods.select_image_files()
        for i in range(len(image_names)):
            print(image_names[i])
            combined_features.extend(feature_combine(subject, task, i))
            labels.append([task] * 3)

    padded_features = pad_features(combined_features)
    print(len(padded_features), len(labels))

    df_data = pd.DataFrame(combined_features)
    df_label = pd.DataFrame(labels)
    df_data.to_csv('features_F001.csv', index=False)
    df_label.to_csv('label_F001.csv', index=False)


def pca(data):
    pca = PCA(n_components=0.95)
    scaled_features = feature_standarlization(data)
    reduced_features = pca.fit_transform(scaled_features)
    print(reduced_features.shape)
    return reduced_features


#generate_features()

df = pd.read_csv('features_F001.csv')
print(df.shape)
df_filled = df.fillna(0)
pca(df_filled)
