import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import feature_extractor as fe
import pre_processing as pre
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import xgboost as xgb


def feature_standarlization(feature_arr):
    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(feature_arr)
    return standardized_features


def compute_statistics(feature):
    return [np.mean(feature), np.median(feature), max(feature),
            min(feature), np.var(feature), np.std(feature)]


def cal_stat_fea(features):
    return [compute_statistics(f) for f in features]


def feature_extraction(subject, task, idx):
    return fe.feature_extraction_hog(subject, task, idx), fe.feature_extraction_lbp(subject, task, idx)


def feature_combine(subject, task, idx):
    hog_features, lbp_features = feature_extraction(subject, task, idx)
    hog_stats = cal_stat_fea(hog_features)
    lbp_stats = cal_stat_fea(lbp_features)
    combined_feature = []
    for hog_stat, lbp_stat in zip(hog_stats, lbp_stats):
        combined_feature.append(hog_stat + lbp_stat)
    combined_features = list(itertools.chain(*combined_feature))
    return combined_features


def pad_features(features, pad_value=0):
    max_length = max(len(sublist) for sublist in features)
    padded_features = [sublist + [pad_value] * (max_length - len(sublist)) for sublist in features]
    return padded_features


def average_every_three_rows(multi_list):
    averages = []
    num_rows = len(multi_list)

    for i in range(0, num_rows, 50):
        chunk = multi_list[i:i + 50]

        if chunk:
            chunk_average = [sum(col) / len(col) for col in zip(*chunk)]
            averages.append(chunk_average)

    return averages


def generate_features():
    tasks = ['T1', 'T6', 'T7', 'T8']
    file_start = r"X:\BP4D+_v0.2\Physiology"
    subject_list = os.listdir(file_start)
    labels = []
    features_list = []
    for subject in subject_list:
        for task in tasks:
            methods = pre.PreProcessing(f'{subject}', f'{task}')
            image_names = methods.select_image_files()
            tmp = []
            for i in range(len(image_names)):
                print(image_names[i])
                try:
                    tmp.append(feature_combine(subject, task, i))
                except ValueError as e:
                    continue
                except IndexError as e:
                    continue
            ave = average_every_three_rows(tmp)
            features_list.extend(ave)
            labels.append([task] * len(ave))

    labels_list = list(itertools.chain(*labels))
    print(len(features_list), len(features_list[0]), len(labels_list))

    '''
    padded_features = pad_features(combined_features)
    print(len(padded_features), len(labels))
    '''
    df_data = pd.DataFrame(features_list)
    df_label = pd.DataFrame(labels)
    df_data.to_csv('features_list.csv', index=False)
    df_label.to_csv('label.csv', index=False)


generate_features()


def pca(data):
    pca = PCA(n_components=0.95)
    scaled_features = feature_standarlization(data)
    reduced_features = pca.fit_transform(scaled_features)
    print(reduced_features.shape)
    return reduced_features


def XGBoost(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    importance = model.feature_importances_
    print(importance)



