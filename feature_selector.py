import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn import svm
import matplotlib as mpl
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import RandomOverSampler

mpl.use('Qt5Agg')
pd.set_option('display.max_seq_items', None)


def feature_standarlization(feature_arr):
    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(feature_arr)
    return standardized_features


def pca(data):
    pca = PCA(n_components=0.95)
    scaled_features = feature_standarlization(data)
    reduced_features = pca.fit_transform(scaled_features)
    print(reduced_features.shape)
    return reduced_features


name = ['mean', 'median', 'max', 'min', 'var', 'std']
prefixes = ['hog1', 'lbp1', 'hog2', 'lbp2', 'hog3', 'lbp3']
feature_names = list(itertools.chain(*[[f"{prefix}_{n}" for n in name] for prefix in prefixes]))


def xgboost(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, random_state=42, test_size=0.2)
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    feature_importance = model.feature_importances_
    N = len(feature_importance)

    important_feature_indices = feature_importance.argsort()[-N:][::-1]
    X_train_important = X_train.iloc[:, important_feature_indices]
    X_test_important = X_test.iloc[:, important_feature_indices]

    important_features_model = RandomForestClassifier(random_state=42)  # svm.NuSVC(gamma="auto")
    important_features_model.fit(X_train_important, y_train)
    important_features_predictions = important_features_model.predict(X_test_important)

    classification_rep = classification_report(y_test, important_features_predictions)
    print(f'Classification Report:\n{classification_rep}')
    return feature_importance


def cross_validation(features, labels):
    _model = RandomForestClassifier(random_state=42)  # svm.NuSVC(gamma="auto")
    scores = cross_val_score(_model, features, labels, cv=5)
    print("Accuracy scores for each fold:", scores)
    print("Mean cross-validation score:", scores.mean())


def process_fea(fea_path, label_path, type):
    label_encoder = LabelEncoder()
    fea = pd.read_csv(fea_path)

    def drop_NaN(features):
        fea_cleaned = features.dropna(axis=1)
        return fea_cleaned

    def delete_ele(df):
        group_42_indices = range((42 - 1) * 4, 42 * 4)
        group_82_indices = range((82 - 1) * 4, 82 * 4)

        indices_to_drop = [group_42_indices[-2], group_42_indices[-1], group_82_indices[0]]

        df_dropped = df.drop(indices_to_drop)
        return df_dropped

    df_label = pd.read_csv(label_path)
    if type == 'phy':
        label = df_label.stack().tolist()
        numeric_labels = label_encoder.fit_transform(label)
    else:
        dropped_label = delete_ele(df_label)
        label = dropped_label.stack().tolist()
        numeric_labels = label_encoder.fit_transform(label)
    return drop_NaN(fea), numeric_labels


fea_img, labels_img = process_fea('features_list.csv', 'label.csv', type='img')
fea_phy, labels_phy = process_fea('features_phy.csv', 'labels_phy.csv', type='phy')


def compare_segments_and_find_last_indices(first_list, second_list):
    def split_into_segments(lst):
        segments = []
        current_segment = []
        for item in lst:
            if not current_segment or item == current_segment[0]:
                current_segment.append(item)
            else:
                segments.append(current_segment)
                current_segment = [item]
        segments.append(current_segment)  # Add the last segment
        return segments

    # Split both lists into segments
    segments_first_list = split_into_segments(first_list)
    segments_second_list = split_into_segments(second_list)

    differing_indices = []
    index_in_second_list = 0

    # Iterate through the segments, considering the shorter length of first_list segments
    for seg1, seg2 in zip(segments_first_list, segments_second_list):
        # If the second segment is longer, find the last differing element
        if len(seg2) > len(seg1):
            last_differing_index = index_in_second_list + len(seg2) - 1
            differing_indices.append(last_differing_index)

        # Update the index in the second list
        index_in_second_list += len(seg2)

    # Handle any remaining elements in the second list
    if index_in_second_list < len(second_list):

        differing_indices.append(len(second_list) - 1)

    return differing_indices


def down_sampling(X_img, y_img, X_phy, y_phy):
    indic_drop = compare_segments_and_find_last_indices(y_phy, y_img)
    print(len(y_phy), len(y_img), len(indic_drop))


down_sampling(fea_img, labels_img, fea_phy, labels_phy)


def feature_importance_plot():
    fea, labels = process_fea('features_phy.csv', 'labels_ohy.csv')
    feature_imp = xgboost(fea, labels)
    plt.figure(figsize=(16, 8))
    plt.barh(feature_names, feature_imp)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Names')
    plt.title('Feature Importance Plot')
    plt.show()

# feature_importance_plot()
