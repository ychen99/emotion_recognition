import itertools

import matplotlib
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
from matplotlib import cm

mpl.use('Qt5Agg')
pd.set_option('display.max_seq_items', None)


def get_index_from_position(row, col, df):
    zero_based_row = row - 1
    count = 0
    # Count non-NaN values up to the given row
    for r in range(zero_based_row):
        count += df.iloc[r, :].notna().sum()
    # Count non-NaN values in the given row up to the column
    count += df.iloc[zero_based_row, :col].notna().sum()
    return count


def delete_elements(fea_path, label_path):
    df_fea = pd.read_csv(fea_path)
    df_label = pd.read_csv(label_path)
    group_42_indices = range((42 - 1) * 4, 42 * 4)
    group_82_indices = range((82 - 1) * 4, 82 * 4)

    indices_to_drop = [group_42_indices[-2], group_42_indices[-1], group_82_indices[0]]
    non_nan_count = [np.where(df_label.iloc[i].notna())[0].tolist() for i in indices_to_drop]
    rows = [[indices_to_drop[i]] * len(non_nan_count[i]) for i in range(3)]

    indics_ele = []
    for row, col in zip(rows, non_nan_count):
        for i, j in zip(row, col):
            indics_ele.append(get_index_from_position(i, j, df_label))
    df_fea_dropped = df_fea.drop(indics_ele)
    print(indics_ele, df_fea_dropped.shape)
    # df_fea_dropped.to_csv('features_image.csv', index=False)


# delete_elements('features_img.csv', 'label.csv')


def calculate_indices_of_additional_elements(df_img, df_phy):
    def find_additional_non_nan_elements(row_img, row_phy):
        return [index for index, (item_img, item_phy) in enumerate(zip(row_img, row_phy)) if
                pd.notnull(item_img) and pd.isnull(item_phy)]

    additional_positions = [find_additional_non_nan_elements(row_img, row_phy) for row_img, row_phy in
                            zip(df_img.values, df_phy.values)]

    flattened_positions = [(row_num, pos) for row_num, positions in enumerate(additional_positions, start=1) for pos in
                           positions]
    # flattened_positions.append((27, 13))
    list_indices = [get_index_from_position(row, col, df_img) for row, col in flattened_positions]

    return flattened_positions, list_indices


def delete_redundant(df, label):
    indics = get_index_from_position(27, 13, label)
    df_dropped = df.drop(indics)
    return df_dropped


def process_fea(fea_phy_path, label_phy_path, fea_img_path, label_img_path):
    label_encoder = LabelEncoder()
    fea_phy = pd.read_csv(fea_phy_path)
    fea_img = pd.read_csv(fea_img_path)
    df_label_phy = pd.read_csv(label_phy_path)
    df_label_img = pd.read_csv(label_img_path)

    def drop_NaN(features):
        fea_cleaned = features.dropna(axis=1)
        return fea_cleaned

    pos0, diff_indics_0 = calculate_indices_of_additional_elements(df_label_phy, df_label_img)

    for row, col in pos0:
        df_label_phy.iat[row - 1, col] = np.nan

    fea_phy_dropped = fea_phy.drop(diff_indics_0)

    pos1, diff_indics_1 = calculate_indices_of_additional_elements(df_label_img, df_label_phy)

    for row, col in pos0:
        df_label_img.iat[row - 1, col] = np.nan

    fea_img_dropped = fea_img.drop(diff_indics_1)

    label_phy = df_label_phy.stack().tolist()
    numeric_labels_phy = label_encoder.fit_transform(label_phy)

    new_fea_img = delete_redundant(drop_NaN(fea_img_dropped), df_label_img)
    new_fea_phy = drop_NaN(fea_phy_dropped)

    new_fea_phy.to_csv('fea_phy_final.csv', index=False)
    new_fea_img.to_csv('fea_img_final.csv', index=False)
    numeric_labels_df = pd.DataFrame(numeric_labels_phy)
    numeric_labels_df.to_csv('labels_final.csv', index=False)


# process_fea('features_phy.csv', 'labels_phy.csv', 'features_image.csv', 'labels_img.csv')

name = ['mean', 'median', 'max', 'min', 'var', 'std']
prefixes = ['hog1', 'lbp1', 'hog2', 'lbp2', 'hog3', 'lbp3']
feature_img_names = list(itertools.chain(*[[f"{prefix}_{n}" for n in name] for prefix in prefixes]))


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


def xgboost(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, random_state=42, test_size=0.2)
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    feature_importance = model.feature_importances_
    N = 40#int(0.1 * len(feature_importance))
    important_feature_indices = feature_importance.argsort()[-N:][::-1]
    X_train_important = X_train.iloc[:, important_feature_indices]
    X_test_important = X_test.iloc[:, important_feature_indices]

    important_features_model = RandomForestClassifier(random_state=42)  # svm.NuSVC(gamma="auto")
    important_features_model.fit(X_train_important, y_train)
    important_features_predictions = important_features_model.predict(X_test_important)

    classification_rep = classification_report(y_test, important_features_predictions)
    print(f'Classification Report:\n{classification_rep}')
    return feature_importance, important_feature_indices


def cross_validation(features, labels):
    _model = RandomForestClassifier(random_state=42)  # svm.NuSVC(gamma="auto")
    scores = cross_val_score(_model, features, labels, cv=5)
    print("Accuracy scores for each fold:", scores)
    print("Mean cross-validation score:", scores.mean())


def feature_importance_plot(fea, labels, feature_names):
    feature_imp, sorted_idx = xgboost(fea, labels)
    # sorted_idx = np.argsort(feature_imp)[::-1]  # Get the indices that would sort the array
    sorted_feature_importances = np.array(feature_imp)[sorted_idx]
    sorted_feature_names = np.array(feature_names)[sorted_idx]
    print(sorted_feature_names)

    plt.figure(figsize=(16, 8))

    colormap = matplotlib.colormaps.get_cmap('viridis')
    colors = colormap(np.linspace(0, 1, len(sorted_feature_importances)))
    plt.barh(sorted_feature_names, sorted_feature_importances, color=colors)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Names')
    plt.title('Feature Importance Plot')
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.show()


fea_phy = pd.read_csv('fea_phy_final.csv')
fea_img = pd.read_csv('fea_img_final.csv')
labels = pd.read_csv('labels_final.csv')

fea_bp = fea_phy.filter(regex='BP Dia_mmHg|BP_mmHg')
fea_res = fea_phy.filter(regex='Resp_Volts| Respiration Rate_BPM')
fea_hr = fea_phy.filter(regex='Pulse Rate_BPM')
fea_eda = fea_phy.filter(regex='EDA_microsiemens')

fea_concentrate = pd.concat([fea_res, fea_img, fea_bp], axis=1)


def result(df):
    feature_importance_plot(df, labels, df.columns)


result(fea_hr)
