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

    list_indices = [get_index_from_position(row, col, df_img) for row, col in flattened_positions]

    return flattened_positions, list_indices


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

    pos1, diff_indics_1 = calculate_indices_of_additional_elements(df_label_img, df_label_phy)
    print(len(diff_indics_1))

    label_phy = df_label_phy.stack().tolist()
    label_img = df_label_img.stack().tolist()

    print(len(label_phy), len(label_img))
    numeric_labels_phy = label_encoder.fit_transform(label_phy)
    numeric_labels_img = label_encoder.fit_transform(label_img)

    return numeric_labels_phy


labels_img = process_fea('features_phy.csv', 'labels_phy.csv', 'features_image.csv', 'labels_img.csv')


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
