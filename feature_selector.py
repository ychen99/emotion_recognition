import itertools
import os.path
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold
import xgboost as xgb
from sklearn import svm
import matplotlib as mpl
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import RandomOverSampler
from matplotlib import cm

mpl.use('Qt5Agg')
pd.set_option('display.max_seq_items', None)


def feature_names_add(image):
    if image == 'ir':
        name = ['mean', 'var', 'skew']
        prefixes = ['red1', 'green1', 'blue1', 'red2', 'green2', 'blue2', 'red3', 'green3', 'blue3']
    elif image == "img":
        name = ['mean', 'median', 'max', 'min', 'var']
        prefixes = ['hog1', 'lbp1', 'hog2', 'lbp2', 'hog3', 'lbp3']

    feature_names = list(itertools.chain(*[[f"{prefix}_{n}" for n in name] for prefix in prefixes]))

    return feature_names


def load_and_preprocess_feature(file_name, folder_path, feature_type):
    feature_path = os.path.join(folder_path, file_name)
    df = pd.read_csv(feature_path).fillna(0)
    if feature_type in ['ir', 'img']:
        df.columns = feature_names_add(image=feature_type)
    elif feature_type == 'au':
        df.columns = ["AU6", "AU10", "AU12", "AU14", "AU17"]
    return df


def process_labels(label_path):
    df_labels = pd.read_csv(label_path)
    label_encoder = LabelEncoder()
    df_labels['encoded'] = label_encoder.fit_transform(df_labels.iloc[:, 1].values.ravel())
    return df_labels['encoded']


def feature_combination(fea1, fea2):
    fea_concentrate = pd.concat([fea1, fea2], axis=1)
    return fea_concentrate


def train_test_split_by_label(df_features, df_labels, test_size=0.2):
    unique_labels = df_labels['F'].drop_duplicates()
    X_train, X_test = train_test_split(unique_labels, random_state=24, test_size=test_size)
    indices_train = df_labels[df_labels['F'].isin(X_train)].index
    indices_test = df_labels[df_labels['F'].isin(X_test)].index
    return df_features.iloc[indices_train], df_features.iloc[indices_test], df_labels['encoded'].iloc[indices_train], \
           df_labels['encoded'].iloc[indices_test]


def select_important_features(X_train, y_train):
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    feature_importances = model.feature_importances_
    length = len(X_train.columns)
    if 6 < length < 50:
        num_features = int(0.5 * len(feature_importances))
    elif length == 5:
        num_features = len(feature_importances)
    else:
        num_features = int(0.1 * len(feature_importances))
    important_feature_indices = np.argsort(feature_importances)[-num_features:][::-1]
    return important_feature_indices, feature_importances


def classify_with_selected_features(X_train, X_test, y_train, y_test, important_feature_indices):
    # Select important features for training and testing datasets
    X_train_important = X_train.iloc[:, important_feature_indices]
    X_test_important = X_test.iloc[:, important_feature_indices]

    # Train classifier using the selected features
    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(X_train_important, y_train)

    # Predict and evaluate
    predictions = classifier.predict(X_test_important)
    return classification_report(y_test, predictions)


def feature_importance_plot(X_train, y_train):
    sorted_idx, feature_imp = select_important_features(X_train, y_train)
    feature_names = X_train.columns
    sorted_feature_importances = np.array(feature_imp)[sorted_idx]
    sorted_feature_names = np.array(feature_names)[sorted_idx]

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


def result(features):
    label_path = r'C:\Users\YChen\PycharmProjects\pythonProject\2sw\labels_final.csv'
    df_labels = process_labels(label_path)
    X_train, X_test, y_train, y_test = train_test_split(features, df_labels, test_size=0.2)

    important_feature_indices, _ = select_important_features(X_train, y_train)

    classification_rep = classify_with_selected_features(X_train, X_test, y_train, y_test, important_feature_indices)

    print(f'Classification Report:\n{classification_rep}')
    # feature_importance_plot(X_train, y_train)


def single_features_generator(sensor_name):
    folder_path = r'C:\Users\YChen\PycharmProjects\pythonProject\2sw'
    sensor = sensor_name
    fea_file_name = 'features_' + sensor + '_final.csv'
    features = load_and_preprocess_feature(fea_file_name, folder_path, sensor)
    return features


def combine_datasets(*datasets):
    combined_dataset = None

    for dataset in datasets:
        if combined_dataset is None:
            combined_dataset = dataset
        else:
            combined_dataset = feature_combination(combined_dataset, dataset)

    return combined_dataset


def phy_select(fea):
    fea_bp = fea.filter(regex='BP Dia_mmHg|BP_mmHg')
    fea_res = fea.filter(regex='Resp_Volts| Respiration Rate_BPM')
    fea_hr = fea.filter(regex='Pulse Rate_BPM')
    fea_eda = fea.filter(regex='EDA_microsiemens')
    return fea_bp, fea_res, fea_hr, fea_eda





def cross_validation(features):
    label_path = r'C:\Users\YChen\PycharmProjects\pythonProject\2sw\labels_final.csv'
    labels = process_labels(label_path)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = []

    for train_index, test_index in kf.split(features):
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

        important_feature_indices, _ = select_important_features(X_train, y_train)

        X_train_important = X_train.iloc[:, important_feature_indices]
        X_test_important = X_test.iloc[:, important_feature_indices]

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train_important, y_train)
        predictions = model.predict(X_test_important)
        score = accuracy_score(y_test, predictions)
        scores.append(score)

    print("Accuracy scores for each fold:", scores)
    print("Mean cross-validation score:", np.mean(scores))


phy = single_features_generator(sensor_name='phy')
bp, res, hr, eda = phy_select(phy)
ir = single_features_generator(sensor_name='ir')
img = single_features_generator(sensor_name='img')
au = single_features_generator(sensor_name='au')
features_combines = combine_datasets(res,  ir, img, eda)
cross_validation(features_combines)
# result(features_combines)