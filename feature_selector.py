import itertools
import os.path
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold
import xgboost as xgb
from sklearn import svm
import matplotlib as mpl
import seaborn as sns

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
    X_train, X_test = train_test_split(unique_labels, random_state=42, test_size=test_size)
    indices_train = df_labels[df_labels['F'].isin(X_train)].index
    indices_test = df_labels[df_labels['F'].isin(X_test)].index
    return df_features.iloc[indices_train], df_features.iloc[indices_test], df_labels['encoded'].iloc[indices_train], \
        df_labels['encoded'].iloc[indices_test]


def select_important_features(X_train, y_train):
    f = 'cover'
    model = xgb.XGBClassifier(importance_type=f)
    model.fit(X_train, y_train)
    feature_importances = model.feature_importances_
    length = len(X_train.columns)
    if 6 < length < 60:
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
    classifier = svm.SVC(kernel='rbf')  # RandomForestClassifier(random_state=42, max_depth=8, n_estimators=120)
    classifier.fit(X_train_important, y_train)

    # Predict and evaluate
    predictions = classifier.predict(X_test_important)
    return classification_report(y_test, predictions, output_dict=True)


def clean_feature_names(elements):
    return [element.replace('EDA_microsiemens_', '') for element in elements]


def feature_importance_plot(X_train, y_train):
    sorted_idx, feature_imp = select_important_features(X_train, y_train)
    feature_names = X_train.columns
    sorted_feature_importances = np.array(feature_imp)[sorted_idx]
    sorted_feature_names = np.array(feature_names)[sorted_idx]
    clean_sorted_feature = clean_feature_names(sorted_feature_names)
    plt.figure(figsize=(16, 8))

    colormap = matplotlib.cm.get_cmap('viridis')
    colors = colormap(np.linspace(0, 1, len(sorted_feature_importances)))
    plt.barh(clean_sorted_feature, sorted_feature_importances, color=colors)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Names')
    plt.title('Ranking of selected features')
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.show()


def plot_classification_report(report):
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.drop(columns=['support'])
    report_df = report_df.drop(index=['accuracy', 'macro avg', 'weighted avg'])
    # Heatmap of the classification report
    colormap = sns.color_palette("coolwarm", as_cmap=True)
    plt.figure(figsize=(10, 6))
    sns.heatmap(data=report_df, annot=True, cmap=colormap, fmt='.2f', cbar_kws={'label': 'Score'})
    plt.yticks(rotation=0)
    plt.title('Classification Report Heatmap')
    plt.xlabel('Metrics')
    plt.ylabel('Classes')
    plt.tight_layout()
    plt.show()


def result(features):
    label_path = '/Users/main/PycharmProjects/emotion_recognition/1sw/labels_final.csv'
    df_labels = process_labels(label_path)
    X_train, X_test, y_train, y_test = train_test_split(features, df_labels, test_size=0.2)

    important_feature_indices, _ = select_important_features(X_train, y_train)

    classification_rep = classify_with_selected_features(X_train, X_test, y_train, y_test, important_feature_indices)
    plot_classification_report(classification_rep)
    print(f'Classification Report:\n{classification_rep}')
    feature_importance_plot(X_train, y_train)


def single_features_generator(sensor_name):
    folder_path = '/Users/main/PycharmProjects/emotion_recognition/1sw'
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
    label_path = '/Users/main/PycharmProjects/emotion_recognition/1sw/labels_final.csv'
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
    return scores


def cross_validation_plot(scores):
    mean_score = np.mean(scores)
    std_dev = np.std(scores)

    # Indices for x-axis
    x_pos = [1]
    # Plot
    plt.figure(figsize=(8, 6))
    plt.boxplot(scores, positions=[1], widths=0.6, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='blue'),
                whiskerprops=dict(color='blue'),
                capprops=dict(color='blue'),
                medianprops=dict(color='red')
                )
    plt.scatter([1] * len(scores), scores, color='blue', label='CV Scores')
    plt.title('10-Fold Cross-Validation Scores')
    plt.ylabel('Score')
    plt.xticks([1], ['Random Forest'])
    plt.ylim(0.9 * min(scores), 1.1 * max(scores))
    plt.grid(axis='y')

    # Show plot
    plt.legend()
    plt.show()


def cross_cl_mutli_plot(df_melted):
    # plt.grid(True)  # Just turns on the grid

    plt.figure(figsize=(10, 6))  # Optional: Adjust the figure size
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')  # Customizes the major grid
    plt.minorticks_on()
    sns.boxplot(x='Dataset', y='score', data=df_melted, palette='pastel')

    # Adding titles and labels
    plt.title('Comparison of 10-Fold Cross-Validation Results')
    plt.xlabel('Modality', fontsize=12)
    plt.ylabel('Mean accuracy', fontsize=12)

    plt.show()


phy = single_features_generator(sensor_name='phy')
bp, res, hr, eda = phy_select(phy)
ir = single_features_generator(sensor_name='ir')
img = single_features_generator(sensor_name='img')
au = single_features_generator(sensor_name='au')


features_combines = combine_datasets(ir, img,eda)
# cv_scores = cross_validation(features_combines)
result(features_combines)


def prepare_cv_data_for_plotting_with_names(cv_names, *cv_results):
    if len(cv_names) != len(cv_results):
        raise ValueError("The number of dataset names must match the number of CV result sets.")

    data = {cv_names[i]: results for i, results in enumerate(cv_results)}

    df = pd.DataFrame(data)
    df_melted = df.melt(var_name='Dataset', value_name='score')

    return df_melted


def plot_cv():
    cv_1 = cross_validation(combine_datasets(ir, img, eda))
    cv_2 = cross_validation(combine_datasets(ir, img, au))
    cv_3 = cross_validation(combine_datasets(ir, img,eda, au))
    #cv_4 = cross_validation(combine_datasets(ir, hr))
    #cv_5 = cross_validation(combine_datasets(ir, img))
    #cv_6 = cross_validation(combine_datasets(ir, au))
    # ['IR+Image+EDA','IR+Image+AUs','IR+Image+EDA+AUs'] ['IR+Image+Blood pressure','IR+EDA','IR+Respiration', 'IR+Heart rate', 'IR+Image', 'IR+AUs']
    data_names = ['IR+Image+EDA','IR+Image+AUs','IR+Image+EDA+AUs']
    cross_cl_mutli_plot(prepare_cv_data_for_plotting_with_names(data_names, cv_1, cv_2, cv_3))


# plot_cv()
