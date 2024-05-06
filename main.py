import os
from sklearn.model_selection import train_test_split, KFold
from feature_selector import load_and_preprocess_feature, combine_datasets, process_labels, select_important_features, \
    classify_with_selected_features, plot_classification_report, feature_importance_plot, single_features_generator, \
    phy_select


def main(data_folder, feature_combined):
    # Load and preprocess data according to the feature_combination_type

    # Process labels
    label_path = os.path.join(data_folder, 'labels_final.csv')
    df_labels = process_labels(label_path)

    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(features_combined, df_labels, test_size=0.2)
    # Feature selection and classification
    important_feature_indices, _ = select_important_features(X_train, y_train)
    classification_report_data = classify_with_selected_features(X_train, X_test, y_train, y_test,
                                                                 important_feature_indices)

    # Plot classification report
    plot_classification_report(classification_report_data)
    print(f'Classification Report:\n{classification_report_data}')
    feature_importance_plot(X_train, y_train)


if __name__ == "__main__":
    # Define paths and feature combination type
    data_folder = '/Users/main/PycharmProjects/emotion_recognition/1sw'

    ir = load_and_preprocess_feature('features_ir_final.csv', data_folder, 'ir')
    img = load_and_preprocess_feature('features_img_final.csv', data_folder, 'img')
    au = load_and_preprocess_feature('features_au_final.csv', data_folder, 'au')
    phy = load_and_preprocess_feature('features_phy_final.csv', data_folder, 'phy')
    bp, res, hr, eda = phy_select(phy)

    feature_combination_type = (ir, au)  # Set to 'ir', 'img', or 'au'

    features_combined = combine_datasets(*feature_combination_type)

    main(data_folder, feature_combination_type)
