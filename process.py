import pandas as pd


def find_discrepancies_between_frames(df1, df2):
    df1_counts = df1.groupby(['F', 'T']).size().reset_index(name='count1')
    df2_counts = df2.groupby(['F', 'T']).size().reset_index(name='count2')

    merged_counts = pd.merge(df1_counts, df2_counts, on=['F', 'T'], how='outer').fillna(0)

    merged_counts['diff'] = merged_counts['count2'] - merged_counts['count1']

    extra_in_df2 = merged_counts[merged_counts['diff'] > 0]

    extra_in_df1 = merged_counts[merged_counts['diff'] > 0]

    extra_rows_df1 = []
    for index, row in extra_in_df1.iterrows():
        extra_rows = df1[(df1['F'] == row['F']) & (df1['T'] == row['T'])].index.tolist()[int(row['count1']):]
        extra_rows_df1.extend(extra_rows)

    extra_rows_df2 = []
    for index, row in extra_in_df2.iterrows():
        extra_rows = df2[(df2['F'] == row['F']) & (df2['T'] == row['T'])].index.tolist()[
                     int(row['count2'] - row['diff']):]
        extra_rows_df2.extend(extra_rows)

    return extra_rows_df2


def file_syn(first_label_path, second_label_path, second_feature_path):
    first_label = pd.read_csv(first_label_path)
    second_label = pd.read_csv(second_label_path)
    first_label.columns = ["F", 'T']
    second_label.columns = ["F", 'T']

    second_fea = pd.read_csv(second_feature_path)

    print(first_label.shape, second_label.shape, second_fea.shape)
    diff_first_second = find_discrepancies_between_frames(first_label, second_label)
    print(len(diff_first_second))
    label_new = second_label.drop(diff_first_second)
    second_fea_new = second_fea.drop(diff_first_second)

    return label_new, second_fea_new

label1 = r'C:\Users\YChen\PycharmProjects\pythonProject\2sw\labels_final.csv'
label2 = r'C:\Users\YChen\PycharmProjects\pythonProject\2sw\labels_img.csv'

fea1 =  r'C:\Users\YChen\PycharmProjects\pythonProject\2sw\features_au_final.csv'
fea2 = r'C:\Users\YChen\PycharmProjects\pythonProject\2sw\features_img.csv'

label, fea = file_syn(label2 ,label1, fea1)
print(label.shape)
#label.to_csv(r'C:\Users\YChen\PycharmProjects\pythonProject\2sw\labels_final0.csv', index=False)