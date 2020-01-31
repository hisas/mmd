import pandas as pd
import random

random.seed(0)


def extract_gaze(df):
    df_ann = pd.read_csv('data/data_new/train_annotations.txt', delimiter=',', header=None)
    gaze = {}
    for _, r in df_ann.iterrows():
        gaze[r[1]] = [r[8], r[9]]

    data = []
    for _, r in df.iterrows():
        data.append([r[0], r[1], gaze[r[2]][0], gaze[r[2]][1], r[3]])

    return pd.DataFrame(data)

def remove_x(df, task):
    ng_words = ['なし', 'なi', '罰', 'ばつ', 'バツ', 'batu ', 'b', 'u', 'v', 'z']
    if task == 'action':
        ng_words.append('はい')
    df = df[~df[task].isin(ng_words)]

    df = df[~df[task].str.contains('×')]
    df = df[~df[task].str.contains('x')]
    df = df[~df[task].str.contains('✖')]
    df = df[~df[task].str.contains('ｘ')]
    df = df[~df[task].str.contains('Ｘ')]
    df = df[~df[task].str.contains('ⅹ')]
    df = df[~df[task].str.contains('✕')]
    df = df[~df[task].str.contains('X')]

    if task == 'action':
        df = df[~df[task].str.contains('http')]
        df = df[~df[task].str.contains('Heydouga')]
        df = df[~df[task].str.contains('楽天Beauty')]
        df = df[~df[task].str.contains('何もしな')]
        df = df[~df[task].str.contains('なにもしな')]
        df = df[~df[task].str.contains('特にな')]
        df = df[~df[task].str.contains('とくにな')]

    return df

def filter_response_df(path):
    df = pd.read_csv(path, delimiter='\t', usecols=['text1', 'image', 'gf_id', 'response'])

    df = remove_x(df, 'response')
    df['text1'] = df['text1'].str[56:-52].str.strip()
    df = df[~df['text1'].isin(['""""""', '##', '$$$'])]
    df['response'] = df['response'].str.strip()

    train_df, val_df, test_df = df[:len(df)-24000], df[len(df)-24000:len(df)-12000], df[len(df)-12000:len(df)]
    train_df, val_df, test_df = extract_gaze(train_df), extract_gaze(val_df), extract_gaze(test_df)

    return train_df, val_df, test_df

def export_response_df(df, set):
    context_response = []
    for _, r in df.iterrows():
        context_response.append([r[0], r[4]])

    data = [['Context', 'Image', 'GazeX', 'GazeY', 'Response', 'Label']]
    for _, r in df.iterrows():
        data.append([r[0], r[1], r[2], r[3], r[4], 1])

        while True:
            t = random.choice(context_response)
            if t[0] != r[0]:
                data.append([r[0], r[1], r[2], r[3], t[1], 0])
                break

    pd.DataFrame(data).to_csv('data/response_' + set + '.csv', index=False, header=False)

def export_response_test_df(df):
    context_response = []
    for _, r in df.iterrows():
        context_response.append([r[0], r[4]])

    data = [['Context', 'Image', 'GazeX', 'GazeY', 'Response', 'Label']]
    for _, r in df.iterrows():
        data.append([r[0], r[1], r[2], r[3], r[4], 1])

        while True:
            t = random.choice(context_response)
            if t[0] != r[0]:
                data.append([r[0], r[1], r[2], r[3], t[1], 0])
                break

    pd.DataFrame(data).to_csv('data/response_test_1.csv', index=False, header=False)

    data = [['Context', 'Image', 'GazeX', 'GazeY', 'Ground Truth Response', 'Distractor_0', 'Distractor_1', 'Distractor_2', 'Distractor_3', 'Distractor_4', 'Distractor_5', 'Distractor_6', 'Distractor_7', 'Distractor_8']]
    for _, r in df.iterrows():
        while True:
            ts = random.sample(context_response, 9)
            flag = True
            ds = []
            for t in ts:
                if t[0] != r[0]:
                    ds.append(t[1])
                else:
                    flag = False

            if flag == True:
                data.append(sum([[r[0]], [r[1]], [r[2]], [r[3]], [r[4]], ds], []))
                break
    pd.DataFrame(data).to_csv('data/response_test_10.csv', index=False, header=False)

def filter_action_df(path):
    df = pd.read_csv(path, delimiter='\t', usecols=['text1', 'image', 'gf_id', 'action'])

    df = remove_x(df, 'action')
    df['text1'] = df['text1'].str[56:-52].str.strip()
    df = df[~df['text1'].isin(['""""""', '##', '$$$'])]
    df['action'] = df['action'].str.strip()

    train_df, val_df, test_df = df[:len(df)-6000], df[len(df)-6000:len(df)-3000], df[len(df)-3000:len(df)]
    train_df, val_df, test_df = extract_gaze(train_df), extract_gaze(val_df), extract_gaze(test_df)

    return train_df, val_df, test_df

def export_action_df(df, set):
    context_action = []
    for _, r in df.iterrows():
        context_action.append([r[0], r[4]])

    data = [['Context', 'Image', 'GazeX', 'GazeY', 'Action', 'Label']]
    for _, r in df.iterrows():
        data.append([r[0], r[1], r[2], r[3], r[4], 1])

        while True:
            t = random.choice(context_action)
            if t[0] != r[0]:
                data.append([r[0], r[1], r[2], r[3], t[1], 0])
                break

    pd.DataFrame(data).to_csv('data/action_' + set + '.csv', index=False, header=False)

def export_action_test_df(df):
    context_action = []
    for _, r in df.iterrows():
        context_action.append([r[0], r[4]])

    data = [['Context', 'Image', 'GazeX', 'GazeY', 'Action', 'Label']]
    for _, r in df.iterrows():
        data.append([r[0], r[1], r[2], r[3], r[4], 1])

        while True:
            t = random.choice(context_action)
            if t[0] != r[0]:
                data.append([r[0], r[1], r[2], r[3], t[1], 0])
                break

    pd.DataFrame(data).to_csv('data/action_test_1.csv', index=False, header=False)

    data = [['Context', 'Image', 'GazeX', 'GazeY', 'Ground Truth Action', 'Distractor_0', 'Distractor_1', 'Distractor_2', 'Distractor_3', 'Distractor_4', 'Distractor_5', 'Distractor_6', 'Distractor_7', 'Distractor_8']]
    for _, r in df.iterrows():
        while True:
            ts = random.sample(context_action, 9)
            flag = True
            ds = []
            for t in ts:
                if t[0] != r[0]:
                    ds.append(t[1])
                else:
                    flag = False

            if flag == True:
                data.append(sum([[r[0]], [r[1]], [r[2]], [r[3]], [r[4]], ds], []))
                break
    pd.DataFrame(data).to_csv('data/action_test_10.csv', index=False, header=False)


train_response_df, val_response_df, test_response_df = filter_response_df('data/jparvsu-response.tsv')
export_response_df(train_response_df, 'train')
export_response_df(val_response_df, 'val')
export_response_test_df(test_response_df)

train_action_df, val_action_df, test_action_df = filter_action_df('data/jparvsu-response.tsv')
export_action_df(train_action_df, 'train')
export_action_df(val_action_df, 'val')
export_action_test_df(test_action_df)
