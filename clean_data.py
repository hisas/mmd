import os
import pandas as pd

def remove_x(df, task):
    x_mark = ['×', 'x', '✖', 'ｘ', 'Ｘ', 'ⅹ', '✕', 'X']
    for mark in x_mark:
        df = df[~df[task].str.contains(mark)]

    if task == 'action':
        words = ['なし', 'なi', '罰', 'ばつ', 'バツ', 'batu ', 'はい', 'b', 'u', 'v', 'z', 'してみる']
        df = df[~df[task].isin(words)]

        words = ['返答する', '返事', '返事する', '返事をする', '返事した']
        df = df[~df[task].isin(words)]

        words = ['http', 'Heydouga', '楽天Beauty', '何もしな', 'なにもしな', '特にな', 'とくにな']
        for word in words:
            df = df[~df[task].str.contains(word)]

    return df

def export_cleaned_response(path):
    df = pd.read_csv(path, delimiter='\t', usecols=['text1', 'image', 'gf_id', 'response'])

    df = remove_x(df, 'response')
    df['text1'] = df['text1'].str[56:-52].str.strip()
    df = df[~df['text1'].isin(['""""""', '##', '$$$'])]
    df['response'] = df['response'].str.strip()

    df.to_csv('data/cleaned_jparvsu-response.tsv', sep='\t', index=False)

def export_cleaned_action(path):
    df = pd.read_csv(path, delimiter='\t', usecols=['text1', 'image', 'gf_id', 'action'])

    df = remove_x(df, 'action')
    df['text1'] = df['text1'].str[56:-52].str.strip()
    df = df[~df['text1'].isin(['""""""', '##', '$$$'])]
    df['action'] = df['action'].str.strip()

    df.to_csv('data/cleaned_jparvsu-action.tsv', sep='\t', index=False)

os.mkdir('data')
export_cleaned_response('original_data/jparvsu-response.tsv')
export_cleaned_action('original_data/jparvsu-response.tsv')
