import json
import pickle
from copy import copy

import numpy as np
import pandas as pd

# ! python -m spacy download ru_core_news_lg -qq
import spacy

from pymorphy3 import MorphAnalyzer


# чтение доразмеченной выборки
train = pd.read_parquet('data/train.parquet')

# чтение категориального древа с внесёнными изменениями
cat_tree = pd.read_csv('data/category_tree.csv')

# категории, разметка которых интересует:
#   - нет дочерних категорий;
#   - число размеченных образцов < 20;
target_ids = np.setdiff1d(cat_tree['cat_id'], cat_tree['parent_id'])
target_cats = cat_tree[cat_tree['cat_id'].isin(target_ids)]['cat_name'].unique()

temp_df = cat_tree[['cat_id', 'cat_name']].merge(
    train['cat_id'].value_counts(),
    how='left',
    left_on='cat_id',
    right_index=True,
)

temp_df['count'] = temp_df['count'].fillna(0)

temp_df = temp_df.sort_values(
    by=['cat_name', 'count'],
    ascending=[True, False],
)

temp_df = temp_df.drop_duplicates('cat_name')

target_cats = np.intersect1d(
    temp_df[temp_df['count'] < 20]['cat_name'].unique(),
    target_cats,
)

# чтение внешних данных
# ссылка: https://www.kaggle.com/datasets/dsitd0/ozon-tech/data
ext_data = pd.read_parquet('external_data/text_and_bert.parquet')
attributes = pd.read_parquet('external_data/attributes.parquet')
# ссылка: https://www.kaggle.com/datasets/vslav27/ozon-dataset
ext_data2 = pd.read_parquet('external_data/train_data.parquet')

# объединение внешних данных
ext_data = ext_data[['variantid', 'name']]
attributes = attributes[['variantid', 'categories']]
ext_data = ext_data.merge(attributes, on='variantid')
ext_data2 = ext_data2[['variantid', 'name', 'categories']]
ext_data = pd.concat([ext_data, ext_data2])

# "размножение" выборки за счёт присвоения категорий всех уровней
ext_data['categories'] = ext_data['categories'].apply(
    lambda x: [v for k, v in json.loads(x).items() if k in ['2', '3', '4']]
)

ext_data = ext_data.explode('categories')

# создание словаря для замены оригинальных названий категорий
cat2cat = {
    k: v for k, v in zip(
        ext_data['categories'].unique(), ext_data['categories'].unique()
    )
}

# категории, не имеющие дочерние
terminal_cats = ext_data.groupby(ext_data.index)['categories'].tail(1).unique()


morph = MorphAnalyzer()
nlp = spacy.load("ru_core_news_lg")

# коррекция меток категорий для совместимости с метками из размеченной выборки
old2new = {}

for old_cat in terminal_cats:
    new_cat = copy(old_cat)

    doc = nlp(new_cat)
    tokens = [str(t) for t in doc]
    pos = [t.pos_ for t in doc]
    deps = [t.dep_ for t in doc]
    heads = [str(t.head) for t in doc]

    try:
        subj_idx = deps.index('nsubj')
    except ValueError:
        subj_idx = deps.index('ROOT')

    old_subj = tokens[subj_idx]

    try:
        new_subj = morph.parse(old_subj)[0].inflect({'plur', 'nomn'}).word
        new_cat = new_cat.replace(old_subj, new_subj)
    except AttributeError:
        pass

    for i in range(len(tokens)):
        if ((pos[i] in ['ADJ', 'VERB']) and
           ((heads[i] == old_subj) or (deps[i] == 'ROOT'))):
            old_prop = tokens[i]

            try:
                new_prop = morph.parse(old_prop)[0].inflect({'plur', 'nomn'}).word
                new_cat = new_cat.replace(old_prop, new_prop)
            except AttributeError:
                pass

    old2new[old_cat] = [old_cat, new_cat.capitalize()]

# применение скорректированных оглавлений категорий
cat2cat.update(old2new)
ext_data['categories'] = ext_data['categories'].map(cat2cat)
ext_data = ext_data.explode('categories')

# сохранение товаров, относящихся с целевым категориям
# присвоение товару наиболее частной из возможных категорий
ext_data = ext_data[ext_data['categories'].isin(target_cats)]
ext_data = ext_data.groupby('variantid').tail(1)

# присвоение товару id категории
with open('cat_id.pkl', mode='rb') as f:
    cat2id = pickle.load(f)

ext_data['cat_id'] = ext_data['categories'].map(cat2id)
ext_data = ext_data.rename(columns={'name': 'source_name'})
ext_data = ext_data.drop(columns=['categories', 'variantid'])
ext_data = ext_data.drop_duplicates()

# отбор 20-ти случайных образцов для каждого из целевых классов
ext_data = (ext_data
            .groupby('cat_id').sample(frac=1, random_state=42)
            .groupby('cat_id').head(20))

# расширение обучающей выборки
train = pd.concat([train, ext_data])
train = train.drop_duplicates('source_name')
train.to_parquet('data/train.parquet', index=False)
