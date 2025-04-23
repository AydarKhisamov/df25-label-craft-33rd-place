import pickle

import pandas as pd
from sentence_transformers import SentenceTransformer, util


# чтение данных
train = pd.read_parquet('data/train.parquet')
unlabeled_train = pd.read_parquet('data/unlabeled_train.parquet')
cat_tree = pd.read_csv('data/category_tree.csv')

# категории, разметка которых интересует:
#   - число размеченных образцов < 20;
cat_tree = cat_tree.merge(
    train['cat_id'].value_counts(), on='cat_id', how='left',
)

cat_tree['count'] = cat_tree['count'].fillna(0)
target_cats = cat_tree[cat_tree['count'] < 20]['cat_name'].values

# инициализация модели
model = SentenceTransformer('models/e5-large-en-ru', trust_remote_code=True)

# поиск наиболее схожих категорий с помощью модели retriever
# время выполнения: около 40 мин
idx2cat = {k: v for k, v in enumerate(cat_tree['cat_name'].unique())}

cat_embs = model.encode(
    cat_tree['cat_name'].unique(), batch_size=64, convert_to_tensor=True,
)

item_embs = model.encode(
    unlabeled_train['source_name'].values,
    batch_size=64,
    convert_to_tensor=True,
)

top_cats = util.semantic_search(item_embs, cat_embs, top_k=1)

predicted_cats = []  # наиболее схожая категория
scores = []  # косинусное сходство между описанием товара и оглавлением категории

for top in top_cats:
    predicted_cats.append(idx2cat[top[0]['corpus_id']])
    scores.append(top[0]['score'])

unlabeled_train['cat_name'] = predicted_cats
unlabeled_train['cos_sim'] = scores

# присвоение id категории товарам, где косинусное сходство > 0.8
with open('cat2id.pkl', mode='rb') as f:
    cat2id = pickle.load(f)

temp_df = unlabeled_train[unlabeled_train['cat_name'].isin(target_cats)]
temp_df = temp_df[temp_df['cos_sim'] > 0.8]
temp_df['cat_id'] = temp_df['cat_name'].map(cat2id)

# удаление дубликатов
temp_df = temp_df[~temp_df['hash_id'].isin(train['hash_id'].unique())]

# отбор 20-ти случайных образцов для каждого из целевых классов
temp_df = temp_df.groupby('cat_id').sample(frac=1).groupby('cat_id').head(20)

# расширение выборки
train = pd.concat([
    train,
    temp_df[['hash_id', 'source_name', 'attributes', 'cat_id']],
])

train = train.drop_duplicates('source_name')
train.to_parquet('data/train.parquet', index=False)
