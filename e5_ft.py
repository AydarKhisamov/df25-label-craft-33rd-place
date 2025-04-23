import pandas as pd
from datasets import Dataset
from sentence_transformers import (
    util,
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

# инициализация параметров обучения
RANDOM_STATE = 42
TRAIN_SAMPLES_PER_CAT = 150  # макс кол-во обучающих примеров для каждого класса
TRAIN_NUM_NEGATIVES = 10     # кол-во негативных примеров для обучения

# чтение расширенных данных
train = pd.read_parquet('data/train.parquet')
cat_tree = pd.read_csv('data/category_tree.csv')


def get_triplets(df,
                 model=None,
                 cat_tree=cat_tree,
                 add_triplets_with_cats=False,
                 num_negatives=None,
                 random_state=RANDOM_STATE):
    '''Создаёт выборку в виде триплетов: query, positive, negative.

    Params:
        df: датафрейм с описанием товара и меткой категории;
        model: модель для генерации негативных кандидатов;
        cat_tree: категориальное древо;
        add_triplets_with_cats: флаг добавления в выборку триплетов с самими
            категориями, где pos - истинная родительская категория, neg -
            имеет общего предка с истинной родительской категорией query;
        num_negatives: кол-во негативных примеров для каждого товара;
        random_state: начальное состояние генератора случайных чисел;

    Returns:
        df: датафрейм в виде триплетов: query, positive, negative;
    '''
    id2text = {
        k: v for k, v in zip(cat_tree['cat_id'], cat_tree['cat_name'])
    }

    cats = cat_tree['cat_name'].unique()
    id2cat = {k: v for k, v in enumerate(cats)}

    enc_params = {'show_progress_bar': False, 'convert_to_tensor': True}
    cat_embs = model.encode(cats, **enc_params)
    item_embs = model.encode(df['source_name'].values, **enc_params)

    candidates = util.semantic_search(
        item_embs, cat_embs, top_k=num_negatives + 1,
    )

    df = df.rename(columns={'cat_id': 'positive_cat'})

    candidates_dict = {}
    h_idxs = df['hash_id'].tolist()
    for i in range(len(df)):
        candidates_dict[h_idxs[i]] = []
        for c in candidates[i]:
            candidates_dict[h_idxs[i]].append(id2cat[c['corpus_id']])

    df['negative_cat'] = df['hash_id'].map(candidates_dict)
    df = df.explode('negative_cat')
    df['positive_cat'] = df['positive_cat'].map(id2text)
    df = df[df['positive_cat'] != df['negative_cat']]

    df = df[['source_name', 'positive_cat', 'negative_cat']]

    if add_triplets_with_cats:
        id2parent = {
            k: v for k, v in zip(cat_tree['cat_id'], cat_tree['parent_id'])
        }

        temp_df = cat_tree[['cat_name', 'parent_id']].dropna().copy()
        temp_df = temp_df.rename(
            columns={'cat_name': 'source_name', 'parent_id': 'positive_cat'},
        )
        temp_df['parent_id'] = temp_df['positive_cat'].map(id2parent)

        temp_df = temp_df.merge(
            cat_tree[['cat_id', 'parent_id']], on='parent_id',
        )
        temp_df = temp_df.rename(columns={'cat_id': 'negative_cat'})
        temp_df = temp_df[temp_df['positive_cat'] != temp_df['negative_cat']]
        temp_df = temp_df[['source_name', 'positive_cat', 'negative_cat']]

        temp_df['positive_cat'] = temp_df['positive_cat'].map(id2text)
        temp_df['negative_cat'] = temp_df['negative_cat'].map(id2text)

        df = pd.concat([df, temp_df])

    df = df.reset_index(drop=True)

    return df[['source_name', 'positive_cat', 'negative_cat']]


# инициализация, обучение и сохранение модели
model = SentenceTransformer('d0rj/e5-large-en-ru', trust_remote_code=True)
loss_func = MultipleNegativesRankingLoss(model)

training_args = SentenceTransformerTrainingArguments(
    output_dir='models/e5-large-en-ru',
    eval_strategy='no',
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=8,
    torch_empty_cache_steps=16,
    num_train_epochs=1,
    lr_scheduler_type='cosine_with_restarts',
    logging_strategy='no',
    save_strategy='no',
    fp16=True,
    bf16=False,
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    report_to='none',
)

train = (train.groupby('cat_id').sample(frac=1, random_state=RANDOM_STATE)
              .groupby('cat_id').head(TRAIN_SAMPLES_PER_CAT))

triplets_train = get_triplets(
    train,
    model=model,
    add_triplets_with_cats=True,
    num_negatives=TRAIN_NUM_NEGATIVES,
)

triplets_train = (Dataset
                  .from_pandas(triplets_train, preserve_index=False)
                  .shuffle(RANDOM_STATE))

trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,
    train_dataset=triplets_train,
    loss=loss_func,
)

trainer.train()

model.save_pretrained('models/e5-large-en-ru')
