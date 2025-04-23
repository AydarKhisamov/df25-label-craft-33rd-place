import re
import pickle
from copy import copy

import numpy as np
import pandas as pd

! python -m spacy download ru_core_news_lg -qq
import spacy

from pymorphy3 import MorphAnalyzer


labeled_train = pd.read_parquet('data/labeled_train.parquet')
cat_tree = pd.read_csv('data/category_tree.csv')
unlabeled_train = pd.read_parquet('data/unlabeled_train.parquet')
special_train = pd.read_parquet('data/unlabeled_special_prize.parquet')

# изменение оглавлений категорий, у которых есть дубликаты по оглавлению, но не по смыслу
cat_tree.loc[292, 'cat_name'] = 'Специальное фото- и видеооборудование'
cat_tree.loc[307, 'cat_name'] = 'Электроника (Умный дом)'
cat_tree.loc[312, 'cat_name'] = 'Аксессуары (Автомобильная электроника)'
cat_tree.loc[796, 'cat_name'] = 'Аксессуары (Стиль)'
cat_tree.loc[939, 'cat_name'] = 'Программное обеспечение'
cat_tree.loc[961, 'cat_name'] = 'Источники бесперебойного питания'
cat_tree.loc[1740, 'cat_name'] = 'Внутренние жесткие диски'
cat_tree.loc[1741, 'cat_name'] = 'Внутренние накопители (SSD)'

cat_tree.to_csv('data/category_tree.csv', index=False)

# объединение размеченной выборки с дополнительными данными от организаторов
labeled_train = pd.concat([labeled_train, special_train])
labeled_train = labeled_train.reset_index(drop=True)
labeled_train = labeled_train.drop_duplicates(['source_name', 'cat_id'])

# индексы дубликатов с противоречивой разметкой
d_idxs = labeled_train[
    labeled_train.duplicated('source_name', keep=False)
].index

# перемещение таких дубликатов в неразмеченную выборку
d_df = (labeled_train
        .loc[d_idxs]
        .drop(columns='cat_id')
        .drop_duplicates('source_name')
)

unlabeled_train = pd.concat([unlabeled_train, d_df])
unlabeled_train = unlabeled_train.reset_index(drop=True)
unlabeled_train = unlabeled_train.drop_duplicates('source_name')
labeled_train = labeled_train.drop(index=d_idxs)

# создание и сохранение словаря duplicates-to-one - d2one
# все id категорий товаров с дублирующимися названиями заменяются на единственный:
#   - самый популярный среди id с таким названием в размеченной выборке;
#   - случайный, если нет ни одного id в размеченной выборке;
temp_df = cat_tree[['cat_id', 'cat_name']].merge(
    labeled_train['cat_id'].value_counts(),
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

temp_df = temp_df.merge(
    cat_tree[['cat_id', 'cat_name']],
    on='cat_name',
    suffixes=(None, '_all'),
)

d2one = {k: v for k, v in zip(temp_df['cat_id_all'], temp_df['cat_id'])}
with open('d2one.pkl', mode='wb') as f:
    pickle.dump(d2one, f)

morph = MorphAnalyzer()
nlp = spacy.load("ru_core_news_lg")

# категории, разметка которых интересует:
#   - нет дочерних категорий;
#   - число размеченных образцов < 20;
target_ids = np.setdiff1d(cat_tree['cat_id'], cat_tree['parent_id'])
target_cats = cat_tree[
    cat_tree['cat_id'].isin(target_ids)
]['cat_name'].unique()

target_cats = np.intersect1d(
    temp_df[temp_df['count'] < 20]['cat_name'].unique(),
    target_cats,
)


def lemmatize(word, morph, pos, target_form=None):
    """Приводит к начальной форме одно слово.

    Params:
        word: слово для приведения в начальную форму;
        morph: лемматизатор;
        pos: часть речи (part-of-speech) начальной формы слова;
        target_form: целевые параметры слова, как падеж, число и пр.;

    Returns:
        начальная форма слова с учётом целевых параметров;
    """
    if target_form:
        target_form.update({'pos': pos.upper()})

    if '-' in word:
        output = {f'p{k}': v for k, v in enumerate(word.split('-'))}

        if pos == 'noun':
            parts = ['p0', 'p1']
        elif pos == 'adjf':
            parts = ['p1']

        if target_form:
            for part in parts:
                for parse in morph.parse(output[part]):
                    try:
                        output[part] = (parse
                                        .inflect(set(target_form.values()))
                                        .word)
                        break

                    except:
                        continue

        else:
            for part in parts:
                output[part] = morph.parse(output[part])[0].normal_form
                
        return '-'.join(list(output.values()))

    else:
        if target_form:
            for parse in morph.parse(word):
                try:
                    word = parse.inflect(set(target_form.values())).word
                    return word
                except:
                    continue

            if pos == 'adjf':
                target_form['pos'] = 'PRTF'
                for parse in morph.parse(word):
                    try:
                        word = parse.inflect(set(target_form.values())).word
                        return word
                    except:
                        continue

        return morph.parse(word)[0].normal_form
    

def lemmatize_phrase(phrase, nlp, morph, target_form=None):
    """Приводит словосочетание к начальной форме.

    Params:
        phrase: словосочетание для приведения в начальную форму;
        nlp: инструмент для морфологического анализа;
        morph: лемматизатор;
        target_form: целевые параметры слова, как падеж, число и пр.;

    Returns:
        pat: словосочетание в начальной форме с учётом целевых параметров;
    """
    pat = copy(phrase)

    words = phrase.split(' ')
    doc = nlp(phrase)
    tokens = [str(w) for w in doc]
    deps = [w.dep_ for w in doc]
    heads = [str(w.head) for w in doc]
    pos = [w.pos_ for w in doc]

    try:
        subj_idx = deps.index('nsubj')
    except ValueError:
        subj_idx = deps.index('ROOT')

    token_in_word = [
        (w.startswith(tokens[subj_idx])) or
        (w.endswith(tokens[subj_idx]))
        for w in words
    ]
    old_subj = words[token_in_word.index(True)]

    if target_form:
        new_subj = lemmatize(old_subj, morph, 'noun', target_form)
    else:
        new_subj = lemmatize(old_subj, morph, 'noun')
    pat = pat.replace(old_subj, new_subj)

    for i in range(len(tokens)):
        if ((pos[i] in ['ADJ', 'VERB']) and
           ((heads[i] == tokens[subj_idx]) or
            ((deps[i] == 'ROOT') and tokens[i] not in old_subj))):
            token_in_word = [w.endswith(tokens[i]) for w in words]

            if any(token_in_word):
                old_adj = words[token_in_word.index(True)]
                tform = {}
                tform['gender'] = morph.parse(new_subj)[0].tag.gender
                tform['number'] = morph.parse(new_subj)[0].tag.number
                tform['case'] = 'nomn'
                if target_form:
                    tform.update(target_form)

                if len(tform) > 0:
                    new_adj = lemmatize(old_adj, morph, 'adjf', tform)
                else:
                    new_adj = lemmatize(old_adj, morph, 'adjf')

                pat = pat.replace(old_adj, new_adj)

    return pat


### словарь, где каждой категории товара соответствуют паттерны описания
cat2pats = {}

for cat in target_cats:
    pats = []
    if ' ' not in cat:
        pat = lemmatize(cat, morph, 'noun')
        pats.append(pat)

        if '-' in pat:
            pats.append(' '.join(pat.split('-')))

    elif re.search(r'\S+ \S+', cat):
        if not re.search(r'\bи\b', cat):
            if re.search(r'^Проч\w{2} ', cat):
                if cat.split(' ')[1] == 'аксессуары':
                    pats.append(' '.join(cat.split(' ')[2:]))
                else:
                    pats.append(' '.join(cat.split(' ')[1:]))

            elif cat.split(' ')[0] == 'Для':
                pats.append(cat)

            elif cat.split(' ')[0] == 'Аксессуары':
                pats.append(' '.join(cat.split(' ')[1:]))

            elif cat.split(' ')[0] == 'Все':
                pats.append(lemmatize(cat.split(' ')[1], morph, 'noun'))

            elif ',' in cat:
                if 'для' in cat:
                    subjs, objs = cat.split(' для ')

                    subjs = subjs.split(', ')
                    new_subjs = []
                    for s in subjs:
                        if ' ' not in s:
                            new_subjs.append(lemmatize(s, morph, 'noun'))
                        else:
                            new_subjs.append(lemmatize_phrase(s, nlp, morph))

                    objs = objs.split(', ')
                    new_objs = []
                    for o in objs:
                        for f in [
                        {'number': 'sing', 'case': 'gent'}, {'case': 'gent'},
                        ]:
                            if ' ' not in o:
                                new_objs.append(lemmatize(o, morph, 'noun', f))
                            else:
                                new_objs.append(
                                    lemmatize_phrase(o, nlp, morph, f))
                    
                    pats.extend(list(set(
                        [f'{s} для {o}' for s in new_subjs for o in new_objs]
                    )))

                else:
                    subjs = cat.split(', ')
                    subjs = [lemmatize(s, morph, 'noun') for s in subjs]
                    pats.extend(subjs)

            else:
                if 'для' in cat:
                    subj, obj = cat.split(' для ')
                    new_subj = []
                    new_obj = []

                    if ' ' not in subj:
                        new_subj = lemmatize(subj, morph, 'noun')
                    else:
                        new_subj = lemmatize_phrase(subj, nlp, morph)

                    for f in [
                        {'number': 'sing', 'case': 'gent'}, {'case': 'gent'},
                    ]:
                        if ' ' not in obj:
                            new_obj = lemmatize(obj, morph, 'noun', f)
                        else:
                            new_obj = lemmatize_phrase(obj, nlp, morph, f)

                        if f'{new_subj} для {new_obj}' not in pats:
                            pats.append(f'{new_subj} для {new_obj}')

                else:
                    pat = lemmatize_phrase(cat, nlp, morph)
                    if len(pat.split(' ')) == 2:
                        if any([
                            pos in [w.pos_ for w in nlp(pat)]
                            for pos in ['ADJ', 'VERB']]):
                            pats.append(' '.join(reversed(pat.split(' '))))

                    pats.append(pat)

        else:
            if re.search(r'^\w+ и \w+$', cat):
                pats.extend(
                    [lemmatize(s, morph, 'noun') for s in cat.split(' и ')]
                )

            elif re.search(r'^(?:\w+,)+ \w+ и \w+$', cat):
                s1, s2 = cat.split(' и ')
                subjs = [s2]
                subjs.extend(s1.split(', '))
                pats.extend([lemmatize(s, morph, 'noun') for s in subjs])

            elif re.search(r'^\w+ для \w+ и \w+$', cat):
                subj, objs = cat.split(' для ')
                subj = lemmatize(subj, morph, 'noun')

                objs = objs.split(' и ')
                new_objs = []
                for f in [
                    {'number': 'sing', 'case': 'gent'}, {'case': 'gent'},
                ]:
                    for o in objs:
                        new_obj = lemmatize(o, morph, 'noun', f)
                        if new_obj not in new_objs:
                            new_objs.append(new_obj)

                pats.extend(
                    [f'{subj} для {o}'.replace('аксессуар ', '')
                     for o in new_objs]
                )

    for p in pats:
        if all([p.startswith('электро'), p != 'электроника', ' ' not in p]):
            a_pats = []
            subj = p.replace('электро', '')
            tform = {}
            tform['gender'] = morph.parse(subj)[0].tag.gender
            tform['number'] = morph.parse(subj)[0].tag.number
            tform['case'] = 'nomn'

            adj = lemmatize('электрический', morph, 'adjf', tform)

            a_pats.append(' '.join([adj, subj]))
            a_pats.append(' '.join([subj, adj]))

            pats.extend(a_pats)

    pats.extend([p.replace('ё', 'е') for p in pats if 'ё' in p])

    pats = [re.sub(r'\(.*\)', ' ', p).strip()
                        for p in pats
                        if p != 'аксессуар']
    pats = [p.replace('  ', '') for p in pats]

    for p in pats:
        words = p.split(' ')
        if len(words) > 2:
            a_pats = []
            doc = nlp(p)
            if [w.pos_ for w in doc][-1] in ['ADJ', 'VERB']:
                a_pat = [words[-1]]
                a_pat.extend(words[:-1])
                pats.append(' '.join(a_pat))

    if len(pats) > 0:
        cat2pats[cat] = pats


# разметка на основании паттернов описания
# время выполнения: около 20 мин.
idx2cat = {}

for cat, pats in cat2pats.items():
    for pat in pats:
        idxs = unlabeled_train[
            unlabeled_train['source_name'].str.contains(
                rf'\b{pat}\b', case=False,
            )
        ].index.tolist()

        for idx in idxs:
            if idx not in idx2cat.keys():
                idx2cat[idx] = []
            idx2cat[idx].append(cat)

unlabeled_train['cat_name'] = unlabeled_train.index.map(idx2cat)

# пост-обработка разметки
df = unlabeled_train.dropna(subset='cat_name')
df = df.explode('cat_name')
df['cat_name_len'] = df['cat_name'].str.len()
df['pattern'] = df['cat_name'].map(cat2pats)
df = df.explode('pattern')
df['pattern_len'] = df['pattern'].str.len()


def locate_pattern(text, pattern):
    """Определяет индекс начала паттерна описания."""
    try:
        return text.index(pattern)
    except ValueError:
        return None


df['pat_start_idx'] = df.progress_apply(
    lambda x: locate_pattern(x['source_name'].lower(), x['pattern'].lower()),
    axis=1,
)

df = df.dropna(subset='pat_start_idx')

# отбор наиболее вероятной метки по следующему алгоритму:
#   - паттерн расположен ближе к началу описания товара;
#   - паттерн имеет наибольшую длину в символах;
#   - оглавление метки-категории имеет наибольшую длину в символах;
df = df.sort_values(
    by=['hash_id', 'pat_start_idx', 'pattern_len', 'cat_name_len'],
    ascending=[True, True, False, True],
)
# в качестве разметки выбирается та категория, паттерн которой:
#   - расположен в самом начале описания товара;
#   - имеет длину в символах > 12;
df = df.groupby('hash_id').head(1)
df = df[(df['pat_start_idx'] == 0) & (df['pattern_len'] > 12)]

# присвоение размеченным данным id категории
cat_tree = cat_tree[['cat_id', 'cat_name']]
cat_tree['cat_id'] = cat_tree['cat_id'].map(d2one)
cat_tree = cat_tree.drop_duplicates()
cat2id = {k: v for k, v in zip(cat_tree['cat_name'], cat_tree['cat_id'])}

df['cat_id'] = df['cat_name'].map(cat2id)

with open('cat2id.pkl', mode='wb') as f:
    pickle.dump(cat2id, f)

# отбор 20-ти случайных образцов для каждого из целевых классов
df = (df
      .groupby('cat_id').sample(frac=1, random_state=42)
      .groupby('cat_id').head(20)
)

# сохранение всех размеченных данных в один файл
df = pd.concat(
    [labeled_train, df[['hash_id', 'source_name', 'attributes', 'cat_id']]],
).reset_index(drop=True)

df = df.drop_duplicates('source_name')
df.to_parquet('data/train.parquet', index=False)
