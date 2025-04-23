import pickle

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.class_weight import compute_sample_weight

# чтение расширенных данных
train = pd.read_parquet('data/train.parquet')
cat_tree = pd.read_csv('data/category_tree.csv')

# предоработка данных
train['source_name'] = train['source_name'].str.lower()

# инициализация, обучение и сохранение моделей
vectorizer = CountVectorizer(lowercase=False, min_df=2)
classifier = MultinomialNB(alpha=5e-4)

X_train = vectorizer.fit_transform(train['source_name'].values)
y_train = train['cat_id'].values

sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)
classifier.fit(X_train, y_train, sample_weight=sample_weight)

with open('models/vectorizer.pkl', mode='wb') as f:
    pickle.dump(vectorizer, f)

with open('models/classifier.pkl', mode='wb') as f:
    pickle.dump(classifier, f)
