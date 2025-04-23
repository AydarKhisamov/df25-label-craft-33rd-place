import pickle
import argparse

import pandas as pd
from sentence_transformers import SentenceTransformer, util


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, help='test data path')
    parser.add_argument('--output_path', type=str, help='output file')
    args = parser.parse_args()

    test_data = pd.read_parquet(args.test_data_path)
    cat_tree = pd.read_csv('data/category_tree.csv')

    with open('cat2id.pkl', mode='rb') as f:
        cat2id = pickle.load(f)

    with open('d2one.pkl', mode='rb') as f:
        d2one = pickle.load(f)

    cat_tree = cat_tree[['cat_name']]
    cat_tree['cat_id'] = cat_tree['cat_name'].map(cat2id)
    cat_tree = cat_tree.drop_duplicates()

    retriever = SentenceTransformer(
        'models/e5-large-en-ru', trust_remote_code=True,
    )

    idx2cat = {k: v for k, v in enumerate(cat_tree['cat_id'])}

    cat_embs = retriever.encode(
        cat_tree['cat_name'].values,
        convert_to_tensor=True,
        show_progress_bar=False,
    )

    item_embs = retriever.encode(
        test_data['source_name'].values,
        convert_to_tensor=True,
        show_progress_bar=False,
    )

    ret_predictions = util.semantic_search(item_embs, cat_embs, top_k=100)

    with open('models/vectorizer.pkl', mode='rb') as f:
        vectorizer = pickle.load(f)

    with open('models/classifier.pkl', mode='rb') as f:
        classifier = pickle.load(f)

    X = vectorizer.transform(test_data['source_name'].str.lower().values)
    cl_predictions = classifier.predict(X)

    # ансамбль из ретривера и классификатора
    predicted_ids = []

    for rp, cp in zip(ret_predictions, cl_predictions):
        ret_candidates = [idx2cat[p['corpus_id']] for p in rp]

        if (cp in ret_candidates) or (d2one[cp] in ret_candidates):
            predicted_ids.append(cp)
        else:
            predicted_ids.append(ret_candidates[0])

    # финальное предсказание
    test_data['predicted_cat'] = predicted_ids
    test_data[['hash_id', 'predicted_cat']].to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()
