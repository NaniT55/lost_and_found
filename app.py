import os
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
from scipy.sparse import vstack

TRAIN_PATH = r"C:/Users/Varun.T/Downloads/lostfound/data/train.csv"
TEST_PATH  = r"C:/Users/Varun.T/Downloads/lostfound/data/test.csv"

# Output directory (created inside data folder)
DATA_DIR = "C:/Users/Varun.T/Downloads/lostfound/data"
OUTPUT_DIR = os.path.join(DATA_DIR, "lr_model_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# TF-IDF / SVD settings
TFIDF_MAX_FEATURES = 15000
TFIDF_NGRAM_RANGE = (1,2)
SVD_COMPONENTS = 150   # set to None to skip SVD

# Reranking / candidate generation
CANDIDATE_K = 100
TOP_K_LIST = [1,3,5]

# Toggle removing trivial leak-features (recommended True for realistic eval)
REMOVE_LEAK_FEATURES = True

# -----------------------------
# Utilities
# -----------------------------
def safe_str(x):
    return "" if pd.isna(x) else str(x)

def jaccard_text(a, b):
    if pd.isna(a) or pd.isna(b):
        return 0.0
    sa = set(str(a).lower().split())
    sb = set(str(b).lower().split())
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def compute_ranking_metrics(df_pairs, scores_col='score', lost_id_col='lost_id', found_id_col='found_id', label_col='label', topk_list=[1,3,5]):
    grouped = df_pairs.groupby(lost_id_col)
    mrr_total = 0.0
    n_q = 0
    topk_counts = {k: 0 for k in topk_list}
    for lost_id, group in grouped:
        labels = group.sort_values(by=scores_col, ascending=False)[label_col].values
        if labels.sum() == 0:
            continue
        n_q += 1
        pos = np.where(labels == 1)[0]
        rank = pos[0] + 1
        mrr_total += 1.0 / rank
        for k in topk_list:
            if rank <= k:
                topk_counts[k] += 1
    mrr = mrr_total / n_q if n_q > 0 else 0.0
    topk_acc = {k: topk_counts[k] / n_q for k in topk_list} if n_q > 0 else {k: 0.0 for k in topk_list}
    return {'mrr': mrr, 'topk': topk_acc, 'n_queries': n_q}

# -----------------------------
# 1) Load data
# -----------------------------
print("Loading datasets...")
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
print("Train shape:", train.shape)
print("Test  shape:", test.shape)

# ensure date columns are parsed
for df in (train, test):
    for c in ['lost_date', 'found_date']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')

# -----------------------------
# 2) Prepare text fields for TF-IDF
# -----------------------------
train = train.copy()
test = test.copy()

train['lost_text_full']  = (train.get('lost_title','').fillna('') + " " + train.get('lost_description','').fillna('')).str.strip()
train['found_text_full'] = (train.get('found_title','').fillna('') + " " + train.get('found_description','').fillna('')).str.strip()

test['lost_text_full']  = (test.get('lost_title','').fillna('') + " " + test.get('lost_description','').fillna('')).str.strip()
test['found_text_full'] = (test.get('found_title','').fillna('') + " " + test.get('found_description','').fillna('')).str.strip()

# Build corpus (fit TF-IDF on union)
all_texts = pd.concat([
    train['lost_text_full'], train['found_text_full'],
    test['lost_text_full'], test['found_text_full']
], axis=0).unique()

# -----------------------------
# 3) TF-IDF (+ optional SVD)
# -----------------------------
print("Fitting TF-IDF vectorizer...")
tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=TFIDF_NGRAM_RANGE)
tfidf.fit(all_texts)

svd = None
if SVD_COMPONENTS is not None:
    print("Fitting TruncatedSVD...")
    svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=42)
    # Fit SVD on combined training vectors for stability
    train_l_tfidf = tfidf.transform(train['lost_text_full'])
    train_f_tfidf = tfidf.transform(train['found_text_full'])
    combined = vstack([train_l_tfidf, train_f_tfidf])
    svd.fit(combined)
    # now transform train/test below using svd.transform

# Save TF-IDF and SVD objects
with open(os.path.join(OUTPUT_DIR, 'tfidf.pkl'), 'wb') as f:
    pickle.dump(tfidf, f)
if svd is not None:
    with open(os.path.join(OUTPUT_DIR, 'svd.pkl'), 'wb') as f:
        pickle.dump(svd, f)

# -----------------------------
# 4) Create pair features
# -----------------------------
print("Creating pairwise features...")

def compute_text_vecs(series_l, series_f):
    l_tfidf = tfidf.transform(series_l)
    f_tfidf = tfidf.transform(series_f)
    if svd is not None:
        l_vec = svd.transform(l_tfidf)
        f_vec = svd.transform(f_tfidf)
    else:
        l_vec = l_tfidf.toarray()
        f_vec = f_tfidf.toarray()
    return l_vec, f_vec

train_l_vec, train_f_vec = compute_text_vecs(train['lost_text_full'], train['found_text_full'])
test_l_vec, test_f_vec   = compute_text_vecs(test['lost_text_full'], test['found_text_full'])

# text cosine per row (pair-level)
# ensure same length and pairwise relation
n_train_pairs = train_l_vec.shape[0]
train_text_cosines = np.array([
    cosine_similarity(train_l_vec[i].reshape(1, -1), train_f_vec[i].reshape(1, -1))[0, 0]
    for i in range(n_train_pairs)
])
train['text_cosine'] = train_text_cosines

n_test_pairs = test_l_vec.shape[0]
test_text_cosines = np.array([
    cosine_similarity(test_l_vec[i].reshape(1, -1), test_f_vec[i].reshape(1, -1))[0, 0]
    for i in range(n_test_pairs)
])
test['text_cosine'] = test_text_cosines

# additional features
cat_keywords = ['water bottle','wallet','laptop','phone','keychain','umbrella','notebook','backpack','headphones','glasses','id card','watch','charger','earbuds','jacket','sneakers']

for df in (train, test):
    df.loc[:, 'title_jaccard'] = df.apply(lambda r: jaccard_text(safe_str(r.get('lost_title','')), safe_str(r.get('found_title',''))), axis=1)
    df.loc[:, 'desc_jaccard']  = df.apply(lambda r: jaccard_text(safe_str(r.get('lost_description','')), safe_str(r.get('found_description',''))), axis=1)
    df.loc[:, 'location_jaccard'] = df.apply(lambda r: jaccard_text(safe_str(r.get('lost_location','')), safe_str(r.get('found_location',''))), axis=1)

    # date diff
    if 'lost_date' in df.columns and 'found_date' in df.columns:
        df.loc[:, 'date_diff_days'] = (df['lost_date'] - df['found_date']).abs().dt.days.fillna(999).astype(int)
    else:
        df.loc[:, 'date_diff_days'] = 999

    df.loc[:, 'desc_len_diff'] = (df.get('lost_description','').fillna('').str.len() - df.get('found_description','').fillna('').str.len()).abs()

    # leak features (may remove later)
    df.loc[:, 'same_color'] = (df.get('lost_color','').fillna('').str.lower().str.strip() == df.get('found_color','').fillna('').str.lower().str.strip()).astype(int)
    df.loc[:, 'same_brand'] = (df.get('lost_brand','').fillna('').str.lower().str.strip() == df.get('found_brand','').fillna('').str.lower().str.strip()).astype(int)
    # basic same_category detection by common keywords
    df.loc[:, 'same_category'] = df.apply(lambda r: int(
        any(k in safe_str(r.get('lost_title','')).lower() for k in cat_keywords)
        and any(k in safe_str(r.get('found_title','')).lower() for k in cat_keywords)
    ), axis=1)

    # interactions
    df.loc[:, 'text_color_interaction'] = df['text_cosine'] * df['same_color']
    df.loc[:, 'text_loc_interaction'] = df['text_cosine'] * df['location_jaccard']

# -----------------------------
# 5) Select features and prepare arrays
# -----------------------------
BASE_FEATURES = [
    'text_cosine','title_jaccard','desc_jaccard',
    'location_jaccard','date_diff_days','desc_len_diff',
    'text_color_interaction','text_loc_interaction'
]
LEAK_FEATURES = ['same_color','same_brand','same_category']
FEATURES = BASE_FEATURES + ([] if REMOVE_LEAK_FEATURES else LEAK_FEATURES)

X_train = train[FEATURES].fillna(0).astype(float)
y_train = train['label'].astype(int)
X_test  = test[FEATURES].fillna(0).astype(float)
y_test  = test['label'].astype(int)

print("Feature matrix shapes:", X_train.shape, X_test.shape)

# scale features for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

with open(os.path.join(OUTPUT_DIR, 'scaler.pkl'),'wb') as f:
    pickle.dump(scaler, f)

# -----------------------------
# 6) Train logistic regression
# -----------------------------
print("Training Logistic Regression...")
clf = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)
clf.fit(X_train_scaled, y_train)

with open(os.path.join(OUTPUT_DIR, 'logreg_model.pkl'), 'wb') as f:
    pickle.dump(clf, f)

# -----------------------------
# 7) Evaluate classification (pair-level)
# -----------------------------
probs = clf.predict_proba(X_test_scaled)[:,1]
preds = (probs >= 0.5).astype(int)

acc = accuracy_score(y_test, preds)
auc = roc_auc_score(y_test, probs)
prec, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average='binary', zero_division=0)

print(f"Test Accuracy: {acc:.4f}")
print(f"Test AUC: {auc:.4f}")
print(f"Precision: {prec:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
if acc >= 0.995:
    print("\nWARNING: Test accuracy extremely high (>=99.5%). Likely leakage or very easy negatives.")

# Save test predictions
test_out = test.copy()
test_out['score'] = probs
test_out['pred'] = preds
test_out.to_csv(os.path.join(OUTPUT_DIR, 'test_with_preds.csv'), index=False)
print("Saved prediction CSV ->", os.path.join(OUTPUT_DIR, 'test_with_preds.csv'))

# -----------------------------
# 8) Inspect coefficients
# -----------------------------
coef_df = pd.DataFrame({'feature': FEATURES, 'coef': clf.coef_[0]})
coef_df['abs_coef'] = coef_df['coef'].abs()
coef_df = coef_df.sort_values(by='abs_coef', ascending=False)
print("Top features by absolute coefficient:")
print(coef_df)
coef_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_coefs.csv'), index=False)

# -----------------------------
# 9) Candidate generation + rerank evaluation
# -----------------------------
print("Building found-item vector store for candidate generation...")
found_pool = pd.concat([
    train[['found_id','found_text_full','found_title','found_description','found_date','found_location','found_color','found_brand','found_category']] if 'found_category' in train.columns else train[['found_id','found_text_full','found_title','found_description','found_date','found_location','found_color','found_brand']],
    test[['found_id','found_text_full','found_title','found_description','found_date','found_location','found_color','found_brand','found_category']] if 'found_category' in test.columns else test[['found_id','found_text_full','found_title','found_description','found_date','found_location','found_color','found_brand']]
], axis=0).drop_duplicates(subset=['found_id']).reset_index(drop=True)

found_texts = found_pool['found_text_full'].fillna('').values
found_tfidf = tfidf.transform(found_texts)
if svd is not None:
    found_vecs = svd.transform(found_tfidf)
else:
    found_vecs = found_tfidf.toarray()

assert found_vecs.shape[0] == found_pool.shape[0]

# Pre-build a quick lookup map from (lost_id, found_id) -> label in test for efficient assignment
test_pair_labels = test[['lost_id','found_id','label']].copy()
test_pair_labels['pair_key'] = test_pair_labels['lost_id'].astype(str) + "||" + test_pair_labels['found_id'].astype(str)
label_map = test_pair_labels.set_index('pair_key')['label'].to_dict()

# rerank function
def rerank_for_lost(lost_row, top_k=CANDIDATE_K):
    lost_text = safe_str(lost_row['lost_text_full'])
    lost_tfidf = tfidf.transform([lost_text])
    if svd is not None:
        lost_vec = svd.transform(lost_tfidf)
    else:
        lost_vec = lost_tfidf.toarray()

    sims = cosine_similarity(lost_vec, found_vecs)[0]
    topk_idx = np.argsort(sims)[::-1][:top_k]
    candidates = found_pool.iloc[topk_idx].copy().reset_index(drop=True)
    candidates['tfidf_sim'] = sims[topk_idx]

    feat_rows = []
    for _, frow in candidates.iterrows():
        title_j = jaccard_text(safe_str(lost_row.get('lost_title','')), safe_str(frow.get('found_title','')))
        desc_j  = jaccard_text(safe_str(lost_row.get('lost_description','')), safe_str(frow.get('found_description','')))
        loc_j   = jaccard_text(safe_str(lost_row.get('lost_location','')), safe_str(frow.get('found_location','')))
        # date diff robust
        try:
            if pd.notna(lost_row.get('lost_date')) and pd.notna(frow.get('found_date')):
                date_diff = abs((pd.to_datetime(lost_row.get('lost_date')) - pd.to_datetime(frow.get('found_date'))).days)
            else:
                date_diff = 999
        except Exception:
            date_diff = 999

        desc_len_diff = abs(len(safe_str(lost_row.get('lost_description',''))) - len(safe_str(frow.get('found_description',''))))
        # compute text cosine (lost_vec vs candidate vector)
        if svd is not None:
            cand_vec = svd.transform(tfidf.transform([frow['found_text_full']]))
        else:
            cand_vec = tfidf.transform([frow['found_text_full']]).toarray()
        text_cos = float(cosine_similarity(lost_vec, cand_vec)[0,0])
        same_color = int(safe_str(lost_row.get('lost_color','')).strip().lower() == safe_str(frow.get('found_color','')).strip().lower())
        same_brand = int(safe_str(lost_row.get('lost_brand','')).strip().lower() == safe_str(frow.get('found_brand','')).strip().lower())
        # compute same_category similarly to training
        same_category = int(any(k in safe_str(lost_row.get('lost_title','')).lower() for k in cat_keywords) and any(k in safe_str(frow.get('found_title','')).lower() for k in cat_keywords))

        text_color_inter = text_cos * same_color
        text_loc_inter = text_cos * loc_j

        row_feats = [text_cos, title_j, desc_j, loc_j, date_diff, desc_len_diff, text_color_inter, text_loc_inter]
        if not REMOVE_LEAK_FEATURES:
            row_feats += [same_color, same_brand, same_category]
        feat_rows.append(row_feats)

    feats_df = pd.DataFrame(feat_rows, columns=FEATURES).fillna(0)
    feats_scaled = scaler.transform(feats_df)
    scores = clf.predict_proba(feats_scaled)[:,1]
    candidates['score'] = scores

    # attach label if pair exists in test (conservative)
    pair_keys = [str(lost_row['lost_id']) + "||" + str(fid) for fid in candidates['found_id'].values]
    candidates['label'] = [int(label_map.get(k, 0)) for k in pair_keys]

    return candidates.sort_values(by='score', ascending=False).reset_index(drop=True)

# run rerank for each lost in test
print("Running rerank across test lost items (this may take some time)...")
rows = []
unique_losts = test['lost_id'].unique()
for lid in unique_losts:
    lost_rows = test[test['lost_id']==lid]
    if lost_rows.shape[0] == 0:
        continue
    lost_row = lost_rows.iloc[0]
    ranked = rerank_for_lost(lost_row, top_k=CANDIDATE_K)
    for _, r in ranked.iterrows():
        rows.append({'lost_id': lid, 'found_id': r['found_id'], 'score': r['score'], 'label': int(r['label'])})

results_df = pd.DataFrame(rows)
if results_df.shape[0] > 0:
    rerank_metrics = compute_ranking_metrics(results_df, scores_col='score', lost_id_col='lost_id', found_id_col='found_id', label_col='label', topk_list=TOP_K_LIST)
    print("Rerank metrics (after candidate generation):")
    print(f"MRR: {rerank_metrics['mrr']:.4f}")
    for k,v in rerank_metrics['topk'].items():
        print(f"Top-{k} accuracy: {v:.4f}")
    results_df.to_csv(os.path.join(OUTPUT_DIR, f'rerank_results_top{CANDIDATE_K}.csv'), index=False)
    print("Saved rerank CSV ->", os.path.join(OUTPUT_DIR, f'rerank_results_top{CANDIDATE_K}.csv'))
else:
    print("No rerank results collected.")

print("Done. Artifacts saved in:", OUTPUT_DIR)
print("TF-IDF ->", os.path.join(OUTPUT_DIR, 'tfidf.pkl'))
if svd is not None:
    print("SVD ->", os.path.join(OUTPUT_DIR, 'svd.pkl'))
print("Scaler ->", os.path.join(OUTPUT_DIR, 'scaler.pkl'))
print("Model ->", os.path.join(OUTPUT_DIR, 'logreg_model.pkl'))
