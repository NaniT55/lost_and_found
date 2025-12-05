import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

# ---------- CONFIG - update if needed ----------
# --- Updated data & model paths (match training script) ---
DATA_DIR = r"C:/Users/Varun.T/Downloads/lostfound/data"
MODEL_DIR = os.path.join(DATA_DIR, "lr_model_outputs")

TFIDF_PATH  = os.path.join(MODEL_DIR, "tfidf.pkl")
SVD_PATH    = os.path.join(MODEL_DIR, "svd.pkl")   # may not exist
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
MODEL_PATH  = os.path.join(MODEL_DIR, "logreg_model.pkl")

# Use the main train/test files (as in the training script)
FOUND_POOL_PATHS = [
    os.path.join(DATA_DIR, "train.csv"),
    os.path.join(DATA_DIR, "test.csv"),
]

# Feedback log (keeps same directory)
FEEDBACK_LOG = os.path.normpath(os.path.join(DATA_DIR, "streamlit_feedback.csv"))



CANDIDATE_K_DEFAULT = 50
TOP_K_SHOW = 5

# Feature column order used by training (base features - matches train setup when REMOVE_LEAK_FEATURES=True)
FEATURE_COLUMNS = [
    'text_cosine','title_jaccard','desc_jaccard','location_jaccard',
    'date_diff_days','desc_len_diff','text_color_interaction','text_loc_interaction'
]

# ---------- Helpers ----------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    if not os.path.exists(TFIDF_PATH):
        st.error(f"TF-IDF vectorizer not found at {TFIDF_PATH}. Run training first.")
        st.stop()
    with open(TFIDF_PATH, "rb") as f:
        tfidf = pickle.load(f)
    svd = None
    if os.path.exists(SVD_PATH):
        with open(SVD_PATH, "rb") as f:
            svd = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(MODEL_PATH, "rb") as f:
        clf = pickle.load(f)
    return tfidf, svd, scaler, clf

@st.cache_data(show_spinner=False)
def build_found_pool():
    dfs = []
    for p in FOUND_POOL_PATHS:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p, dtype=str)
                dfs.append(df)
            except Exception:
                continue
    if len(dfs) == 0:
        st.warning("No found-item source CSVs found. Put train/test CSVs in data folder.")
        return pd.DataFrame(columns=['found_id','found_text_full','found_title','found_description','found_date','found_location','found_color','found_brand'])
    pool = pd.concat(dfs, axis=0, ignore_index=True).fillna('')
    pool['found_id'] = pool['found_id'].astype(str)
    pool['found_text_full'] = (pool.get('found_title','') + ' ' + pool.get('found_description','')).astype(str)
    pool = pool.drop_duplicates(subset=['found_id']).reset_index(drop=True)
    return pool

def jaccard_text(a, b):
    if pd.isna(a) or pd.isna(b):
        return 0.0
    sa = set(str(a).lower().split())
    sb = set(str(b).lower().split())
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def compute_pair_features(lost_row, found_row, tfidf, svd):
    # lost_row: dict with keys 'title','description','date','location','color','brand'
    lost_text = (lost_row.get('title','') + ' ' + lost_row.get('description','')).strip()
    found_text = found_row.get('found_text_full','')
    # TF-IDF vectors
    l_tfidf = tfidf.transform([lost_text])
    f_tfidf = tfidf.transform([found_text])
    if svd is not None:
        l_vec = svd.transform(l_tfidf)
        f_vec = svd.transform(f_tfidf)
    else:
        l_vec = l_tfidf.toarray()
        f_vec = f_tfidf.toarray()
    # compute similarities and hand-crafted features
    try:
        text_cos = float(cosine_similarity(l_vec, f_vec)[0,0])
    except Exception:
        text_cos = 0.0
    title_j = jaccard_text(lost_row.get('title',''), found_row.get('found_title',''))
    desc_j  = jaccard_text(lost_row.get('description',''), found_row.get('found_description',''))
    loc_j   = jaccard_text(lost_row.get('location',''), found_row.get('found_location',''))
    try:
        date_diff = abs((pd.to_datetime(lost_row.get('date')) - pd.to_datetime(found_row.get('found_date'))).days)
        if pd.isna(date_diff):
            date_diff = 999
    except Exception:
        date_diff = 999
    desc_len_diff = abs(len(lost_row.get('description','')) - len(found_row.get('found_description','') or ''))
    same_color = int(str(lost_row.get('color','')).strip().lower() == str(found_row.get('found_color','')).strip().lower())
    same_brand = int(str(lost_row.get('brand','')).strip().lower() == str(found_row.get('found_brand','')).strip().lower())
    text_color_inter = text_cos * same_color
    text_loc_inter = text_cos * loc_j

    feature_vector = [
        text_cos, title_j, desc_j, loc_j, date_diff, desc_len_diff, text_color_inter, text_loc_inter
    ]
    return np.array(feature_vector, dtype=float)

# ---------- UI ----------
st.set_page_config(page_title="Campus Lost & Found — AutoMatch", layout="wide")
st.title("Campus Lost & Found — AutoMatch (TF-IDF + Logistic Regression)")

# Load artifacts and pool
tfidf, svd, scaler, clf = load_artifacts()
found_pool = build_found_pool()
st.sidebar.header("Settings")
candidate_k = st.sidebar.slider("Candidate pool K (TF-IDF retrieval)", min_value=10, max_value=500, value=CANDIDATE_K_DEFAULT, step=10)
top_k_show = st.sidebar.slider("Show top-K matches", min_value=1, max_value=10, value=TOP_K_SHOW)
st.sidebar.markdown("Model artifacts loaded from:")
st.sidebar.text(MODEL_DIR)

mode = st.radio("Action", ["I lost an item — find matched found items", "I found an item — find potential owners"])

col1, col2 = st.columns([2,1])
with col1:
    st.subheader("Describe the item")
    title = st.text_input("Title (e.g. 'blue water bottle')", "")
    desc = st.text_area("Description (short)", "")
    st.markdown("Optional metadata (helps ranking):")
    location = st.text_input("Location (e.g. Library, Canteen)", "")
    color = st.text_input("Color (single word)", "")
    brand = st.text_input("Brand (if known)", "")
    date_str = st.date_input("Date reported (approx)", value=datetime.now()).strftime("%Y-%m-%d")
    search_btn = st.button("Find matches")

with col2:
    st.subheader("Quick sample found items")
    st.write("Found-pool size:", len(found_pool))
    if len(found_pool) > 0:
        sample = st.selectbox("Pick a sample found item to preview", options=found_pool['found_id'].tolist()[:200] if len(found_pool)>200 else found_pool['found_id'].tolist())
        if sample:
            row = found_pool[found_pool['found_id']==str(sample)].iloc[0]
            st.markdown("**Title:** " + str(row.get('found_title','')))
            st.markdown("**Description:** " + str(row.get('found_description',''))[:300])

st.markdown("---")
st.write("Match results and feedback will be saved locally to:", FEEDBACK_LOG)

# When user clicks search
if search_btn:
    if not title and not desc:
        st.warning("Please enter a title or description to search.")
    else:
        lost_row = {
            'title': title,
            'description': desc,
            'date': date_str,
            'location': location,
            'color': color,
            'brand': brand
        }
        # retrieve tfidf top-K
        lost_text = (title + " " + desc).strip()
        lost_vec_tfidf = tfidf.transform([lost_text])
        if svd is not None:
            lost_vec = svd.transform(lost_vec_tfidf)
            found_vecs = svd.transform(tfidf.transform(found_pool['found_text_full'].tolist()))
        else:
            lost_vec = lost_vec_tfidf.toarray()
            # convert to array if sparse
            found_vecs = tfidf.transform(found_pool['found_text_full'].tolist())
            if hasattr(found_vecs, "toarray"):
                found_vecs = found_vecs.toarray()

        sims = cosine_similarity(lost_vec, found_vecs)[0]
        topk_idx = np.argsort(sims)[::-1][:candidate_k]
        candidates = found_pool.iloc[topk_idx].copy().reset_index(drop=True)
        candidates['tfidf_sim'] = sims[topk_idx]

        # build pair features and score with LR (robust to NaNs)
        feat_rows = []
        for _, frow in candidates.iterrows():
            fv = compute_pair_features(lost_row, frow, tfidf, svd)
            feat_rows.append(fv)

        # handle case with zero candidates
        if len(feat_rows) == 0:
            st.info("No candidate found items to score.")
        else:
            # Build DataFrame with exact feature column order
            feats_df = pd.DataFrame(feat_rows, columns=FEATURE_COLUMNS)

            # Impute missing values: first attempt fill with 0.0
            feats_df = feats_df.fillna(0.0).astype(float)

            # Scale and predict, with a safe fallback to median imputation if sklearn still complains
            try:
                feats_scaled = scaler.transform(feats_df.values)
                scores = clf.predict_proba(feats_scaled)[:,1]
            except Exception as e:
                # fallback: median impute then predict
                med = feats_df.median()
                feats_df = feats_df.fillna(med).astype(float)
                feats_scaled = scaler.transform(feats_df.values)
                scores = clf.predict_proba(feats_scaled)[:,1]

            candidates['score'] = scores

            # show top-k_show
            results = candidates.sort_values(by='score', ascending=False).head(top_k_show).reset_index(drop=True)

            st.success(f"Top {top_k_show} matches (ranked using LR scorer):")
            for i, r in results.iterrows():
                st.markdown(f"### Rank {i+1} — score: {r['score']:.3f} (TF-IDF sim {r['tfidf_sim']:.3f})")
                st.markdown(f"**Title:** {r.get('found_title','')}")
                st.markdown(f"**Description:** {r.get('found_description','')}")
                st.markdown(f"**Location:** {r.get('found_location','')}, Color: {r.get('found_color','')}, Brand: {r.get('found_brand','')}")
                # Interpretability: show contribution (approx) using linear model coef * feat
                try:
                    fv = compute_pair_features(lost_row, r, tfidf, svd)
                    coef = clf.coef_[0]
                    # align with FEATURE_COLUMNS length
                    contrib = coef[:len(FEATURE_COLUMNS)] * fv[:len(FEATURE_COLUMNS)]
                    contrib_df = pd.DataFrame({'feature': FEATURE_COLUMNS, 'contrib': contrib})
                    st.table(contrib_df.sort_values(by='contrib', ascending=False).head(6))
                except Exception:
                    pass

                # feedback buttons with unique keys
                colA, colB = st.columns([1,1])
                with colA:
                    key_confirm = f"confirm_{i}_{r['found_id']}_{int(datetime.now().timestamp())}"
                    if st.button(f"Confirm match — rank{i}_id{r['found_id']}", key=key_confirm):
                        fb = {
                            'timestamp': datetime.now().isoformat(),
                            'mode': mode,
                            'lost_title': title,
                            'lost_description': desc,
                            'found_id': r['found_id'],
                            'found_title': r.get('found_title',''),
                            'score': float(r['score']),
                            'label': 1
                        }
                        if os.path.exists(FEEDBACK_LOG):
                            df_fb = pd.read_csv(FEEDBACK_LOG)
                            df_fb = pd.concat([df_fb, pd.DataFrame([fb])], ignore_index=True)
                        else:
                            df_fb = pd.DataFrame([fb])
                        df_fb.to_csv(FEEDBACK_LOG, index=False)
                        st.success("Thanks — confirmed. Feedback saved.")
                with colB:
                    key_reject = f"reject_{i}_{r['found_id']}_{int(datetime.now().timestamp())}"
                    if st.button(f"Reject match — rank{i}_id{r['found_id']}", key=key_reject):
                        fb = {
                            'timestamp': datetime.now().isoformat(),
                            'mode': mode,
                            'lost_title': title,
                            'lost_description': desc,
                            'found_id': r['found_id'],
                            'found_title': r.get('found_title',''),
                            'score': float(r['score']),
                            'label': 0
                        }
                        if os.path.exists(FEEDBACK_LOG):
                            df_fb = pd.read_csv(FEEDBACK_LOG)
                            df_fb = pd.concat([df_fb, pd.DataFrame([fb])], ignore_index=True)
                        else:
                            df_fb = pd.DataFrame([fb])
                        df_fb.to_csv(FEEDBACK_LOG, index=False)
                        st.info("Rejected — feedback saved.")

            st.markdown("---")
            st.write("You can adjust candidate K in the sidebar (bigger K = better recall, slower).")

