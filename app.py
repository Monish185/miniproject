# app.py
import os
import re
import json
import pickle
import random
from datetime import datetime

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

import numpy as np
import pandas as pd

# ML libs
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import praw  # 🔥 official reddit api

# ---------------------------
# Reddit API Credentials
# ---------------------------
reddit = praw.Reddit(
    client_id="LAv6QlsTL5m9DyLAukLoVg",         # <---- fill this
    client_secret="gmfpVyBssoBO5h5F-1OMKLsg0Lyk3Q", # <---- fill this
    user_agent="capsulecnn app by u/Most-Car4403"   # <---- fill this 
)

# ---------------------------
# Config
# ---------------------------
MAX_VOCAB = 50000
MAX_LEN = 300
ISOT_MODEL_PATH = "isot_capsule_final.h5"
ISOT_TOKENIZER_PATH = "isot_tokenizer.pkl"
TFIDF_MODEL_PATH = "tfidf_fallback.pkl"
TRUE_CSV = "True.csv"
FAKE_CSV = "Fake.csv"

# ---------------------------
# Clean text
# ---------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# ---------------------------
# Capsule Layer Definition
# ---------------------------
from tensorflow.keras import layers

class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsules=10, dim_capsule=16, routings=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsule = dim_capsule
        self.routings = routings

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=[input_shape[-1], self.num_capsules * self.dim_capsule],
            initializer="glorot_uniform",
            trainable=True,
        )

    def squash(self, s, axis=-1):
        s_norm = tf.norm(s, axis=axis, keepdims=True) + 1e-7
        return (s_norm**2 / (1 + s_norm**2)) * (s / s_norm)

    def call(self, inputs):
        u_hat = tf.matmul(inputs, self.W)
        u_hat = tf.reshape(
            u_hat, (-1, inputs.shape[1], self.num_capsules, self.dim_capsule)
        )
        b = tf.zeros_like(u_hat[..., 0])
        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=2)
            s = tf.reduce_sum(c[..., None] * u_hat, axis=1)
            v = self.squash(s)
            if i < self.routings - 1:
                b += tf.reduce_sum(u_hat * v[:, None, :, :], axis=-1)
        return v

# ---------------------------
# Global model vars
# ---------------------------
keras_model = None
tokenizer = None
fallback_model = None

def load_keras_model_and_tokenizer():
    global keras_model, tokenizer

    if keras_model is not None and tokenizer is not None:
        return True

    if os.path.exists(ISOT_MODEL_PATH) and os.path.exists(ISOT_TOKENIZER_PATH):
        try:
            keras_model = tf.keras.models.load_model(
                ISOT_MODEL_PATH, custom_objects={"CapsuleLayer": CapsuleLayer}
            )
            with open(ISOT_TOKENIZER_PATH, "rb") as f:
                tokenizer = pickle.load(f)
            print("Loaded Capsule model + Tokenizer.")
            return True
        except Exception as e:
            print("Failed to load Capsule Model:", e)
    return False


def build_or_load_fallback():
    """TF-IDF fallback model"""
    global fallback_model
    if fallback_model:
        return True

    if os.path.exists(TFIDF_MODEL_PATH):
        with open(TFIDF_MODEL_PATH, "rb") as f:
            fallback_model = pickle.load(f)
        print("Loaded fallback TF-IDF model")
        return True

    if os.path.exists(TRUE_CSV) and os.path.exists(FAKE_CSV):
        try:
            tdf = pd.read_csv(TRUE_CSV)
            fdf = pd.read_csv(FAKE_CSV)
            tdf["label"] = 1
            fdf["label"] = 0

            df = pd.concat([tdf, fdf]).sample(frac=1, random_state=42)
            df["clean_text"] = (
                df["title"].fillna("") + " " + df["text"].fillna("")
            ).apply(clean_text)

            vect = TfidfVectorizer(max_features=20000)
            Xv = vect.fit_transform(df["clean_text"])
            clf = LogisticRegression(max_iter=300)
            clf.fit(Xv, df["label"])

            fallback_model = (vect, clf)
            with open(TFIDF_MODEL_PATH, "wb") as f:
                pickle.dump(fallback_model, f)

            print("Trained fallback model")
            return True
        except Exception as e:
            print("Fallback training error:", e)
            return False

    return False

# ---------------------------
# Prediction function
# ---------------------------
def predict_fake_or_real(text):
    text = clean_text(text)

    # 1) Try capsule model
    if load_keras_model_and_tokenizer():
        try:
            seq = tokenizer.texts_to_sequences([text])
            pad = pad_sequences(seq, maxlen=MAX_LEN)
            prob = keras_model.predict(pad)[0][0]
            return {
                "model": "capsule",
                "label": "real" if prob > 0.5 else "fake",
                "score": float(prob),
            }
        except:
            pass

    # 2) Fallback TF-IDF
    if build_or_load_fallback():
        vect, clf = fallback_model
        Xv = vect.transform([text])
        prob = clf.predict_proba(Xv)[0][1]
        return {
            "model": "tfidf",
            "label": "real" if prob > 0.5 else "fake",
            "score": float(prob),
        }

    # 3) Last fallback (keyword heuristic)
    if "fake" in text or "fraud" in text or "scam" in text:
        return {"model": "heuristic", "label": "fake", "score": 0.1}
    return {"model": "heuristic", "label": "real", "score": 0.9}

# ---------------------------
# Fetch Reddit Posts (REAL)
# ---------------------------
def fetch_top_posts_from_subreddit(subreddit="news", size=10):
    posts = []
    try:
        for post in reddit.subreddit(subreddit).hot(limit=size):
            posts.append(
                {
                    "id": post.id,
                    "title": post.title,
                    "text": post.selftext or "",
                    "url": post.url,
                    "created_utc": datetime.utcfromtimestamp(post.created_utc).isoformat(),
                }
            )
    except Exception as e:
        print("Reddit API error:", e)
    return posts

# ---------------------------
# Flask App
# ---------------------------
app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/topnews")
def api_topnews():
    sub = request.args.get("subreddit", "news")
    posts = fetch_top_posts_from_subreddit(subreddit=sub, size=20)
    if not posts:
        return jsonify({"success": False, "error": "Reddit fetch failed"})
    return jsonify({"success": True, "posts": random.sample(posts, 5)})

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.json
    text = (data.get("title", "") + " " + data.get("text", "")).strip()
    if not text:
        return jsonify({"success": False, "error": "No text"}), 400
    return jsonify({"success": True, "result": predict_fake_or_real(text)})

if __name__ == "__main__":
    build_or_load_fallback()
    print("Server running at http://127.0.0.1:5000")
    app.run(debug=True)
