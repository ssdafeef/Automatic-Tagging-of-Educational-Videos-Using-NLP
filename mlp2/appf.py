# app.py
"""
Automatic Tagging of Educational Videos Using NLP
Single-file version (FastAPI backend + Streamlit frontend).
"""

import os
import tempfile
from typing import List, Dict, Any

import numpy as np
import torch
import whisper
import spacy
from transformers import AutoTokenizer, AutoModel
from bertopic import BERTopic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

import streamlit as st
import requests


# ===================== CONFIG =====================
WHISPER_MODEL = "tiny"   # tiny | base | small | medium | large
TRANSFORMER_MODEL = "distilbert-base-uncased"
MAX_LEN = 512
DIFFICULTY_LABELS = ["Beginner", "Intermediate", "Advanced"]
XGB_MODEL_PATH = ".cache/difficulty_xgb.json"

os.makedirs(".cache", exist_ok=True)


# ===================== UTILS =====================
def chunk_text(text: str, max_tokens: int = 200) -> List[str]:
    words = text.split()
    chunks, cur = [], []
    for w in words:
        cur.append(w)
        if len(cur) >= max_tokens:
            chunks.append(" ".join(cur))
            cur = []
    if cur:
        chunks.append(" ".join(cur))
    return chunks


# ===================== EMBEDDINGS =====================
class Embedder:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: List[str]) -> np.ndarray:
        all_embeds = []
        for t in texts:
            tokens = self.tokenizer(t, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            outputs = self.model(**tokens)
            last_hidden = outputs.last_hidden_state
            mask = tokens.attention_mask.unsqueeze(-1)
            summed = (last_hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            mean_pooled = (summed / counts).squeeze(0).cpu().numpy()
            all_embeds.append(mean_pooled)
        return np.vstack(all_embeds)


# ===================== NLP PIPELINE =====================
class NLPPipeline:
    def __init__(self):
        self.asr = whisper.load_model(WHISPER_MODEL)
        self.nlp = spacy.load("en_core_web_sm")
        self.embedder = Embedder(TRANSFORMER_MODEL)
        self.topic_model = BERTopic(embedding_model=self.embedder, calculate_probabilities=True)

    def transcribe(self, file_path: str) -> Dict[str, Any]:
        result = self.asr.transcribe(file_path)
        return {
            "text": result.get("text", "").strip(),
            "language": result.get("language", None),
            "segments": result.get("segments", []),
        }

    def analyze_text(self, text: str) -> Dict[str, Any]:
        doc = self.nlp(text)
        keyphrases = list({chunk.text.strip().lower() for chunk in doc.noun_chunks if len(chunk.text.strip()) > 2})
        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

        chunks = chunk_text(text, max_tokens=120)
        embeddings = self.embedder.encode(chunks)
        topics, _ = self.topic_model.fit_transform(chunks, embeddings)
        topic_info = self.topic_model.get_topic_info()

        top_topics = []
        for _, row in topic_info.iterrows():
            if row["Topic"] == -1:
                continue
            words = self.topic_model.get_topic(row["Topic"])[:5]
            top_topics.append({
                "topic_id": int(row["Topic"]),
                "size": int(row["Count"]),
                "repr": [w for w, _ in words],
            })

        stats = {
            "num_tokens": len(text.split()),
            "num_sentences": max(1, text.count(".") + text.count("!") + text.count("?")),
            "avg_sentence_len": round(len(text.split()) / max(1, (text.count(".") + text.count("!") + text.count("?"))), 2),
            "reading_time_min": round(len(text.split()) / 200.0, 2),
        }

        return {
            "keyphrases": keyphrases[:50],
            "entities": entities[:50],
            "topics": top_topics[:5],
            "stats": stats,
        }


# ===================== DIFFICULTY =====================
class DifficultyModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1,2))
        self.scaler = StandardScaler(with_mean=False)
        self.model = XGBClassifier(
            n_estimators=60, max_depth=4, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9,
            objective="multi:softprob", num_class=3,
            tree_method="hist", verbosity=0
        )
        self._is_fit = False
        self._maybe_load()

    def _maybe_load(self):
        if os.path.exists(XGB_MODEL_PATH):
            try:
                self.model.load_model(XGB_MODEL_PATH)
                self._is_fit = True
            except Exception:
                self._is_fit = False

    def _save(self):
        self.model.save_model(XGB_MODEL_PATH)

    def _bootstrap_train(self, texts):
        lengths = np.array([len(t.split()) for t in texts])
        tech = np.array([sum(c.isdigit() or c in {"/","-","=","+"} for c in t) for t in texts])
        scores = lengths * 0.6 + tech * 0.4
        thr1, thr2 = np.quantile(scores, [0.33, 0.66])
        y = np.digitize(scores, [thr1, thr2])
        X = self.vectorizer.fit_transform(texts)
        X = self.scaler.fit_transform(X)
        self.model.fit(X, y)
        self._is_fit = True
        self._save()

    def predict(self, text: str) -> Dict:
        if not self._is_fit:
            seed_texts = [
                "Introduction to variables and data types in Python with examples.",
                "Understanding gradient descent and its variants.",
                "Convex optimization with KKT conditions and proofs.",
                "Basic algebra practice with fractions and percentages.",
                "Neural network backpropagation derivations with calculus.",
                "Sorting algorithms overview: bubble, merge, quicksort."
            ]
            self._bootstrap_train(seed_texts)
        X = self.vectorizer.transform([text])
        X = self.scaler.transform(X)
        proba = self.model.predict_proba(X)[0]
        label = int(np.argmax(proba))
        return {
            "label": DIFFICULTY_LABELS[label],
            "proba": {DIFFICULTY_LABELS[i]: float(p) for i, p in enumerate(proba)}
        }


# ===================== BACKEND (FastAPI) =====================
app = FastAPI(title="EduVideo Tagger API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

pipeline = NLPPipeline()
diff_model = DifficultyModel()

class TextIn(BaseModel):
    text: str

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        suffix = os.path.splitext(file.filename)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        result = pipeline.transcribe(tmp_path)
        os.unlink(tmp_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tag")
async def tag_text(inp: TextIn):
    if not inp.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")
    try:
        analysis = pipeline.analyze_text(inp.text)
        difficulty = diff_model.predict(inp.text)
        return {**analysis, "difficulty": difficulty}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process")
async def process(file: UploadFile = File(None)):
    try:
        transcript = ""
        if file is not None:
            suffix = os.path.splitext(file.filename)[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name
            result = pipeline.transcribe(tmp_path)
            os.unlink(tmp_path)
            transcript = result.get("text", "").strip()
        else:
            raise HTTPException(status_code=400, detail="Upload a file or use /tag with text")
        analysis = pipeline.analyze_text(transcript)
        difficulty = diff_model.predict(transcript)
        return {"transcript": transcript, **analysis, "difficulty": difficulty}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===================== FRONTEND (Streamlit) =====================
def run_streamlit_ui():
    st.set_page_config(page_title="EduVideo Tagger", page_icon="🎓", layout="wide")
    st.title("🎓 Automatic Tagging of Educational Videos")
    st.caption("Whisper + Transformers + spaCy + BERTopic + XGBoost | Streamlit UI")

    api_base = st.sidebar.text_input("API Base URL", value="http://localhost:8000")
    mode = st.sidebar.radio("Mode", ["Upload & Process", "Paste Transcript -> Tag"], index=0)

    cols = st.columns([1,1.2,0.8])

    if mode == "Upload & Process":
        with cols[0]:
            st.subheader("Upload video/audio")
            media = st.file_uploader("File (.mp3/.wav/.mp4)", type=["mp3","wav","mp4","m4a","aac","flac","ogg"])
            if st.button("Process", use_container_width=True) and media is not None:
                with st.spinner("Transcribing and tagging..."):
                    files = {"file": (media.name, media.getvalue(), media.type)}
                    resp = requests.post(f"{api_base}/process", files=files, timeout=600)
                    if resp.ok:
                        st.session_state["last_result"] = resp.json()
                    else:
                        st.error(resp.text)
    else:
        with cols[0]:
            st.subheader("Paste transcript")
            text = st.text_area("Transcript text", height=260)
            if st.button("Tag Transcript", use_container_width=True) and text.strip():
                with st.spinner("Analyzing text..."):
                    resp = requests.post(f"{api_base}/tag", json={"text": text}, timeout=300)
                    if resp.ok:
                        data = resp.json()
                        data["transcript"] = text
                        st.session_state["last_result"] = data
                    else:
                        st.error(resp.text)

    result = st.session_state.get("last_result")
    if result:
        with cols[1]:
            st.subheader("Transcript")
            st.write(result.get("transcript", "(from upload)"))
            stats = result.get("stats", {})
            if stats: st.json(stats)
        with cols[2]:
            st.subheader("Difficulty")
            diff = result.get("difficulty", {})
            if diff:
                st.metric("Predicted Level", diff.get("label", "?"))
                st.json(diff.get("proba", {}))
            st.subheader("Topics")
            for t in result.get("topics", []):
                st.markdown(f"**Topic {t['topic_id']}** — size {t['size']}<br>• " + ", ".join(t["repr"]), unsafe_allow_html=True)
            st.subheader("Keyphrases")
            kp = result.get("keyphrases", [])
            if kp: st.write(", ".join(kp[:100]))
            st.subheader("Entities")
            ents = result.get("entities", [])
            if ents: st.json(ents)


# ===================== ENTRY =====================
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
    else:
        run_streamlit_ui()
