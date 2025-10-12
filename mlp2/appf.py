# app.py
"""
Automatic Tagging of Educational Videos Using NLP
Single-file version (FastAPI backend + Streamlit frontend).
"""

import os
import tempfile
from typing import List, Dict, Any, Optional

import numpy as np
import torch
import whisper
import spacy
from transformers import AutoTokenizer, AutoModel
from bertopic import BERTopic
from umap import UMAP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from xgboost import XGBClassifier

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Streamlit and requests are UI-specific; import them lazily inside run_streamlit_ui()
from keyword_matcher import KeywordMatcher


# ===================== CONFIG =====================
WHISPER_MODEL = "tiny"   # tiny | base | small | medium | large
TRANSFORMER_MODEL = "distilbert-base-uncased"
MAX_LEN = 512
DIFFICULTY_LABELS = ["Beginner", "Intermediate", "Advanced"]
XGB_MODEL_PATH = ".cache/difficulty_xgb_v4.json"

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: List[str]) -> np.ndarray:
        tokens = self.tokenizer(texts, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        outputs = self.model(**tokens)
        last_hidden = outputs.last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1)
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        mean_pooled = summed / counts
        return mean_pooled.cpu().numpy()


# ===================== NLP PIPELINE =====================
class NLPPipeline:
    def __init__(self):
        self.asr = whisper.load_model(WHISPER_MODEL)
        self.nlp = spacy.load("en_core_web_sm")
        self.embedder = Embedder(TRANSFORMER_MODEL)
        self.topic_model = BERTopic(embedding_model=self.embedder, calculate_probabilities=True, min_topic_size=2, umap_model=UMAP(n_neighbors=2))
        # Pre-fit BERTopic with dummy data to avoid re-fitting per text
        dummy_texts = [
            "Introduction to machine learning basics.",
            "Advanced algorithms in data science.",
            "Simple programming concepts for beginners.",
            "Complex mathematical derivations.",
            "Basic algebra and geometry.",
            "Statistics and probability for data analysis.",
            "Neural networks and deep learning.",
            "Database management systems.",
            "Web development with HTML and CSS.",
            "Cybersecurity fundamentals.",
            "Quantum computing principles.",
            "Blockchain technology explained.",
            "Artificial intelligence ethics.",
            "Cloud computing services.",
            "Mobile app development."
        ]
        dummy_embeddings = self.embedder.encode(dummy_texts)
        self.topic_model.fit(dummy_texts, dummy_embeddings)

    def transcribe(self, file_path: str) -> Dict[str, Any]:
        result = self.asr.transcribe(file_path)
        return {
            "text": result.get("text", "").strip(),
            "language": result.get("language", None),
            "segments": result.get("segments", []),
        }

    def analyze_text(self, text: str) -> Dict[str, Any]:
        if not text.strip():
            return {
                "keyphrases": [],
                "entities": [],
                "topics": [],
                "stats": {"num_tokens": 0, "num_sentences": 0, "avg_sentence_len": 0, "reading_time_min": 0}
            }

        doc = self.nlp(text)
        keyphrases = list({chunk.text.strip().lower() for chunk in doc.noun_chunks if len(chunk.text.strip()) > 2})
        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

        top_topics = []
        try:
            chunks = chunk_text(text, max_tokens=120)
            if not chunks:
                chunks = [text]  # fallback
            embeddings = self.embedder.encode(chunks)
            topics, _ = self.topic_model.transform(chunks, embeddings)
            topic_info = self.topic_model.get_topic_info()

            for _, row in topic_info.iterrows():
                if row["Topic"] == -1:
                    continue
                words = self.topic_model.get_topic(row["Topic"])[:5]
                top_topics.append({
                    "topic_id": int(row["Topic"]),
                    "size": int(row["Count"]),
                    "repr": [w for w, _ in words],
                })
        except Exception:
            top_topics = []

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
        self.scaler = MaxAbsScaler()
        self.model = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
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
        # Manually label the seed texts: 0=Beginner, 1=Intermediate, 2=Advanced
        y = np.array([0, 1, 2, 0, 2, 1, 0, 2, 0, 2, 2, 2, 2, 1, 1, 1])
        X = self.vectorizer.fit_transform(texts)
        X = self.scaler.fit_transform(X)
        self.model.fit(X, y)
        self._is_fit = True
        self._save()

    def predict(self, text: str) -> Dict:
        if not text.strip():
            return {
                "label": DIFFICULTY_LABELS[0],  # Beginner
                "proba": {l: 1.0 if i == 0 else 0.0 for i, l in enumerate(DIFFICULTY_LABELS)}
            }
        if not self._is_fit:
            seed_texts = [
                "Introduction to variables and data types in Python with examples.",
                "Understanding gradient descent and its variants.",
                "Convex optimization with KKT conditions and proofs, min f(x) s.t. g(x)=0, h(x)<=0.",
                "Basic algebra practice with fractions and percentages.",
                "Neural network backpropagation derivations with calculus, ∂L/∂w = ∑ δ * a.",
                "Sorting algorithms overview: bubble, merge, quicksort.",
                "What is a variable in programming?",
                "Advanced machine learning techniques for big data.",
                "Simple math addition and subtraction.",
                "The Schrödinger equation iħ ∂ψ/∂t = Hψ governs quantum dynamics, with H = -ħ²/2m ∇² + V.",
                "In general relativity, the Einstein field equations G_μν = 8πG T_μν describe spacetime curvature.",
                "Quantum field theory involves Feynman diagrams for particle interactions, ∫ Dφ e^{iS[φ]}.",
                "Advanced topology: manifolds, homotopy groups π_n(X), and cohomology H^*(X; Z).",
                "Machine learning algorithms include linear regression, decision trees, and neural networks. Linear regression models the relationship between variables using a line. Decision trees split data based on features. Neural networks consist of layers of nodes.",
                "Object-oriented programming in Java involves classes, objects, inheritance, and polymorphism. A class is a blueprint for objects. Inheritance allows a class to inherit properties from another. Polymorphism enables methods to behave differently based on the object.",
                "Database normalization reduces redundancy. First normal form eliminates repeating groups. Second normal form removes partial dependencies. Third normal form removes transitive dependencies."
            ]
            self._bootstrap_train(seed_texts)
        X = self.vectorizer.transform([text])
        if X.shape[1] == 0:
            return {
                "label": DIFFICULTY_LABELS[0],
                "proba": {l: 1.0 if i == 0 else 0.0 for i, l in enumerate(DIFFICULTY_LABELS)}
            }
        X = self.scaler.transform(X)
        # Protect against cases where predict_proba returns an empty/zero-size array
        try:
            proba = self.model.predict_proba(X)[0]
        except Exception:
            # If prediction fails for any reason, fall back to uniform probabilities
            proba = np.ones(len(DIFFICULTY_LABELS), dtype=float) / float(len(DIFFICULTY_LABELS))

        # If proba is zero-size (shape (0,) or similar), use a safe uniform fallback
        if getattr(proba, "size", None) == 0 or len(proba) == 0:
            proba = np.ones(len(DIFFICULTY_LABELS), dtype=float) / float(len(DIFFICULTY_LABELS))

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
# CSV keyword matcher (loads auto_tags_college_expanded.csv from repository root)
kw_matcher = KeywordMatcher()

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
        # Attach CSV match info
        csv_match = kw_matcher.match(inp.text)
        return {**analysis, "difficulty": difficulty, "csv_match": csv_match}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process")
async def process(file: Optional[UploadFile] = File(None)):
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
        csv_match = kw_matcher.match(transcript)
        return {"transcript": transcript, **analysis, "difficulty": difficulty, "csv_match": csv_match}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===================== FRONTEND (Streamlit) =====================
def run_streamlit_ui():
    import streamlit as st
    import requests

    st.set_page_config(page_title="EduVideo Tagger", page_icon="🎓", layout="wide")
    st.title("🎓 Automatic Tagging of Educational Videos")
    st.caption("Whisper + Transformers + spaCy + BERTopic + XGBoost | Streamlit UI")

    api_base = st.sidebar.text_input("API Base URL", value="http://localhost:8001")
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
            # Show CSV-based keyword match if available
            st.subheader("Matched Tag (from CSV)")
            csvm = result.get("csv_match")
            if csvm and csvm.get("chosen"):
                c = csvm["chosen"]
                st.write(f"Keyword: {c.get('keyword', '')}")
                st.write(f"Subject: {c.get('subject', '')}")
                st.write(f"Difficulty: {c.get('difficulty', '').title()}")
            else:
                st.write("No matching keyword found in CSV.")
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
        uvicorn.run("appf:app", host="0.0.0.0", port=8001, reload=False)
    else:
        run_streamlit_ui()
