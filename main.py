

import io 
import json 
import logging 
import os 
import asyncio 
import tempfile 
from contextlib import asynccontextmanager 
import torch 
import torchvision.transforms as T 
from fastapi import FastAPI, File, UploadFile, HTTPException 
from fastapi.middleware.cors import CORSMiddleware 
from huggingface_hub import hf_hub_download 
from PIL import Image 
from pydantic import BaseModel 
logging.basicConfig( 
level=logging.INFO, 
format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", 
) 
logger = logging.getLogger("yaqeen") 
HF_REPO = "H831A/Yaqeen.models" 
HF_TOKEN = os.environ.get("HF_TOKEN") 
MODEL_DIR = "./cached_model" 
LABEL_MAPPING: dict[int, str] = {0: "fake", 1: "real"} 
device = torch.device("cpu") 
logger.info("Yaqeen API v3.0 — starting") 
logger.info("Inference device: %s", device) 
import re as _re 
def _preprocess_arabic(text: str) -> str: 
"""Same normalization used during model training.""" 
if not isinstance(text, str): 
return "" 
text = _re.sub(r"http\S+", "", text) 
text = _re.sub("[إأآا]", "ا", text) 
text = _re.sub("ة", "ه", text) 
text = _re.sub("ى", "ي", text) 
text = _re.sub("[ؤئ]", "ء", text) 
text = _re.sub(r"[\u064B-\u065F\u0640]", "", text) 
text = _re.sub(r"[^\u0600-\u06FF\w\s،؟!.]", "", text) 
text = _re.sub(r"\s+", " ", text).strip() 
return text 
TRUSTED_SOURCE_KEYWORDS: dict[str, list[str]] = {} 
def _build_trusted_source_keywords(knowledge_base: list[dict]) -> dict[str, list[str]]: 
mapping: dict[str, list[str]] = {} 
for entry in knowledge_base: 
if str(entry.get("credibility", "")).strip().lower() != "true": 
continue 
source = entry.get("source", "").strip() 
# Use pre-stored preprocessed keywords if available, else compute on the fly 
raw_kws = entry.get("keywords_preprocessed") or entry.get("keywords", []) 
keywords = [_preprocess_arabic(kw).strip() if not entry.get("keywords_preprocessed") else kw.strip() 
for kw in raw_kws if str(kw).strip()] 
if source and keywords: 
mapping.setdefault(source, []) 
for kw in keywords: 
if kw and kw not in mapping[source]: 
mapping[source].append(kw) 
logger.info( 
"Trusted sources loaded: %d sources, %d keywords total", 
len(mapping), 
sum(len(v) for v in mapping.values()), 
) 
return mapping 
_DEATH_CLAIMS = [ 
"وفاة", "توفي", "وفى", "لقي حتفه", "مقتل", "اغتيال", 
] 
_HIGH_PROFILE = [ 
"ولي العهد", "الأمير", "الملك", "الرئيس", 
"محمد بن سلمان", "محمد بن زايد", "بن سلمان", "بن زايد", 
"الوزير", "الأمين العام", "رئيس الوزراء", 
] 
_NEGATIVE_CONTEXT = [ 
"انقلاب", "خيانة", "فضيحة", 
"يكذب", "كذب", "ملفق", "مزيف", "شائعة", 
] 
def detect_trusted_source(text: str) -> str | None: 
# Check on ORIGINAL text for death/VIP/negative patterns (human-readable) 
is_death_of_vip = ( 
any(w in text for w in _DEATH_CLAIMS) and 
any(w in text for w in _HIGH_PROFILE) 
) 
if is_death_of_vip: 
logger.info("Death-of-VIP claim detected — bypassing trusted source shortcut") 
return None 
if any(neg in text for neg in _NEGATIVE_CONTEXT): 
return None 
# Match keywords on PREPROCESSED text (same normalization as training) 
proc_text = _preprocess_arabic(text) 
for source, keywords in TRUSTED_SOURCE_KEYWORDS.items(): 
if any(kw and kw in proc_text for kw in keywords): 
logger.info("Trusted source matched: %s", source) 
return source 
return None 
_tokenizer = None 
_text_model = None 
_embedding_model = None 
_faiss_index = None 
_metadata = None 
_xception = None 
_mtcnn_detector = None 
def load_text_models(): 
global _tokenizer, _text_model, LABEL_MAPPING 
if _tokenizer is not None: 
return _tokenizer, _text_model 
from transformers import AutoTokenizer, AutoModelForSequenceClassification 
logger.info("Loading news classification model ...") 
os.makedirs(MODEL_DIR, exist_ok=True) 
required_files = [ 
"config.json", 
"model.safetensors", 
"tokenizer.json", 
"tokenizer_config.json", 
"special_tokens_map.json", 
"vocab.txt", 
] 
for filename in required_files: 
dest = os.path.join(MODEL_DIR, filename) 
if os.path.exists(dest): 
logger.info("Already cached, skipping: %s", filename) 
continue 
try: 
hf_hub_download( 
repo_id=HF_REPO, 
filename=filename, 
token=HF_TOKEN, 
local_dir=MODEL_DIR, 
) 
logger.info("Downloaded: %s", filename) 
except Exception as exc: 
logger.warning("Failed to download %s: %s", filename, exc) 
_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR) 
_text_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR) 
_text_model.eval() 
if _text_model.config.id2label: 
loaded_labels = { 
int(k): v for k, v in _text_model.config.id2label.items() 
} 
label_values = list(loaded_labels.values()) 
known_labels = ("fake", "false", "fabricated", "real", "authentic", "true") 
if any(v.lower() in known_labels for v in label_values): 
LABEL_MAPPING = loaded_labels 
logger.info("Loaded label mapping from model config: %s", LABEL_MAPPING) 
else: 
logger.warning( 
"Unexpected id2label %s — keeping default mapping", 
label_values, 
) 
logger.info("News model ready. Label mapping: %s", LABEL_MAPPING) 
return _tokenizer, _text_model 
def load_faiss(): 
global _faiss_index, _metadata, TRUSTED_SOURCE_KEYWORDS 
if _faiss_index is not None: 
return _faiss_index, _metadata 
import faiss 
logger.info("Loading FAISS index ...") 
faiss_path = hf_hub_download( 
repo_id=HF_REPO, filename="faiss_index.bin", token=HF_TOKEN 
) 
_faiss_index = faiss.read_index(faiss_path) 
kb_path = hf_hub_download( 
repo_id=HF_REPO, filename="knowledge_base_yaqeen_v3.json", token=HF_TOKEN 
) 
with open(kb_path, "r", encoding="utf-8") as fh: 
knowledge_base = json.load(fh) 
_metadata = [ 
{ 
"source": entry.get("source", ""), 
"credibility": entry.get("credibility", ""), 
"credibility_score": float(entry.get("credibility_score", 0.0)), 
"url": entry.get("url", ""), 
"summary": entry.get("summary", ""), 
"category": entry.get("category", ""), 
"country": entry.get("country", ""), 
} 
for entry in knowledge_base 
] 
TRUSTED_SOURCE_KEYWORDS = _build_trusted_source_keywords(knowledge_base) 
logger.info("FAISS index ready — %d vectors", _faiss_index.ntotal) 
return _faiss_index, _metadata 
def load_embedding_model(): 
global _embedding_model 
if _embedding_model is not None: 
return _embedding_model 
from sentence_transformers import SentenceTransformer 
logger.info("Loading multilingual embedding model ...") 
_embedding_model = SentenceTransformer( 
"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" 
) 
logger.info("Embedding model ready") 
return _embedding_model 
def get_mtcnn_detector(): 
global _mtcnn_detector 
if _mtcnn_detector is not None: 
return _mtcnn_detector 
try: 
from facenet_pytorch import MTCNN 
_mtcnn_detector = MTCNN(keep_all=False, device=device) 
logger.info("MTCNN ready (facenet_pytorch)") 
except ImportError: 
logger.warning( 
"facenet_pytorch not installed — full-image analysis will be used" 
) 
_mtcnn_detector = None 
return _mtcnn_detector 
def crop_face_mtcnn(pil_image: Image.Image) -> torch.Tensor | None: 
detector = get_mtcnn_detector() 
if detector is None: 
return None 
try: 
face_tensor = detector(pil_image) 
if face_tensor is None: 
logger.info("MTCNN: no face detected") 
return None 
logger.info("MTCNN: face cropped successfully") 
return face_tensor 
except Exception as exc: 
logger.warning("MTCNN face crop failed: %s", exc) 
return None 
def load_xception(): 
global _xception 
if _xception is not None: 
return _xception 
import timm 
logger.info("Loading Xception model (timm legacy_xception) ...") 
weights_path = hf_hub_download( 
repo_id=HF_REPO, filename="best_xception.pth", token=HF_TOKEN 
) 
_xception = timm.create_model("legacy_xception", pretrained=False, num_classes=2) 
state_dict = torch.load(weights_path, map_location=device, weights_only=False) 
_xception.load_state_dict(state_dict, strict=True) 
_xception = _xception.to(device) 
_xception.eval() 
logger.info("Xception model ready (timm)") 
return _xception 
_eval_transform = T.Compose([ 
T.Resize((299, 299)), 
T.ToTensor(), 
T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), 
]) 
_face_tensor_resize = T.Resize((299, 299)) 
def _prepare_image_tensor(pil_image: Image.Image) -> tuple[torch.Tensor, bool]: 
face_tensor = crop_face_mtcnn(pil_image) 
if face_tensor is not None: 
resized = _face_tensor_resize(face_tensor) 
return resized.unsqueeze(0).to(device), True 
logger.info("Analysing full image (no face crop)") 
return _eval_transform(pil_image).unsqueeze(0).to(device), False 
def _run_xception(tensor: torch.Tensor) -> tuple[float, float]: 
model = load_xception() 
with torch.no_grad(): 
logits = model(tensor) 
probs = torch.softmax(logits, dim=1)[0] 
return float(probs[0]), float(probs[1]) 
def _build_image_response(p_fake: float, p_real: float, used_face: bool) -> dict: 
is_fabricated = p_fake >= 0.5 
logger.info( 
"Image classification: real=%.1f%% fake=%.1f%% result=%s face=%s", 
p_real * 100, p_fake * 100, 
"fake" if is_fabricated else "real", 
"yes" if used_face else "no", 
) 
confidence = round(max(p_real, p_fake) * 100, 1) 
if abs(p_fake - p_real) < 0.10: 
return { 
"is_fabricated": False, 
"verdict": "الصورة غير محسومة", 
"confidence": confidence, 
"analysis": ( 
f"درجة ثقة النموذج غير كافية للجزم بنتيجة. " 
f"احتمال التزوير: {round(p_fake * 100, 1)}% — " 
f"احتمال الأصالة: {round(p_real * 100, 1)}%. " 
f"يُوصى بالمراجعة اليدوية." 
), 
"indicators": ["نتيجة غير محسومة", "يُوصى بمراجعة بصرية متخصصة"], 
"probability_scores": { 
"authentic": round(p_real * 100, 1), 
"fabricated": round(p_fake * 100, 1), 
}, 
} 
if is_fabricated: 
indicators = [] 
if p_fake > 0.90: 
indicators.append("مؤشر تزوير رقمي عالٍ جداً") 
elif p_fake > 0.70: 
indicators.append("أنماط اصطناعية واضحة") 
else: 
indicators.append("مؤشرات أولية على التزوير") 
indicators.append("بصمة نموذج توليد اصطناعي محتملة") 
indicators.append("Xception: الصورة مزيّفة") 
if not used_face: 
indicators.append("تحليل بدون كروب وجه — دقة أقل") 
verdict = "الصورة مزيّفة" 
analysis = ( 
f"صنّف نموذج Xception هذه الصورة بوصفها مزيّفة رقمياً. " 
f"احتمال التزوير: {round(p_fake * 100, 1)}% — " 
f"احتمال الأصالة: {round(p_real * 100, 1)}%." 
) 
else: 
indicators = ["لا توجد أنماط تزوير مكتشفة", "Xception: الصورة حقيقية"] 
if p_real < 0.75: 
indicators.append("درجة ثقة متوسطة — يُوصى بالمراجعة البصرية") 
if not used_face: 
indicators.append("تحليل بدون كروب وجه — دقة أقل") 
verdict = "الصورة حقيقية" 
analysis = ( 
f"صنّف نموذج Xception هذه الصورة بوصفها حقيقية. " 
f"احتمال التزوير: {round(p_fake * 100, 1)}% — " 
f"احتمال الأصالة: {round(p_real * 100, 1)}%." 
) 
return { 
"is_fabricated": is_fabricated, 
"verdict": verdict, 
"confidence": confidence, 
"analysis": analysis, 
"indicators": indicators, 
"probability_scores": { 
"authentic": round(p_real * 100, 1), 
"fabricated": round(p_fake * 100, 1), 
}, 
} 
def _analyse_video_sync(video_path: str) -> dict: 
import cv2 
import numpy as np 
detector = get_mtcnn_detector() 
cap = cv2.VideoCapture(video_path) 
if not cap.isOpened(): 
raise ValueError("Could not open video file") 
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
fps = cap.get(cv2.CAP_PROP_FPS) 
duration = total_frames / fps if fps > 0 else 0 
logger.info( 
"Video: %d frames | %.1f FPS | %.1f seconds", 
total_frames, fps, duration, 
) 
fake_probs = [] 
frame_idx = 0 
sample_rate = 3 
model = load_xception() 
model.eval() 
with torch.no_grad(): 
while True: 
ret, frame = cap.read() 
if not ret: 
break 
if frame_idx % sample_rate == 0: 
pil_frame = Image.fromarray( 
cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
) 
if detector is not None: 
try: 
face_tensor = detector(pil_frame) 
except Exception: 
face_tensor = None 
else: 
face_tensor = None 
if face_tensor is not None: 
tensor = _face_tensor_resize(face_tensor).unsqueeze(0).to(device) 
else: 
tensor = _eval_transform(pil_frame).unsqueeze(0).to(device) 
probs = torch.softmax(model(tensor), dim=1) 
fake_prob = float(probs[0][0]) 
fake_probs.append(fake_prob) 
frame_idx += 1 
cap.release() 
if not fake_probs: 
return { 
"is_fabricated": False, 
"verdict": "تعذّر اكتشاف وجوه في الفيديو", 
"confidence": 0.0, 
"analysis": "لم يتمكن النموذج من اكتشاف وجوه واضحة في مقاطع الفيديو المُحللة.", 
"indicators": ["لم يُكتشف أي وجه في الفيديو"], 
"probability_scores": {"authentic": 0.0, "fabricated": 0.0}, 
"frames_analysed": 0, 
} 
avg_fake_prob = float(np.median(fake_probs)) 
is_fabricated = avg_fake_prob >= 0.5 
confidence = round( 
avg_fake_prob * 100 if is_fabricated else (1 - avg_fake_prob) * 100, 
1, 
) 
logger.info( 
"Video result: %d frames analysed | median fake=%.1f%% | verdict=%s", 
len(fake_probs), 
avg_fake_prob * 100, 
"fake" if is_fabricated else "real", 
) 
if is_fabricated: 
verdict = "الفيديو مزيّف" 
analysis = ( 
f"حلّل النموذج {len(fake_probs)} إطاراً من الفيديو. " 
f"بلغ متوسط احتمال التزوير {round(avg_fake_prob * 100, 1)}%، " 
f"مما يُشير إلى أن هذا الفيديو مُولَّد أو مُعدَّل رقمياً." 
) 
indicators = [ 
f"عدد الإطارات المُحللة: {len(fake_probs)}", 
f"متوسط احتمال التزوير: {round(avg_fake_prob * 100, 1)}%", 
"Xception: الفيديو مزيّف", 
] 
else: 
verdict = "الفيديو حقيقي" 
analysis = ( 
f"حلّل النموذج {len(fake_probs)} إطاراً من الفيديو. " 
f"بلغ متوسط احتمال التزوير {round(avg_fake_prob * 100, 1)}%، " 
f"مما يُشير إلى أن هذا الفيديو حقيقي." 
) 
indicators = [ 
f"عدد الإطارات المُحللة: {len(fake_probs)}", 
f"متوسط احتمال التزوير: {round(avg_fake_prob * 100, 1)}%", 
"Xception: الفيديو حقيقي", 
] 
return { 
"is_fabricated": is_fabricated, 
"verdict": verdict, 
"confidence": confidence, 
"analysis": analysis, 
"indicators": indicators, 
"probability_scores": { 
"authentic": round((1 - avg_fake_prob) * 100, 1), 
"fabricated": round(avg_fake_prob * 100, 1), 
}, 
"frames_analysed": len(fake_probs), 
} 
@asynccontextmanager 
async def lifespan(app: FastAPI): 
loop = asyncio.get_event_loop() 
async def warm(fn, name): 
try: 
await loop.run_in_executor(None, fn) 
logger.info("Ready: %s", name) 
except Exception as exc: 
logger.error("Failed to load %s: %s", name, exc) 
logger.info("=== Warming up models ===") 
await warm(load_text_models, "news model") 
await warm(load_embedding_model, "embedding model") 
await warm(load_faiss, "FAISS index") 
await warm(load_xception, "Xception model") 
await warm(get_mtcnn_detector, "MTCNN detector") 
logger.info("=== All models ready ===") 
yield 
logger.info("Shutting down ...") 
app = FastAPI( 
title="Yaqeen API", 
description=( 
"Automated fact-checking and deepfake detection API. " 
"Supports Arabic text classification via CAMeLBERT + FAISS " 
"and image/video authenticity analysis via Xception." 
), 
version="3.0", 
lifespan=lifespan, 
) 
app.add_middleware( 
CORSMiddleware, 
allow_origins=["*"], 
allow_methods=["GET", "POST", "OPTIONS"], 
allow_headers=["*"], 
max_age=3600, 
) 
@app.get("/health", tags=["system"]) 
def health_check(): 
return { 
"status": "operational", 
"version": "3.0", 
"models_loaded": { 
"text_classifier": _text_model is not None, 
"embedding_model": _embedding_model is not None, 
"faiss_index": _faiss_index is not None, 
"xception": _xception is not None, 
"mtcnn": _mtcnn_detector is not None, 
}, 
"inference_device": str(device), 
} 
class TextAnalysisRequest(BaseModel): 
text: str 
@app.post("/predict", tags=["text"]) 
async def predict_text(request: TextAnalysisRequest): 
if not request.text or len(request.text.strip()) < 10: 
raise HTTPException(status_code=400, detail="النص قصير جداً للتحليل.") 
import faiss as faiss_lib 
import numpy as np 
trusted_source = detect_trusted_source(request.text) 
if trusted_source: 
logger.info("Trusted source detected: %s", trusted_source) 
related_sources = [] 
try: 
emb_model = load_embedding_model() 
faiss_index, meta = load_faiss() 
q_vec = emb_model.encode([request.text], convert_to_numpy=True).astype("float32") 
faiss_lib.normalize_L2(q_vec) 
dists, idxs = faiss_index.search(q_vec, min(3, faiss_index.ntotal)) 
related_sources = [ 
{**meta[idx], "similarity": round(float(dists[0][r]), 4)} 
for r, idx in enumerate(idxs[0]) if idx != -1 
] 
except Exception: 
logger.exception("FAISS retrieval failed in trusted-source path") 
if not related_sources: 
related_sources = [{ 
"source": trusted_source, 
"url": "", 
"credibility": "Verified", 
"credibility_score": 1.0, 
"similarity": 1.0, 
}] 
return { 
"claim": request.text, 
"classification": "authentic", 
"verdict": "الخبر حقيقي", 
"confidence_score": 100.0, 
"verified_source": trusted_source, 
"related_sources": related_sources, 
} 
tokenizer, text_model = load_text_models() 
# Preprocess text exactly as done during training 
processed_text = _preprocess_arabic(request.text) 
logger.info("Preprocessed text: %s", processed_text[:100]) 
encoded = tokenizer( 
processed_text, 
return_tensors="pt", 
truncation=True, 
padding=True, 
max_length=384, 
) 
with torch.no_grad(): 
output = text_model(**encoded) 
probs = torch.softmax(output.logits, dim=1) 
pred = torch.argmax(probs, dim=1).item() 
raw_label = LABEL_MAPPING.get(pred, "real") 
confidence = round(probs[0][pred].item() * 100, 2) 
logger.info( 
"Prediction → index=%d | label=%s | all_probs=%s | mapping=%s", 
pred, raw_label, probs[0].tolist(), LABEL_MAPPING, 
) 
is_fake = raw_label.lower() in ("fake", "false", "fabricated") 
classification = "fabricated" if is_fake else "authentic" 
verdict_label = "الخبر مزيف" if is_fake else "الخبر حقيقي" 
logger.info( 
"Text classification: idx=%d label=%s class=%s conf=%.2f%%", 
pred, raw_label, classification, confidence, 
) 
emb_model = load_embedding_model() 
faiss_index, meta = load_faiss() 
q_vec = emb_model.encode([request.text], convert_to_numpy=True).astype("float32") 
faiss_lib.normalize_L2(q_vec) 
dists, idxs = faiss_index.search(q_vec, min(3, faiss_index.ntotal)) 
related_sources = [ 
{**meta[idx], "similarity": round(float(dists[0][r]), 4)} 
for r, idx in enumerate(idxs[0]) if idx != -1 
] 
top = related_sources[0] if related_sources else { 
"credibility_score": 0.5, "source": "Unknown", 
"credibility": "Unverified", "url": "", 
} 
fused = round(confidence * 0.6 + top["credibility_score"] * 100 * 0.4, 2) 
return { 
"claim": request.text, 
"classification": classification, 
"verdict": verdict_label, 
"confidence_score": fused, 
"verified_source": top["source"], 
"related_sources": related_sources, 
} 
@app.post("/predict-media", tags=["media"]) 
async def predict_media(file: UploadFile = File(...)): 
content_type = file.content_type or "" 
is_image = content_type.startswith("image/") 
is_video = content_type.startswith("video/") 
if not is_image and not is_video: 
raise HTTPException( 
status_code=400, 
detail="نوع الملف غير مدعوم. يُقبل JPG وPNG وWEBP للصور، وMP4 وMOV وAVI للفيديو.", 
) 
raw = await file.read() 
if len(raw) > 200 * 1024 * 1024: 
raise HTTPException(status_code=413, detail="حجم الملف يتجاوز 200 ميغابايت.") 
if is_video: 
suffix = "." + (file.filename or "video.mp4").rsplit(".", 1)[-1] 
with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp: 
tmp.write(raw) 
tmp_path = tmp.name 
try: 
result = await asyncio.get_event_loop().run_in_executor( 
None, _analyse_video_sync, tmp_path 
) 
finally: 
try: 
os.unlink(tmp_path) 
except OSError: 
pass 
return result 
try: 
image = Image.open(io.BytesIO(raw)).convert("RGB") 
except Exception: 
raise HTTPException(status_code=400, detail="تعذّر فك ترميز الصورة.") 
tensor, used_face = await asyncio.get_event_loop().run_in_executor( 
None, _prepare_image_tensor, image 
) 
p_fake, p_real = _run_xception(tensor) 
return _build_image_response(p_fake, p_real, used_face) 
if __name__ == "__main__": 
import uvicorn 
uvicorn.run(app, host="0.0.0.0", port=7860)


