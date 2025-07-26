# main.py – Production YouTube Q&A API (Complete Updated Version)

import os
import re
import logging
import time
from typing import List, Optional

from flask import Flask, request, jsonify, g
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_talisman import Talisman
from dotenv import load_dotenv

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    TooManyRequests
)
from pytube import YouTube
from deep_translator import GoogleTranslator
import wikipedia
import requests
import html

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Security configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY') or os.urandom(24)
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
Talisman(app, force_https=False)

# CORS configuration
allowed_origins = os.getenv('ALLOWED_ORIGINS', 'chrome-extension://*').split(',')
CORS(app, origins=allowed_origins + ['http://localhost:*'])

# Rate limiting with Redis
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri=REDIS_URL,
    default_limits=["100 per hour", "10 per minute"],
    headers_enabled=True
)

# OpenAI key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OpenAI API key not set! Please set OPENAI_API_KEY environment variable.")

# Optional proxy
PROXY_HOST = os.getenv("PROXY_HOST", "")
PROXY_PORT = os.getenv("PROXY_PORT", "9050")
proxies = None
if PROXY_HOST:
    proxies = {
        "http": f"socks5://{PROXY_HOST}:{PROXY_PORT}",
        "https": f"socks5://{PROXY_HOST}:{PROXY_PORT}"
    }
    logger.info(f"Using proxy: {PROXY_HOST}:{PROXY_PORT}")

# Custom exceptions
class TranscriptError(Exception): pass
class QuotaExceededError(TranscriptError): pass

# Utility functions
def extract_video_id(url: str) -> str:
    patterns = [r"(?:v=|/videos/|embed/|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})"]
    for pat in patterns:
        m = re.search(pat, url)
        if m: return m.group(1)
    raise ValueError("No valid video ID found in URL")

def translate_to_english(text: str) -> str:
    try:
        det = GoogleTranslator(source="auto", target="en").detect(text[:160])
        if det and det.lower() != "en":
            return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception as e:
        logger.warning(f"Translation failed: {e}")
    return text

def get_transcript_docs(video_id: str) -> Optional[List[Document]]:
    try:
        entries = YouTubeTranscriptApi.get_transcript(
            video_id, languages=["en","en-US","en-IN","hi"], proxies=proxies
        )
        text = " ".join(d["text"] for d in entries)
        return [Document(page_content=translate_to_english(text))]
    except TranscriptsDisabled:
        return None
    except NoTranscriptFound:
        return None
    except TooManyRequests:
        raise QuotaExceededError("YouTube API rate limit exceeded")
    except VideoUnavailable:
        return None
    except Exception as e:
        logger.error(f"Transcript fetch error: {e}")
        return None

def get_video_title(url: str) -> Optional[str]:
    try:
        vid = extract_video_id(url)
        yt = YouTube(f"https://www.youtube.com/watch?v={vid}")
        return yt.title
    except Exception as e:
        logger.warning(f"Pytube title fetch failed: {e}")
        try:
            r = requests.get(url, timeout=8)
            m = re.search(r"<title>(.*?) - YouTube</title>", r.text)
            if m: return html.unescape(m.group(1)).strip()
        except: pass
    return None

def wikipedia_search(query: str) -> Optional[str]:
    try:
        return "According to Wikipedia:\n" + wikipedia.summary(query, sentences=2)
    except Exception:
        return None

def web_search_links(query: str) -> str:
    import urllib.parse
    q = urllib.parse.quote(query)
    return (f"Couldn't find answer. Try:\n"
            f"- Google: https://www.google.com/search?q={q}\n"
            f"- DuckDuckGo: https://duckduckgo.com/?q={q}")

def clean_for_wikipedia(q: str) -> str:
    m = re.match(
        r"(who|what|when|where|why|how)\s+(?:is|are|was|were|do|does|did|has|have|can|could|should|would)?\s*(.*)",
        q, flags=re.IGNORECASE
    )
    return m.group(1).strip(" .?") if m else q

VAGUE_PATTERNS = ["no idea","not mentioned","insufficient information","sorry","unfortunately"]
def is_summary_question(q: str) -> bool:
    s = q.lower()
    return any(x in s for x in ["summarize","summary","what is this video about","main topic"])

# Q&A Service
class YouTubeConversationalQA:
    def __init__(self, model="gpt-3.5-turbo"):
        self.emb = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        self.cache = {}
        self.llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=model, temperature=0.4, max_tokens=512)
        self.memories = {}

    def build_chain(self, url: str, session_id: str):
        vid = extract_video_id(url)
        if vid not in self.cache:
            docs = get_transcript_docs(vid)
            if not docs: return None
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200, separators=["\n\n","\n",". "," ",""]
            )
            splits = splitter.split_documents(docs)
            self.cache[vid] = FAISS.from_documents(splits, self.emb)
        retr = self.cache[vid].as_retriever()
        if session_id not in self.memories:
            self.memories[session_id] = ConversationBufferMemory(
                memory_key="chat_history", return_messages=True, output_key="answer"
            )
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm, retriever=retr, memory=self.memories[session_id], return_source_documents=False
        )

    def is_incomplete(self, txt: str) -> bool:
        if not txt or len(txt.strip()) < 8: return True
        lo = txt.lower()
        return any(p in lo for p in VAGUE_PATTERNS)

    def ask(self, url: str, question: str, session_id: str="default") -> str:
        chain = self.build_chain(url, session_id)
        ans = None; fallback = False
        if chain:
            try:
                if is_summary_question(question):
                    tpl = "Given the transcript, summarize the main topic only. Transcript: {context}"
                    res = chain.invoke({"question": tpl})
                else:
                    res = chain.invoke({"question": question})
                ans = res.get("answer","").strip()
            except Exception as e:
                logger.warning(f"QA chain failed: {e}")
                fallback = True
        else:
            fallback = True

        if ans and not self.is_incomplete(ans):
            return ans
        title = get_video_title(url)
        term = title or question
        wiki = wikipedia_search(term)
        if wiki and not self.is_incomplete(wiki): return wiki
        wiki2 = wikipedia_search(clean_for_wikipedia(question))
        if wiki2 and not self.is_incomplete(wiki2): return wiki2
        return web_search_links(question)

qa_service = YouTubeConversationalQA()

@app.before_request
def before_req():
    g.start = time.time()

@app.after_request
def after_req(resp):
    logger.info("Completed in %.3fs %s %s %d",
                time.time()-g.start, request.method, request.path, resp.status_code)
    return resp

@app.errorhandler(QuotaExceededError)
def err_quota(e):
    return jsonify(error="Quota exceeded, try later"), 429

@app.errorhandler(ValueError)
def err_val(e):
    return jsonify(error=str(e)), 400

@app.errorhandler(Exception)
def err_any(e):
    logger.error("Unexpected error: %s", e)
    return jsonify(error="Internal server error"), 500

@app.route('/', methods=['GET','HEAD'])
def index():
    return jsonify({
        "service":"YouTube Q&A API","version":"1.0.0","status":"operational",
        "endpoints":{"health":"/health","qa":"/api/v1/youtube-qa","status":"/api/v1/status"},
        "timestamp":time.time()
    }), 200

@app.route('/health')
def health():
    return jsonify(status="healthy", timestamp=time.time())

@app.route('/api/v1/status')
def status():
    return jsonify(openai_configured=bool(OPENAI_API_KEY),
                   proxy=bool(proxies), redis=bool(REDIS_URL))

@app.route('/api/v1/youtube-qa', methods=['POST'])
@limiter.limit("5 per minute")
def youtube_qa():
    data = request.get_json() or {}
    url = data.get("url","").strip()
    q = data.get("question","").strip()
    if not url or not q:
        return jsonify(error="Missing url or question"), 400
    if len(q)>500:
        return jsonify(error="Question too long (max 500 chars)"), 400
    try:
        vid = extract_video_id(url)
    except ValueError:
        return jsonify(error="Invalid YouTube URL"), 400
    ans = qa_service.ask(url, q, data.get("session_id","default"))
    return jsonify(answer=ans, video_id=vid, timestamp=time.time())

if __name__=="__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT",5000)), debug=False)
