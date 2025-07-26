# main.py - Production YouTube Q&A API (Updated with Root Route)
import os
import re
import logging
import time
from typing import List, Optional
from functools import wraps

from flask import Flask, request, jsonify, g
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_talisman import Talisman
from dotenv import load_dotenv

# YouTube and AI libraries
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    TooManyRequests
)
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from deep_translator import GoogleTranslator
from pytube import YouTube
import wikipedia
import requests
import html

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s |)s",
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Security configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
if not app.config['SECRET_KEY']:
    raise RuntimeError("SECRET_KEY environment variable not set!")

# Additional security configurations
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Security headers with Talisman
Talisman(app, force_https=False)  # Set to True in production with HTTPS

# CORS configuration for Chrome extension
allowed_origins = os.getenv('ALLOWED_ORIGINS', 'chrome-extension://*').split(',')
CORS(app, origins=allowed_origins + ['http://localhost:*'])

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per hour", "10 per minute"],
    headers_enabled=True
)

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OpenAI API key not set! Please set OPENAI_API_KEY environment variable.")

# Proxy configuration
PROXY_HOST = os.getenv("PROXY_HOST", "")
PROXY_PORT = os.getenv("PROXY_PORT", "9050")

proxies = None
if PROXY_HOST:
    proxies = {
        "http": f"socks5://{PROXY_HOST}:{PROXY_PORT}",
        "https": f"socks5://{PROXY_HOST}:{PROXY_PORT}",
    }
    logger.info(f"Using proxy: {PROXY_HOST}:{PROXY_PORT}")

# Custom exceptions
class TranscriptError(Exception):
    """Custom exception for transcript-related errors"""
    pass

class QuotaExceededError(TranscriptError):
    """Raised when API quota is exceeded"""
    pass

# Utility functions
def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from various URL formats"""
    patterns = [
        r"(?:v=|\/videos\/|embed\/|youtu\.be\/|shorts\/)([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError("No valid video ID found in URL")

def translate_to_english(text: str) -> str:
    """Translate text to English if it's not already in English"""
    try:
        detected = GoogleTranslator(source="auto", target="en").detect(text[:160])
        if detected and detected.lower() != "en":
            translated = GoogleTranslator(source="auto", target="en").translate(text)
            return translated
    except Exception as e:
        logger.warning(f"Translation failed: {e}")
    return text

def get_transcript_docs(video_id: str) -> Optional[List[Document]]:
    """
    Fetch transcript using YouTube Transcript API with robust error handling
    """
    try:
        transcript_entries = YouTubeTranscriptApi.get_transcript(
            video_id, 
            languages=["en", "en-US", "en-IN", "hi"], 
            proxies=proxies
        )
        logger.info(f"Fetched transcript with {len(transcript_entries)} entries")
        text = " ".join([d["text"] for d in transcript_entries])
        text_en = translate_to_english(text)
        return [Document(page_content=text_en)]
        
    except TranscriptsDisabled:
        logger.info(f"Transcripts disabled for video {video_id}")
        return None
        
    except NoTranscriptFound:
        logger.warning(f"No transcript found for video {video_id}")
        return None
        
    except TooManyRequests:
        logger.error(f"Rate limit exceeded for video {video_id}")
        raise QuotaExceededError("YouTube API rate limit exceeded")
        
    except VideoUnavailable:
        logger.error(f"Video {video_id} is unavailable")
        return None
        
    except Exception as e:
        logger.error(f"Unexpected error fetching transcript for {video_id}: {e}")
        return None

def get_video_title(youtube_url: str) -> Optional[str]:
    """Get YouTube video title with fallback methods"""
    try:
        video_id = extract_video_id(youtube_url)
        clean_url = f"https://www.youtube.com/watch?v={video_id}"
        yt = YouTube(clean_url)
        return yt.title
    except Exception as e:
        logger.warning(f"Could not fetch video title with pytube: {e}")
        # HTML parse fallback
        try:
            page_url = f"https://www.youtube.com/watch?v={extract_video_id(youtube_url)}"
            r = requests.get(page_url, timeout=8)
            if r.status_code == 200:
                m = re.search(r"<title>(.*?) - YouTube</title>", r.text)
                if m:
                    title = html.unescape(m.group(1)).strip()
                    return title
        except Exception as e2:
            logger.warning(f"Could not fetch/parse video title from HTML: {e2}")
    return None

def clean_video_title_for_wikipedia(title: str) -> str:
    """Clean video title for Wikipedia search"""
    for sep in ["|", "-"]:
        if sep in title:
            title = title.split(sep)[0]
    return title.strip()

def wikipedia_search(query: str) -> Optional[str]:
    """Search Wikipedia with error handling"""
    try:
        summary = wikipedia.summary(query, sentences=2)
        return f"According to Wikipedia:\n{summary}"
    except wikipedia.exceptions.DisambiguationError as e:
        try:
            sub_summary = wikipedia.summary(e.options[0], sentences=2)
            return f"According to Wikipedia ({e.options[0]}):\n{sub_summary}"
        except Exception:
            return None
    except wikipedia.exceptions.PageError:
        return None
    except Exception as ex:
        logger.warning(f"Wikipedia search error: {ex}")
        return None

def web_search_links(query: str) -> str:
    """Generate web search links as fallback"""
    import urllib.parse
    q_url = urllib.parse.quote(query)
    return (
        f"Sorry, I couldn't answer from the transcript or Wikipedia.\n"
        f"You can try searching the web:\n"
        f"- [Google](https://www.google.com/search?q={q_url})\n"
        f"- [DuckDuckGo](https://duckduckgo.com/?q={q_url})"
    )

def clean_for_wikipedia(query: str) -> str:
    """Clean query for Wikipedia search"""
    query = query.strip()
    match = re.match(
        r"(who|what|when|where|why|how)\s+(is|are|was|were|do|does|did|has|have|can|could|should|would)?\s*(.*)",
        query,
        flags=re.IGNORECASE,
    )
    if match:
        topic = match.group(3).strip(" .?")
        return topic
    return query

# Response quality patterns
VAGUE_PATTERNS = [
    "do not like each other",
    "i don't know",
    "i do not know",
    "not mentioned",
    "not provided",
    "not stated",
    "no idea",
    "no information",
    "no details",
    "insufficient information",
    "unclear",
    "unable to determine",
    "cannot determine",
    "can't say",
    "no context",
    "context not found",
    "the transcript does not",
    "sorry",
    "unfortunately",
]

def is_summary_question(question: str) -> bool:
    """Check if the question is asking for a summary"""
    qs = question.lower()
    return (
        "what is this video about" in qs
        or "what is the topic" in qs
        or "main topic" in qs
        or "summarize" in qs
        or "summary" in qs
    )

class YouTubeConversationalQA:
    """Main YouTube Q&A service class"""
    
    def __init__(self, model="gpt-3.5-turbo"):
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        self.vectorstore_cache = {}
        self.llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY, 
            model=model, 
            temperature=0.4, 
            max_tokens=512
        )
        self.convs = {}

    def build_chain(self, video_url: str, session_id: str = "default"):
        """Build conversational retrieval chain for video"""
        video_id = extract_video_id(video_url)
        
        if video_id not in self.vectorstore_cache:
            docs = get_transcript_docs(video_id)
            if not docs:
                logger.warning(f"No transcript docs found for video_id={video_id}")
                return None
                
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            splits = splitter.split_documents(docs)
            vdb = FAISS.from_documents(splits, self.embeddings)
            self.vectorstore_cache[video_id] = vdb
            
        retriever = self.vectorstore_cache[video_id].as_retriever()
        
        if session_id not in self.convs:
            self.convs[session_id] = ConversationBufferMemory(
                memory_key="chat_history", 
                return_messages=True, 
                output_key="answer"
            )
            
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.convs[session_id],
            return_source_documents=False,
        )

    def is_incomplete(self, text: str) -> bool:
        """Check if response is incomplete or vague"""
        if not text or len(text.strip()) < 8:
            return True
        lowered = text.lower()
        for pat in VAGUE_PATTERNS:
            if pat in lowered:
                return True
        return False

    def ask(self, video_url: str, question: str, session_id: str = "default") -> str:
        """Main Q&A method with fallback chain"""
        fallback_to_title = False
        context_answer = None
        
        # Try transcript-based Q&A first
        chain = self.build_chain(video_url, session_id)
        if chain is not None:
            try:
                if is_summary_question(question):
                    custom_template = (
                        "Given the following video transcript, briefly summarize the main topic or content "
                        "of this video. Only use context, do NOT speculate. Transcript: {context}\n"
                        "In English, answer in 2-4 sentences."
                    )
                    result = chain.invoke({"question": custom_template})
                else:
                    result = chain.invoke({"question": question})
                context_answer = (result.get("answer", "") if result else "").strip()
            except Exception as e:
                logger.warning(f"Transcript-based QA failed: {e}")
                context_answer = None
                fallback_to_title = True
        else:
            fallback_to_title = True

        # Return if we have a good answer from transcript
        if context_answer and not self.is_incomplete(context_answer):
            return context_answer

        # Fallback to Wikipedia search
        title_q = None
        if fallback_to_title:
            title_q = get_video_title(video_url)
            search_term = title_q if title_q else question
            wiki_ans = wikipedia_search(search_term)

            if (not wiki_ans or self.is_incomplete(wiki_ans)) and title_q:
                short_search = clean_video_title_for_wikipedia(title_q)
                if short_search != search_term:
                    wiki_ans = wikipedia_search(short_search)

            if wiki_ans and not self.is_incomplete(wiki_ans):
                return wiki_ans

        # Try direct Wikipedia search on question
        wiki_ans = wikipedia_search(question)
        if wiki_ans and not self.is_incomplete(wiki_ans):
            return wiki_ans

        # Try cleaned question
        topic = clean_for_wikipedia(question)
        if topic != question:
            wiki_ans2 = wikipedia_search(topic)
            if wiki_ans2 and not self.is_incomplete(wiki_ans2):
                return wiki_ans2

        # Final fallback
        return web_search_links(question)

# Initialize Q&A service
qa_service = YouTubeConversationalQA()

# Request timing middleware
@app.before_request
def before_request():
    g.start_time = time.time()

@app.after_request
def after_request(response):
    duration = time.time() - g.start_time
    logger.info(f"Request completed in {duration:.3f}s - {request.method} {request.path} - {response.status_code}")
    return response

# Error handlers
@app.errorhandler(QuotaExceededError)
def handle_quota_exceeded(e):
    logger.error(f"Quota exceeded: {e}")
    return jsonify({'error': 'API quota exceeded. Please try again later.'}), 429

@app.errorhandler(ValueError)
def handle_value_error(e):
    logger.error(f"Value error: {e}")
    return jsonify({'error': str(e)}), 400

@app.errorhandler(Exception)
def handle_general_error(e):
    logger.error(f"Unexpected error: {e}")
    return jsonify({'error': 'Internal server error'}), 500

# Routes
@app.route('/')
def index():
    """Root endpoint - API information (FIXES 404 ERROR)"""
    return jsonify({
        'service': 'YouTube Q&A API',
        'version': '1.0.0',
        'status': 'operational',
        'endpoints': {
            'health': '/health',
            'youtube_qa': '/api/v1/youtube-qa',
            'api_status': '/api/v1/status'
        },
        'documentation': 'Send POST requests to /api/v1/youtube-qa with url and question fields',
        'usage': {
            'method': 'POST',
            'endpoint': '/api/v1/youtube-qa',
            'required_fields': ['url', 'question'],
            'optional_fields': ['session_id']
        },
        'timestamp': time.time()
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'YouTube Q&A API',
        'version': '1.0.0',
        'timestamp': time.time()
    })

@app.route('/api/v1/youtube-qa', methods=['POST'])
@limiter.limit("5 per minute")
def youtube_qa():
    """Main YouTube Q&A endpoint"""
    try:
        # Input validation
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        if 'url' not in data or 'question' not in data:
            return jsonify({'error': 'Missing required fields: url and question'}), 400
        
        url = data['url'].strip()
        question = data['question'].strip()
        session_id = data.get('session_id', 'default')
        
        # Additional validation
        if not url or not question:
            return jsonify({'error': 'URL and question cannot be empty'}), 400
        
        if len(question) > 500:
            return jsonify({'error': 'Question too long (max 500 characters)'}), 400
        
        # Validate YouTube URL
        try:
            video_id = extract_video_id(url)
        except ValueError as e:
            return jsonify({'error': 'Invalid YouTube URL format'}), 400
        
        logger.info(f"Processing Q&A request for video {video_id}: {question[:50]}...")
        
        # Process the question
        answer = qa_service.ask(url, question, session_id)
        
        response_data = {
            'answer': answer,
            'status': 'success',
            'video_id': video_id,
            'timestamp': time.time()
        }
        
        logger.info(f"Successfully processed Q&A for video {video_id}")
        return jsonify(response_data)
        
    except QuotaExceededError:
        raise  # Let error handler deal with it
    except ValueError:
        raise  # Let error handler deal with it
    except Exception as e:
        logger.error(f"Unexpected error in youtube_qa: {e}")
        raise  # Let error handler deal with it

@app.route('/api/v1/status')
def api_status():
    """API status endpoint"""
    return jsonify({
        'api_version': '1.0.0',
        'service': 'YouTube Q&A API',
        'openai_configured': bool(OPENAI_API_KEY),
        'proxy_configured': bool(proxies),
        'supported_languages': ['en', 'en-US', 'en-IN', 'hi'],
        'max_question_length': 500,
        'rate_limits': {
            'youtube_qa': '5 per minute',
            'general': '100 per hour, 10 per minute'
        }
    })

if __name__ == '__main__':
    # Development server (never use in production)
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
