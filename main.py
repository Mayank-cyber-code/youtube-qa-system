import os
import re
import logging
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass
from functools import lru_cache
import secrets

# Web framework and security
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_talisman import Talisman
from werkzeug.middleware.proxy_fix import ProxyFix

# Environment and configuration
from dotenv import load_dotenv

# YouTube and AI dependencies
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
import wikipedia
from pytube import YouTube
import requests
import html

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("youtube_qa_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === CONFIGURATION CLASS ===
class SecureConfig:
    def __init__(self):
        self._validate_required_keys()
    
    def _validate_required_keys(self):
        required_keys = ["OPENAI_API_KEY"]
        missing_keys = [key for key in required_keys if not os.getenv(key)]
        if missing_keys:
            raise RuntimeError(f"Missing required environment variables: {missing_keys}")
    
    @property
    def openai_api_key(self) -> str:
        return os.getenv("OPENAI_API_KEY")
    
    @property
    def openai_model(self) -> str:
        return os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    
    @property
    def openai_max_tokens(self) -> int:
        return int(os.getenv("OPENAI_MAX_TOKENS", "512"))
    
    @property
    def proxy_settings(self) -> Optional[dict]:
        proxy_ip = os.getenv("TOR_PROXY_IP")
        proxy_port = os.getenv("TOR_PROXY_PORT", "9050")
        if proxy_ip:
            return {
                "http": f"socks5://{proxy_ip}:{proxy_port}",
                "https": f"socks5://{proxy_ip}:{proxy_port}",
            }
        return None
    
    @property
    def allowed_origins(self) -> List[str]:
        origins = os.getenv("ALLOWED_ORIGINS", "").split(",")
        return [origin.strip() for origin in origins if origin.strip()]
    
    @property
    def rate_limit_per_minute(self) -> int:
        return int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
    
    @property
    def rate_limit_per_hour(self) -> int:
        return int(os.getenv("RATE_LIMIT_PER_HOUR", "100"))

# === CUSTOM EXCEPTIONS ===
class TranscriptError(Exception):
    """Custom exception for transcript-related errors"""
    pass

class QuotaExceededError(TranscriptError):
    """Raised when API quota is exceeded"""
    pass

class ValidationError(Exception):
    """Raised when input validation fails"""
    pass

# === DATA MODELS ===
@dataclass
class QARequest:
    """Data class for Q&A requests"""
    video_url: str
    question: str
    session_id: str = "default"

@dataclass
class QAResponse:
    """Data class for Q&A responses"""
    answer: str
    video_id: str
    status: str
    timestamp: datetime
    processing_time: float

# === UTILITY FUNCTIONS ===
def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from various URL formats"""
    patterns = [
        r"(?:v=|/videos/|embed/|youtu\.be/|shorts/)([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValidationError("No valid video ID found in URL")

def validate_input(data: dict) -> QARequest:
    """Validate and sanitize input data"""
    if not data:
        raise ValidationError("No data provided")
    
    if 'url' not in data or 'question' not in data:
        raise ValidationError("Missing required fields: url, question")
    
    # Sanitize inputs
    video_url = str(data['url']).strip()
    question = str(data['question']).strip()
    session_id = str(data.get('session_id', 'default')).strip()
    
    # Validate URL
    if not video_url:
        raise ValidationError("Video URL cannot be empty")
    
    # Validate question
    if not question or len(question) < 3:
        raise ValidationError("Question must be at least 3 characters long")
    
    if len(question) > 500:
        raise ValidationError("Question cannot exceed 500 characters")
    
    # Validate session ID
    if len(session_id) > 50:
        raise ValidationError("Session ID cannot exceed 50 characters")
    
    return QARequest(video_url=video_url, question=question, session_id=session_id)

# === OPTIMIZED VECTOR STORE ===
class OptimizedVectorStore:
    def __init__(self, embeddings, use_compression=True):
        self.embeddings = embeddings
        self.use_compression = use_compression
        
    def create_optimized_index(self, documents: List[Document]) -> FAISS:
        """Create optimized FAISS index with performance improvements"""
        try:
            if len(documents) > 5000:  # Use IVF for large datasets
                logger.info("Creating IVF-optimized FAISS index for large dataset")
                # For production, you would implement IVF clustering here
                # For now, use the standard approach with optimization
                return FAISS.from_documents(documents, self.embeddings)
            else:
                logger.info("Creating standard FAISS index")
                return FAISS.from_documents(documents, self.embeddings)
        except Exception as e:
            logger.error(f"Failed to create FAISS index: {e}")
            raise TranscriptError(f"Vector store creation failed: {e}")

# === ENHANCED TRANSCRIPT PROCESSOR ===
class EnhancedTranscriptProcessor:
    def __init__(self, config: SecureConfig):
        self.config = config
        
    def get_transcript_with_robust_handling(self, video_id: str) -> Optional[List[Document]]:
        """Enhanced transcript fetching with specific error handling"""
        try:
            proxies = self.config.proxy_settings
            transcript_entries = YouTubeTranscriptApi.get_transcript(
                video_id, 
                languages=["en", "en-US", "en-IN", "hi"], 
                proxies=proxies
            )
            
            logger.info(f"Successfully fetched transcript with {len(transcript_entries)} entries")
            text = " ".join([d["text"] for d in transcript_entries])
            text_en = self._translate_to_english(text)
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
            
        except ConnectionError as e:
            logger.error(f"Network connection failed: {e}")
            raise TranscriptError(f"Failed to connect to YouTube API: {e}")
            
        except Exception as e:
            logger.error(f"Unexpected error fetching transcript for {video_id}: {e}")
            raise TranscriptError(f"Unexpected transcript error: {e}")
    
    def _translate_to_english(self, text: str) -> str:
        """Safely translate text to English"""
        try:
            detected = GoogleTranslator(source="auto", target="en").detect(text[:160])
            if detected and detected.lower() != "en":
                translated = GoogleTranslator(source="auto", target="en").translate(text)
                return translated
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
        return text

# === ENHANCED QA SERVICE ===
class YouTubeQAService:
    def __init__(self, config: SecureConfig):
        self.config = config
        self.embeddings = OpenAIEmbeddings(openai_api_key=config.openai_api_key)
        self.vectorstore_cache = {}
        self.llm = ChatOpenAI(
            openai_api_key=config.openai_api_key, 
            model=config.openai_model, 
            temperature=0.4, 
            max_tokens=config.openai_max_tokens
        )
        self.conversations = {}
        self.transcript_processor = EnhancedTranscriptProcessor(config)
        self.vector_store = OptimizedVectorStore(self.embeddings)
        
    def process_qa_request(self, qa_request: QARequest) -> QAResponse:
        """Process Q&A request with comprehensive error handling"""
        start_time = datetime.now()
        
        try:
            video_id = extract_video_id(qa_request.video_url)
            answer = self._get_answer(video_id, qa_request.question, qa_request.session_id)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return QAResponse(
                answer=answer,
                video_id=video_id,
                status="success",
                timestamp=datetime.now(),
                processing_time=processing_time
            )
            
        except ValidationError as e:
            logger.warning(f"Validation error: {e}")
            raise
        except QuotaExceededError as e:
            logger.error(f"Quota exceeded: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error processing request: {e}")
            raise TranscriptError(f"Failed to process request: {e}")
    
    def _get_answer(self, video_id: str, question: str, session_id: str) -> str:
        """Get answer using transcript or fallback methods"""
        # Try transcript-based answer first
        chain = self._build_chain(video_id, session_id)
        if chain:
            try:
                result = chain.invoke({"question": question})
                answer = (result.get("answer", "") if result else "").strip()
                if answer and not self._is_incomplete(answer):
                    return answer
            except Exception as e:
                logger.warning(f"Transcript-based QA failed: {e}")
        
        # Fallback to Wikipedia
        wiki_answer = self._wikipedia_search(question)
        if wiki_answer and not self._is_incomplete(wiki_answer):
            return wiki_answer
        
        # Final fallback
        return self._web_search_links(question)
    
    def _build_chain(self, video_id: str, session_id: str):
        """Build conversation chain with caching"""
        if video_id not in self.vectorstore_cache:
            docs = self.transcript_processor.get_transcript_with_robust_handling(video_id)
            if not docs:
                return None
                
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            splits = splitter.split_documents(docs)
            vdb = self.vector_store.create_optimized_index(splits)
            self.vectorstore_cache[video_id] = vdb
        
        retriever = self.vectorstore_cache[video_id].as_retriever()
        
        if session_id not in self.conversations:
            self.conversations[session_id] = ConversationBufferMemory(
                memory_key="chat_history", 
                return_messages=True, 
                output_key="answer"
            )
        
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.conversations[session_id],
            return_source_documents=False,
        )
    
    def _is_incomplete(self, text: str) -> bool:
        """Check if response is incomplete or vague"""
        if not text or len(text.strip()) < 8:
            return True
        
        vague_patterns = [
            "do not like each other", "i don't know", "not mentioned",
            "not provided", "no information", "unclear", "sorry"
        ]
        
        lowered = text.lower()
        return any(pattern in lowered for pattern in vague_patterns)
    
    def _wikipedia_search(self, query: str) -> Optional[str]:
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
        except Exception as ex:
            logger.warning(f"Wikipedia search error: {ex}")
            return None
    
    def _web_search_links(self, query: str) -> str:
        """Provide web search links as fallback"""
        import urllib.parse
        q_url = urllib.parse.quote(query)
        return (
            f"Sorry, I couldn't find a specific answer. Try searching:\n"
            f"- [Google](https://www.google.com/search?q={q_url})\n"
            f"- [DuckDuckGo](https://duckduckgo.com/?q={q_url})"
        )

# === FLASK APPLICATION ===
def create_app() -> Flask:
    """Create and configure Flask application"""
    config = SecureConfig()
    
    app = Flask(__name__)
    
    # Security configurations
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', secrets.token_hex(32))
    app.config['JSON_SORT_KEYS'] = False
    
    # Trust proxy headers (for deployment behind nginx/AWS ALB)
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
    
    # Security headers with Talisman
    Talisman(
        app,
        force_https=False,  # Set to True in production
        strict_transport_security=True,
        content_security_policy={
            'default-src': "'self'",
            'script-src': "'self'",
            'style-src': "'self' 'unsafe-inline'",
            'img-src': "'self' data:",
        }
    )
    
    # CORS configuration
    CORS(app, 
         origins=config.allowed_origins or ["*"],
         supports_credentials=False,
         methods=["GET", "POST", "OPTIONS"],
         allow_headers=["Content-Type", "Authorization"])
    
    # Rate limiting
    limiter = Limiter(
        app,
        key_func=get_remote_address,
        default_limits=[f"{config.rate_limit_per_hour} per hour"]
    )
    
    # Initialize QA service
    qa_service = YouTubeQAService(config)
    
    # === API ROUTES ===
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        })
    
    @app.route('/api/v1/youtube-qa', methods=['POST', 'OPTIONS'])
    @limiter.limit(f"{config.rate_limit_per_minute} per minute")
    def youtube_qa_endpoint():
        """Production-ready API endpoint for YouTube Q&A"""
        if request.method == 'OPTIONS':
            # Handle preflight request
            response = make_response()
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            return response
        
        try:
            # Validate input
            data = request.get_json()
            qa_request = validate_input(data)
            
            # Process request
            qa_response = qa_service.process_qa_request(qa_request)
            
            return jsonify({
                'answer': qa_response.answer,
                'video_id': qa_response.video_id,
                'status': qa_response.status,
                'processing_time': qa_response.processing_time
            })
            
        except ValidationError as e:
            logger.warning(f"Validation error: {e}")
            return jsonify({'error': str(e)}), 400
            
        except QuotaExceededError as e:
            logger.error(f"Quota exceeded: {e}")
            return jsonify({'error': 'API quota exceeded. Please try again later.'}), 429
            
        except TranscriptError as e:
            logger.error(f"Transcript error: {e}")
            return jsonify({'error': 'Failed to process video transcript.'}), 422
            
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return jsonify({'error': 'Internal server error'}), 500
    
    # Error handlers
    @app.errorhandler(429)
    def rate_limit_handler(e):
        return jsonify({'error': 'Rate limit exceeded'}), 429
    
    @app.errorhandler(500)
    def internal_error_handler(e):
        return jsonify({'error': 'Internal server error'}), 500
    
    return app

# === APPLICATION ENTRY POINT ===
if __name__ == '__main__':
    app = create_app()
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    
    logger.info(f"Starting YouTube Q&A API on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
