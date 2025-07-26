// YouTube Q&A Extension Popup Script

class YouTubeQAPopup {
    constructor() {
        this.API_BASE_URL = 'https://your-api-domain.com/api/v1';
        this.currentVideoUrl = '';
        this.currentVideoTitle = '';
        
        this.initializeElements();
        this.attachEventListeners();
        this.initializePopup();
    }
    
    initializeElements() {
        // Get DOM elements
        this.elements = {
            videoInfo: document.getElementById('videoInfo'),
            videoTitle: document.getElementById('videoTitle'),
            videoUrl: document.getElementById('videoUrl'),
            questionInput: document.getElementById('questionInput'),
            charCount: document.getElementById('charCount'),
            askButton: document.getElementById('askButton'),
            clearButton: document.getElementById('clearButton'),
            answerSection: document.getElementById('answerSection'),
            answerContent: document.getElementById('answerContent'),
            copyAnswer: document.getElementById('copyAnswer'),
            errorSection: document.getElementById('errorSection'),
            errorContent: document.getElementById('errorContent'),
            apiStatus: document.getElementById('apiStatus'),
            statusIndicator: document.getElementById('statusIndicator'),
            statusText: document.getElementById('statusText'),
            loadingSpinner: document.querySelector('.loading-spinner'),
            btnText: document.querySelector('.btn-text')
        };
    }
    
    attachEventListeners() {
        // Question input events
        this.elements.questionInput.addEventListener('input', (e) => {
            this.updateCharCount();
            this.toggleAskButton();
        });
        
        this.elements.questionInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                this.handleAskQuestion();
            }
        });
        
        // Button events
        this.elements.askButton.addEventListener('click', () => {
            this.handleAskQuestion();
        });
        
        this.elements.clearButton.addEventListener('click', () => {
            this.clearQuestion();
        });
        
        this.elements.copyAnswer.addEventListener('click', () => {
            this.copyAnswerToClipboard();
        });
    }
    
    async initializePopup() {
        try {
            // Check API status
            await this.checkApiStatus();
            
            // Get current YouTube video info
            await this.getCurrentVideoInfo();
            
            // Load previous question if exists
            await this.loadPreviousQuestion();
            
        } catch (error) {
            console.error('Failed to initialize popup:', error);
            this.showError('Failed to initialize extension');
        }
    }
    
    async getCurrentVideoInfo() {
        try {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            
            if (!tab.url.includes('youtube.com/watch')) {
                this.showError('Please navigate to a YouTube video page');
                return;
            }
            
            // Extract video info from URL
            const urlParams = new URLSearchParams(new URL(tab.url).search);
            const videoId = urlParams.get('v');
            
            if (!videoId) {
                this.showError('Could not detect YouTube video');
                return;
            }
            
            this.currentVideoUrl = tab.url;
            this.currentVideoTitle = tab.title.replace(' - YouTube', '');
            
            // Update UI
            this.elements.videoTitle.textContent = this.currentVideoTitle;
            this.elements.videoUrl.textContent = this.currentVideoUrl;
            this.elements.videoInfo.style.display = 'block';
            
            // Enable question input
            this.elements.questionInput.disabled = false;
            this.toggleAskButton();
            
        } catch (error) {
            console.error('Error getting video info:', error);
            this.showError('Could not get current video information');
        }
    }
    
    async checkApiStatus() {
        try {
            const response = await fetch(`${this.API_BASE_URL}/health`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            if (response.ok) {
                this.updateApiStatus('online', 'API Connected');
            } else {
                this.updateApiStatus('offline', 'API Unavailable');
            }
        } catch (error) {
            this.updateApiStatus('offline', 'Connection Failed');
        }
    }
    
    updateApiStatus(status, text) {
        this.elements.statusIndicator.className = `status-indicator ${status}`;
        this.elements.statusText.textContent = text;
    }
    
    updateCharCount() {
        const count = this.elements.questionInput.value.length;
        this.elements.charCount.textContent = count;
        
        if (count > 450) {
            this.elements.charCount.style.color = 'var(--error-color)';
        } else if (count > 400) {
            this.elements.charCount.style.color = 'var(--warning-color)';
        } else {
            this.elements.charCount.style.color = '#666';
        }
    }
    
    toggleAskButton() {
        const hasQuestion = this.elements.questionInput.value.trim().length > 2;
        const hasVideo = this.currentVideoUrl !== '';
        this.elements.askButton.disabled = !(hasQuestion && hasVideo);
    }
    
    async handleAskQuestion() {
        const question = this.elements.questionInput.value.trim();
        
        if (!question) {
            this.showError('Please enter a question');
            return;
        }
        
        if (!this.currentVideoUrl) {
            this.showError('No YouTube video detected');
            return;
        }
        
        this.setLoadingState(true);
        this.hideError();
        this.hideAnswer();
        
        try {
            const response = await fetch(`${this.API_BASE_URL}/youtube-qa`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    url: this.currentVideoUrl,
                    question: question,
                    session_id: await this.getSessionId()
                })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                this.showAnswer(data.answer);
                await this.saveQuestion(question);
            } else {
                this.showError(data.error || 'Failed to get answer');
            }
            
        } catch (error) {
            console.error('API request failed:', error);
            this.showError('Network error. Please check your connection.');
        } finally {
            this.setLoadingState(false);
        }
    }
    
    setLoadingState(loading) {
        this.elements.askButton.disabled = loading;
        
        if (loading) {
            this.elements.btnText.style.display = 'none';
            this.elements.loadingSpinner.style.display = 'block';
        } else {
            this.elements.btnText.style.display = 'block';
            this.elements.loadingSpinner.style.display = 'none';
        }
    }
    
    showAnswer(answer) {
        this.elements.answerContent.textContent = answer;
        this.elements.answerSection.style.display = 'block';
        
        // Scroll to answer
        this.elements.answerSection.scrollIntoView({ behavior: 'smooth' });
    }
    
    hideAnswer() {
        this.elements.answerSection.style.display = 'none';
    }
    
    showError(message) {
        this.elements.errorContent.textContent = message;
        this.elements.errorSection.style.display = 'block';
    }
    
    hideError() {
        this.elements.errorSection.style.display = 'none';
    }
    
    clearQuestion() {
        this.elements.questionInput.value = '';
        this.updateCharCount();
        this.toggleAskButton();
        this.hideAnswer();
        this.hideError();
        this.elements.questionInput.focus();
    }
    
    async copyAnswerToClipboard() {
        try {
            const answer = this.elements.answerContent.textContent;
            await navigator.clipboard.writeText(answer);
            
            // Show feedback
            const originalText = this.elements.copyAnswer.textContent;
            this.elements.copyAnswer.textContent = '✓';
            
            setTimeout(() => {
                this.elements.copyAnswer.textContent = originalText;
            }, 1000);
            
        } catch (error) {
            console.error('Failed to copy:', error);
        }
    }
    
    async getSessionId() {
        try {
            const result = await chrome.storage.local.get(['sessionId']);
            if (result.sessionId) {
                return result.sessionId;
            }
            
            // Generate new session ID
            const sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            await chrome.storage.local.set({ sessionId });
            return sessionId;
        } catch (error) {
            console.error('Session ID error:', error);
            return 'default';
        }
    }
    
    async saveQuestion(question) {
        try {
            await chrome.storage.local.set({ 
                lastQuestion: question,
                lastVideoUrl: this.currentVideoUrl 
            });
        } catch (error) {
            console.error('Failed to save question:', error);
        }
    }
    
    async loadPreviousQuestion() {
        try {
            const result = await chrome.storage.local.get(['lastQuestion', 'lastVideoUrl']);
            
            if (result.lastQuestion && result.lastVideoUrl === this.currentVideoUrl) {
                this.elements.questionInput.value = result.lastQuestion;
                this.updateCharCount();
                this.toggleAskButton();
            }
        } catch (error) {
            console.error('Failed to load previous question:', error);
        }
    }
}

// Initialize popup when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new YouTubeQAPopup();
});