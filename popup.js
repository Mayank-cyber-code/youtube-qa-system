// YouTube Q&A Extension Popup Script
// Updated for production deployment on Render

class YouTubeQAPopup {
    constructor() {
        // Updated API base URL to match your Render deployment
        this.API_BASE_URL = 'https://youtube-qa-system.onrender.com';
        this.currentVideoUrl = '';
        this.currentVideoTitle = '';
        this.isProcessing = false;
        
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
        
        // Handle API status refresh
        this.elements.apiStatus.addEventListener('click', () => {
            this.checkApiStatus();
        });
    }
    
    async initializePopup() {
        try {
            this.showLoadingState('Initializing...');
            
            // Check API status first
            await this.checkApiStatus();
            
            // Get current YouTube video info
            await this.getCurrentVideoInfo();
            
            // Load previous question if exists
            await this.loadPreviousQuestion();
            
            this.hideLoadingState();
            
        } catch (error) {
            console.error('Failed to initialize popup:', error);
            this.showError('Failed to initialize extension');
            this.hideLoadingState();
        }
    }
    
    showLoadingState(message) {
        if (this.elements.statusText) {
            this.elements.statusText.textContent = message;
        }
    }
    
    hideLoadingState() {
        // Reset to normal state
    }
    
    async getCurrentVideoInfo() {
        try {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            
            if (!tab.url.includes('youtube.com/watch')) {
                this.showError('Please navigate to a YouTube video page');
                this.disableInterface();
                return;
            }
            
            // Extract video info from URL
            const urlParams = new URLSearchParams(new URL(tab.url).search);
            const videoId = urlParams.get('v');
            
            if (!videoId) {
                this.showError('Could not detect YouTube video');
                this.disableInterface();
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
            
            console.log('Video detected:', { videoId, title: this.currentVideoTitle });
            
        } catch (error) {
            console.error('Error getting video info:', error);
            this.showError('Could not get current video information');
            this.disableInterface();
        }
    }
    
    disableInterface() {
        this.elements.questionInput.disabled = true;
        this.elements.askButton.disabled = true;
    }
    
    async checkApiStatus() {
        try {
            console.log('Checking API status...');
            
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout
            
            const response = await fetch(`${this.API_BASE_URL}/health`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                },
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            
            if (response.ok) {
                const data = await response.json();
                this.updateApiStatus('online', 'API Connected');
                console.log('API status:', data);
            } else {
                this.updateApiStatus('offline', `API Error (${response.status})`);
                console.error('API returned error status:', response.status);
            }
        } catch (error) {
            if (error.name === 'AbortError') {
                this.updateApiStatus('offline', 'Connection Timeout');
                console.error('API request timed out');
            } else {
                this.updateApiStatus('offline', 'Connection Failed');
                console.error('API check failed:', error);
            }
        }
    }
    
    updateApiStatus(status, text) {
        this.elements.statusIndicator.className = `status-indicator ${status}`;
        this.elements.statusText.textContent = text;
        
        if (status === 'offline') {
            this.elements.askButton.disabled = true;
        }
    }
    
    updateCharCount() {
        const count = this.elements.questionInput.value.length;
        this.elements.charCount.textContent = count;
        
        // Color coding for character count
        if (count > 450) {
            this.elements.charCount.style.color = 'var(--error-color, #dc3545)';
        } else if (count > 400) {
            this.elements.charCount.style.color = 'var(--warning-color, #ffc107)';
        } else {
            this.elements.charCount.style.color = '#666';
        }
        
        // Disable button if too long
        if (count > 500) {
            this.elements.askButton.disabled = true;
        }
    }
    
    toggleAskButton() {
        const hasQuestion = this.elements.questionInput.value.trim().length > 2;
        const hasVideo = this.currentVideoUrl !== '';
        const notTooLong = this.elements.questionInput.value.length <= 500;
        const notProcessing = !this.isProcessing;
        
        this.elements.askButton.disabled = !(hasQuestion && hasVideo && notTooLong && notProcessing);
    }
    
    async handleAskQuestion() {
        const question = this.elements.questionInput.value.trim();
        
        // Validation
        if (!question) {
            this.showError('Please enter a question');
            return;
        }
        
        if (question.length > 500) {
            this.showError('Question is too long (max 500 characters)');
            return;
        }
        
        if (!this.currentVideoUrl) {
            this.showError('No YouTube video detected');
            return;
        }
        
        this.isProcessing = true;
        this.setLoadingState(true);
        this.hideError();
        this.hideAnswer();
        
        try {
            console.log('Sending request to API:', {
                url: this.currentVideoUrl,
                question: question.substring(0, 50) + '...'
            });
            
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 second timeout
            
            const response = await fetch(`${this.API_BASE_URL}/api/v1/youtube-qa`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    url: this.currentVideoUrl,
                    question: question,
                    session_id: await this.getSessionId()
                }),
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            
            const data = await response.json();
            console.log('API response:', data);
            
            if (response.ok) {
                this.showAnswer(data.answer);
                await this.saveQuestion(question);
                
                // Track successful request
                this.trackUsage('question_answered', {
                    video_id: data.video_id,
                    question_length: question.length
                });
            } else {
                let errorMessage = data.error || 'Failed to get answer';
                
                // Handle specific error cases
                if (response.status === 429) {
                    errorMessage = 'Rate limit exceeded. Please wait a moment and try again.';
                } else if (response.status === 400) {
                    errorMessage = 'Invalid request. Please check your YouTube URL and question.';
                } else if (response.status >= 500) {
                    errorMessage = 'Server error. Please try again later.';
                }
                
                this.showError(errorMessage);
            }
            
        } catch (error) {
            console.error('API request failed:', error);
            
            if (error.name === 'AbortError') {
                this.showError('Request timed out. Please try again.');
            } else if (error.message.includes('fetch')) {
                this.showError('Network error. Please check your connection.');
            } else {
                this.showError('An unexpected error occurred. Please try again.');
            }
        } finally {
            this.isProcessing = false;
            this.setLoadingState(false);
            this.toggleAskButton();
        }
    }
    
    setLoadingState(loading) {
        if (loading) {
            this.elements.btnText.style.display = 'none';
            this.elements.loadingSpinner.style.display = 'block';
            this.elements.questionInput.disabled = true;
        } else {
            this.elements.btnText.style.display = 'block';
            this.elements.loadingSpinner.style.display = 'none';
            this.elements.questionInput.disabled = false;
        }
        
        this.toggleAskButton();
    }
    
    showAnswer(answer) {
        // Format answer with basic markdown-like formatting
        const formattedAnswer = this.formatAnswer(answer);
        this.elements.answerContent.innerHTML = formattedAnswer;
        this.elements.answerSection.style.display = 'block';
        
        // Scroll to answer
        setTimeout(() => {
            this.elements.answerSection.scrollIntoView({ behavior: 'smooth' });
        }, 100);
    }
    
    formatAnswer(answer) {
        // Basic formatting for better readability
        return answer
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>')
            .replace(/^/, '<p>')
            .replace(/$/, '</p>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // Bold
            .replace(/\*(.*?)\*/g, '<em>$1</em>'); // Italic
    }
    
    hideAnswer() {
        this.elements.answerSection.style.display = 'none';
    }
    
    showError(message) {
        this.elements.errorContent.textContent = message;
        this.elements.errorSection.style.display = 'block';
        
        // Auto-hide error after 5 seconds
        setTimeout(() => {
            this.hideError();
        }, 5000);
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
            this.elements.copyAnswer.textContent = '✓ Copied';
            this.elements.copyAnswer.style.background = '#28a745';
            
            setTimeout(() => {
                this.elements.copyAnswer.textContent = originalText;
                this.elements.copyAnswer.style.background = '';
            }, 2000);
            
        } catch (error) {
            console.error('Failed to copy:', error);
            
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = this.elements.answerContent.textContent;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            
            this.elements.copyAnswer.textContent = '✓ Copied';
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
            return 'default_session';
        }
    }
    
    async saveQuestion(question) {
        try {
            const timestamp = new Date().toISOString();
            await chrome.storage.local.set({ 
                lastQuestion: question,
                lastVideoUrl: this.currentVideoUrl,
                lastQuestionTime: timestamp
            });
        } catch (error) {
            console.error('Failed to save question:', error);
        }
    }
    
    async loadPreviousQuestion() {
        try {
            const result = await chrome.storage.local.get(['lastQuestion', 'lastVideoUrl', 'lastQuestionTime']);
            
            // Only load if it's the same video and within the last hour
            if (result.lastQuestion && 
                result.lastVideoUrl === this.currentVideoUrl &&
                result.lastQuestionTime) {
                
                const lastTime = new Date(result.lastQuestionTime);
                const now = new Date();
                const hoursSince = (now - lastTime) / (1000 * 60 * 60);
                
                if (hoursSince < 1) { // Within last hour
                    this.elements.questionInput.value = result.lastQuestion;
                    this.updateCharCount();
                    this.toggleAskButton();
                }
            }
        } catch (error) {
            console.error('Failed to load previous question:', error);
        }
    }
    
    async trackUsage(event, data) {
        try {
            // Simple usage tracking for analytics
            const usage = await chrome.storage.local.get(['usageStats']) || { usageStats: {} };
            const stats = usage.usageStats || {};
            
            if (!stats[event]) {
                stats[event] = 0;
            }
            stats[event]++;
            
            await chrome.storage.local.set({ usageStats: stats });
        } catch (error) {
            console.error('Failed to track usage:', error);
        }
    }
}

// Initialize popup when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('YouTube Q&A Extension initializing...');
    new YouTubeQAPopup();
});

// Handle extension unload
window.addEventListener('beforeunload', () => {
    console.log('YouTube Q&A Extension closing...');
});
