// popup.js – YouTube Q&A Chrome Extension Popup Script

class YouTubeQAPopup {
    constructor() {
        this.API_BASE_URL = 'https://youtube-qa-systemrender.com';
        this.currentVideoUrl = '';
        this.currentVideoTitle = '';
        this.isProcessing = false;

        this.initializeElements();
        this.attachEventListeners();
        this.initializePopup();
    }

    initializeElements() {
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
            statusIndicator: document.getElementById('statusIndicator'),
            statusText: document.getElementById('statusText'),
            loadingSpinner: document.querySelector('.loading-spinner'),
            btnText: document.querySelector('.btn-text')
        };
    }

    attachEventListeners() {
        this.elements.questionInput.addEventListener('input', () => {
            this.updateCharCount();
            this.toggleAskButton();
        });

        this.elements.questionInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                this.handleAskQuestion();
            }
        });

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
            await this.checkApiStatus();
            await this.getCurrentVideoInfo();
            await this.loadPreviousQuestion();
        } catch (error) {
            console.error('Initialization failed:', error);
            this.showError('Failed to initialize extension');
        }
    }

    async checkApiStatus() {
        try {
            const response = await fetch(`${this.API_BASE_URL}/health`, { method: 'GET' });
            if (response.ok) {
                this.updateApiStatus('online', 'API Connected');
            } else {
                this.updateApiStatus('offline', `API Error (${response.status})`);
            }
        } catch {
            this.updateApiStatus('offline', 'Connection Failed');
        }
    }

    updateApiStatus(status, text) {
        this.elements.statusIndicator.className = `status-indicator ${status}`;
        this.elements.statusText.textContent = text;
        this.elements.askButton.disabled = (status !== 'online');
    }

    async getCurrentVideoInfo() {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        if (!tab.url.includes('youtube.com/watch')) {
            this.showError('Navigate to a YouTube video');
            this.disableInterface();
            return;
        }
        const urlParams = new URLSearchParams(new URL(tab.url).search);
        const videoId = urlParams.get('v');
        if (!videoId) {
            this.showError('Could not detect YouTube video');
            this.disableInterface();
            return;
        }
        this.currentVideoUrl = tab.url;
        this.currentVideoTitle = tab.title.replace(' - YouTube', '');
        this.elements.videoTitle.textContent = this.currentVideoTitle;
        this.elements.videoUrl.textContent = this.currentVideoUrl;
        this.elements.videoInfo.style.display = 'block';
        this.elements.questionInput.disabled = false;
        this.toggleAskButton();
    }

    disableInterface() {
        this.elements.questionInput.disabled = true;
        this.elements.askButton.disabled = true;
    }

    updateCharCount() {
        const count = this.elements.questionInput.value.length;
        this.elements.charCount.textContent = count;
        this.elements.charCount.style.color = count > 450 ? 'var(--error-color)' :
                                             count > 400 ? 'var(--warning-color)' : '#666';
        this.toggleAskButton();
    }

    toggleAskButton() {
        const q = this.elements.questionInput.value.trim();
        this.elements.askButton.disabled = !q || !this.currentVideoUrl || q.length > 500 || this.isProcessing;
    }

    async handleAskQuestion() {
        const question = this.elements.questionInput.value.trim();
        if (!question) return this.showError('Please enter a question');

        this.isProcessing = true;
        this.setLoadingState(true);
        this.hideError();
        this.hideAnswer();

        try {
            const response = await fetch(`${this.API_BASE_URL}/api/v1/youtube-qa`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
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
            this.showError('Network error. Check connection.');
        } finally {
            this.isProcessing = false;
            this.setLoadingState(false);
            this.toggleAskButton();
        }
    }

    setLoadingState(loading) {
        this.elements.btnText.style.display = loading ? 'none' : 'block';
        this.elements.loadingSpinner.style.display = loading ? 'block' : 'none';
        this.elements.questionInput.disabled = loading;
    }

    showAnswer(answer) {
        this.elements.answerContent.textContent = answer;
        this.elements.answerSection.style.display = 'block';
        this.elements.answerSection.scrollIntoView({ behavior: 'smooth' });
    }

    hideAnswer() {
        this.elements.answerSection.style.display = 'none';
    }

    showError(message) {
        this.elements.errorContent.textContent = message;
        this.elements.errorSection.style.display = 'block';
        setTimeout(() => this.hideError(), 5000);
    }

    hideError() {
        this.elements.errorSection.style.display = 'none';
    }

    clearQuestion() {
        this.elements.questionInput.value = '';
        this.updateCharCount();
        this.hideAnswer();
        this.hideError();
        this.elements.questionInput.focus();
    }

    async copyAnswerToClipboard() {
        try {
            await navigator.clipboard.writeText(this.elements.answerContent.textContent);
            const orig = this.elements.copyAnswer.textContent;
            this.elements.copyAnswer.textContent = '✓ Copied';
            setTimeout(() => this.elements.copyAnswer.textContent = orig, 2000);
        } catch (e) {
            console.error('Copy failed:', e);
        }
    }

    async getSessionId() {
        const result = await chrome.storage.local.get(['sessionId']);
        if (result.sessionId) return result.sessionId;
        const sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        await chrome.storage.local.set({ sessionId });
        return sessionId;
    }

    async saveQuestion(question) {
        const timestamp = new Date().toISOString();
        await chrome.storage.local.set({
            lastQuestion: question,
            lastVideoUrl: this.currentVideoUrl,
            lastQuestionTime: timestamp
        });
    }

    async loadPreviousQuestion() {
        const result = await chrome.storage.local.get(['lastQuestion', 'lastVideoUrl', 'lastQuestionTime']);
        if (
            result.lastQuestion &&
            result.lastVideoUrl === this.currentVideoUrl &&
            new Date() - new Date(result.lastQuestionTime) < 60*60*1000
        ) {
            this.elements.questionInput.value = result.lastQuestion;
            this.updateCharCount();
            this.toggleAskButton();
        }
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new YouTubeQAPopup();
});
