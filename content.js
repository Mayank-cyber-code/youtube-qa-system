// YouTube Q&A Extension Content Script

class YouTubeQAContent {
    constructor() {
        this.videoUrl = '';
        this.videoTitle = '';
        this.isInitialized = false;
        
        this.init();
    }
    
    init() {
        if (this.isInitialized) return;
        
        // Wait for YouTube to load
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.setupVideoDetection());
        } else {
            this.setupVideoDetection();
        }
        
        this.isInitialized = true;
    }
    
    setupVideoDetection() {
        // Detect when user navigates to a new video
        this.detectVideoChange();
        
        // Listen for YouTube's navigation events
        window.addEventListener('yt-navigate-finish', () => {
            this.detectVideoChange();
        });
        
        // Fallback for URL changes
        let lastUrl = location.href;
        new MutationObserver(() => {
            const url = location.href;
            if (url !== lastUrl) {
                lastUrl = url;
                this.detectVideoChange();
            }
        }).observe(document, { subtree: true, childList: true });
    }
    
    detectVideoChange() {
        const url = window.location.href;
        
        if (url.includes('/watch?v=')) {
            const urlParams = new URLSearchParams(window.location.search);
            const videoId = urlParams.get('v');
            
            if (videoId && this.videoUrl !== url) {
                this.videoUrl = url;
                this.videoTitle = this.getVideoTitle();
                
                // Notify background script
                this.notifyVideoChange(videoId, this.videoTitle);
                
                // Add UI enhancement (optional)
                this.addQuickActionButton();
            }
        }
    }
    
    getVideoTitle() {
        // Try multiple selectors for video title
        const selectors = [
            'h1.ytd-video-primary-info-renderer',
            '#title h1',
            '.title.style-scope.ytd-video-primary-info-renderer',
            'h1.title'
        ];
        
        for (const selector of selectors) {
            const element = document.querySelector(selector);
            if (element && element.textContent) {
                return element.textContent.trim();
            }
        }
        
        return document.title.replace(' - YouTube', '');
    }
    
    notifyVideoChange(videoId, title) {
        // Send message to background script
        chrome.runtime.sendMessage({
            type: 'VIDEO_CHANGED',
            data: {
                videoId: videoId,
                videoUrl: this.videoUrl,
                videoTitle: title,
                timestamp: Date.now()
            }
        });
    }
    
    addQuickActionButton() {
        // Remove existing button if present
        const existingButton = document.querySelector('#youtube-qa-quick-button');
        if (existingButton) {
            existingButton.remove();
        }
        
        // Find the video actions container
        const actionsContainer = document.querySelector('#top-level-buttons-computed');
        if (!actionsContainer) return;
        
        // Create quick action button
        const quickButton = document.createElement('button');
        quickButton.id = 'youtube-qa-quick-button';
        quickButton.className = 'yt-spec-button-shape-next yt-spec-button-shape-next--tonal yt-spec-button-shape-next--mono yt-spec-button-shape-next--size-m';
        quickButton.innerHTML = `
            <div class="yt-spec-button-shape-next__button-text-content">
                <span class="yt-core-attributed-string">Ask AI</span>
            </div>
        `;
        
        quickButton.addEventListener('click', (e) => {
            e.preventDefault();
            this.openExtensionPopup();
        });
        
        // Add custom styles
        quickButton.style.cssText = `
            margin-left: 8px;
            background-color: #ff6b6b !important;
            color: white !important;
            border-radius: 18px !important;
            font-weight: 500 !important;
            transition: all 0.2s ease !important;
        `;
        
        quickButton.addEventListener('mouseenter', () => {
            quickButton.style.backgroundColor = '#ff5252 !important';
        });
        
        quickButton.addEventListener('mouseleave', () => {
            quickButton.style.backgroundColor = '#ff6b6b !important';
        });
        
        actionsContainer.appendChild(quickButton);
    }
    
    openExtensionPopup() {
        // This will trigger the popup to open
        chrome.runtime.sendMessage({
            type: 'OPEN_POPUP',
            data: {
                videoUrl: this.videoUrl,
                videoTitle: this.videoTitle
            }
        });
    }
}

// Initialize content script
if (window.location.hostname.includes('youtube.com')) {
    new YouTubeQAContent();
}