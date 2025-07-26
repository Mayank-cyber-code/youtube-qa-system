// YouTube Q&A Extension Background Script (Service Worker)

class YouTubeQABackground {
    constructor() {
        this.currentVideoData = null;
        this.setupEventListeners();
    }
    
    setupEventListeners() {
        // Handle installation
        chrome.runtime.onInstalled.addListener((details) => {
            this.handleInstallation(details);
        });
        
        // Handle messages from content script and popup
        chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
            this.handleMessage(message, sender, sendResponse);
            return true; // Keep message channel open for async response
        });
        
        // Handle extension icon click
        chrome.action.onClicked.addListener((tab) => {
            this.handleIconClick(tab);
        });
        
        // Handle tab changes
        chrome.tabs.onActivated.addListener((activeInfo) => {
            this.handleTabActivated(activeInfo);
        });
        
        chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
            this.handleTabUpdated(tabId, changeInfo, tab);
        });
    }
    
    handleInstallation(details) {
        console.log('YouTube Q&A Extension installed/updated:', details.reason);
        
        if (details.reason === 'install') {
            // Set default settings
            chrome.storage.local.set({
                extensionEnabled: true,
                apiEndpoint: 'https://your-api-domain.com/api/v1',
                showQuickButtons: true,
                sessionId: this.generateSessionId()
            });
            
            // Open welcome page or show notification
            this.showWelcomeNotification();
        }
    }
    
    handleMessage(message, sender, sendResponse) {
        switch (message.type) {
            case 'VIDEO_CHANGED':
                this.handleVideoChanged(message.data);
                sendResponse({ success: true });
                break;
                
            case 'OPEN_POPUP':
                this.handleOpenPopup(message.data);
                sendResponse({ success: true });
                break;
                
            case 'GET_CURRENT_VIDEO':
                sendResponse({ 
                    success: true, 
                    data: this.currentVideoData 
                });
                break;
                
            case 'API_REQUEST':
                this.handleApiRequest(message.data)
                    .then(response => sendResponse({ success: true, data: response }))
                    .catch(error => sendResponse({ success: false, error: error.message }));
                break;
                
            default:
                sendResponse({ success: false, error: 'Unknown message type' });
        }
    }
    
    handleVideoChanged(videoData) {
        this.currentVideoData = videoData;
        
        // Update badge or icon if needed
        this.updateExtensionBadge();
        
        // Log for debugging
        console.log('Video changed:', videoData);
    }
    
    handleOpenPopup(data) {
        // Chrome doesn't allow programmatically opening popups
        // This is just for logging or other background tasks
        console.log('Popup open requested for:', data);
    }
    
    handleIconClick(tab) {
        // Check if we're on a YouTube video page
        if (tab.url && tab.url.includes('youtube.com/watch')) {
            // Popup will open automatically due to manifest configuration
            console.log('Extension icon clicked on YouTube video:', tab.url);
        } else {
            // Show notification or redirect to YouTube
            this.showNotYouTubeNotification();
        }
    }
    
    handleTabActivated(activeInfo) {
        // Clear current video data when switching tabs
        chrome.tabs.get(activeInfo.tabId, (tab) => {
            if (!tab.url || !tab.url.includes('youtube.com/watch')) {
                this.currentVideoData = null;
                this.updateExtensionBadge();
            }
        });
    }
    
    handleTabUpdated(tabId, changeInfo, tab) {
        // Handle URL changes within the same tab
        if (changeInfo.url && !changeInfo.url.includes('youtube.com/watch')) {
            this.currentVideoData = null;
            this.updateExtensionBadge();
        }
    }
    
    async handleApiRequest(requestData) {
        try {
            const settings = await chrome.storage.local.get(['apiEndpoint']);
            const apiEndpoint = settings.apiEndpoint || 'https://your-api-domain.com/api/v1';
            
            const response = await fetch(`${apiEndpoint}/youtube-qa`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });
            
            if (!response.ok) {
                throw new Error(`API request failed: ${response.status}`);
            }
            
            return await response.json();
            
        } catch (error) {
            console.error('API request error:', error);
            throw error;
        }
    }
    
    updateExtensionBadge() {
        if (this.currentVideoData) {
            chrome.action.setBadgeText({ text: '●' });
            chrome.action.setBadgeBackgroundColor({ color: '#4CAF50' });
            chrome.action.setTitle({ 
                title: `YouTube Q&A - Ready for: ${this.currentVideoData.videoTitle}` 
            });
        } else {
            chrome.action.setBadgeText({ text: '' });
            chrome.action.setTitle({ title: 'YouTube Q&A - Navigate to a YouTube video' });
        }
    }
    
    showWelcomeNotification() {
        chrome.notifications.create({
            type: 'basic',
            iconUrl: 'icons/icon48.png',
            title: 'YouTube Q&A Extension',
            message: 'Extension installed! Navigate to any YouTube video to get started.'
        });
    }
    
    showNotYouTubeNotification() {
        chrome.notifications.create({
            type: 'basic',
            iconUrl: 'icons/icon48.png',
            title: 'YouTube Q&A Extension',
            message: 'Please navigate to a YouTube video to use this extension.'
        });
    }
    
    generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
}

// Initialize background script
new YouTubeQABackground();