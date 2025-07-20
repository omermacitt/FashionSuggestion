// Fashion AI - Enhanced Frontend Application

const API_BASE_URL = 'http://localhost:3000/api';

class FashionAIApp {
    constructor() {
        this.selectedFile = null;
        this.fileId = null;
        this.isProcessing = false;
        this.progress = 0;
        this.progressSteps = [
            { key: 'upload', title: 'Görsel Yükleniyor', duration: 20 },
            { key: 'detect', title: 'Obje Tespiti Yapılıyor', duration: 40 },
            { key: 'search', title: 'Benzer Ürünler Aranıyor', duration: 40 }
        ];
        
        this.initializeEventListeners();
        this.initializeAnimations();
    }

    initializeEventListeners() {
        // File input and upload area
        const fileInput = document.getElementById('file-input');
        const uploadArea = document.getElementById('upload-area');
        const analyzeBtn = document.getElementById('analyze-btn');
        const cancelBtn = document.getElementById('cancel-btn');

        if (fileInput) {
            fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        }

        if (uploadArea) {
            uploadArea.addEventListener('click', () => this.triggerFileSelect());
            uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
            uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
            uploadArea.addEventListener('drop', (e) => this.handleDrop(e));
        }

        if (analyzeBtn) {
            analyzeBtn.addEventListener('click', () => this.startAnalysis());
        }

        if (cancelBtn) {
            cancelBtn.addEventListener('click', () => this.resetForm());
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => this.handleKeyboard(e));
    }

    initializeAnimations() {
        // Animate elements on page load
        this.animateOnLoad();
        
        // Intersection Observer for scroll animations
        const observer = new IntersectionObserver(
            (entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add('fade-in');
                    }
                });
            },
            { threshold: 0.1 }
        );

        // Observe feature cards
        document.querySelectorAll('.feature-card').forEach(card => {
            observer.observe(card);
        });
    }

    animateOnLoad() {
        // Hero section animation
        const heroContent = document.querySelector('.hero-content');
        if (heroContent) {
            setTimeout(() => {
                heroContent.classList.add('fade-in');
            }, 300);
        }

        // Stagger feature badges animation
        const badges = document.querySelectorAll('.feature-badge');
        badges.forEach((badge, index) => {
            setTimeout(() => {
                badge.classList.add('fade-in');
            }, 500 + (index * 100));
        });
    }

    triggerFileSelect() {
        const fileInput = document.getElementById('file-input');
        if (fileInput) {
            fileInput.click();
        }
    }

    handleFileSelect(event) {
        const file = event.target.files[0];
        if (this.validateFile(file)) {
            this.selectedFile = file;
            this.showPreview(file);
        }
    }

    handleDragOver(event) {
        event.preventDefault();
        const uploadArea = document.getElementById('upload-area');
        uploadArea.classList.add('dragover');
    }

    handleDragLeave(event) {
        event.preventDefault();
        const uploadArea = document.getElementById('upload-area');
        uploadArea.classList.remove('dragover');
    }

    handleDrop(event) {
        event.preventDefault();
        const uploadArea = document.getElementById('upload-area');
        uploadArea.classList.remove('dragover');
        
        const files = event.dataTransfer.files;
        if (files.length > 0 && this.validateFile(files[0])) {
            this.selectedFile = files[0];
            this.showPreview(files[0]);
        }
    }

    handleKeyboard(event) {
        // ESC to cancel/reset
        if (event.key === 'Escape' && !this.isProcessing) {
            this.resetForm();
        }
        
        // Enter to start analysis (if file is selected)
        if (event.key === 'Enter' && this.selectedFile && !this.isProcessing) {
            this.startAnalysis();
        }
    }

    validateFile(file) {
        if (!file) {
            this.showNotification('Lütfen bir dosya seçin.', 'error');
            return false;
        }

        // Check file type
        const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
        if (!allowedTypes.includes(file.type)) {
            this.showNotification('Geçersiz dosya formatı. JPG, PNG veya WEBP dosyası seçin.', 'error');
            return false;
        }

        // Check file size (10MB)
        const maxSize = 10 * 1024 * 1024;
        if (file.size > maxSize) {
            this.showNotification('Dosya boyutu çok büyük. Maksimum 10MB olmalıdır.', 'error');
            return false;
        }

        return true;
    }

    showPreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const previewImage = document.getElementById('preview-image');
            const previewSection = document.getElementById('preview-section');
            const uploadArea = document.getElementById('upload-area');

            if (previewImage && previewSection) {
                previewImage.src = e.target.result;
                
                // Hide upload area and show preview
                uploadArea.style.display = 'none';
                previewSection.classList.remove('d-none');
                previewSection.classList.add('slide-up');

                // Update step status
                this.updateStepStatus('upload', 'completed');
            }
        };
        reader.readAsDataURL(file);
    }

    async startAnalysis() {
        if (!this.selectedFile || this.isProcessing) {
            return;
        }

        this.isProcessing = true;
        this.progress = 0;

        try {
            // Show progress section
            this.showProgressSection();
            
            // Update UI state
            this.updateAnalyzeButton(true);
            this.updateStepStatus('detect', 'active');

            // Step 1: Upload file
            this.updateProgress(0, 'Görsel yükleniyor...');
            const uploadResult = await this.uploadFile();
            
            if (!uploadResult.success) {
                throw new Error('Dosya yüklenemedi');
            }

            this.fileId = uploadResult.file_id;
            this.updateProgress(20, 'Dosya başarıyla yüklendi');

            // Step 2: Detect objects
            this.updateProgress(25, 'Obje tespiti başlatılıyor...');
            this.updateStepStatus('detect', 'active');
            
            const detectResult = await this.detectObjects();
            
            if (!detectResult.success) {
                throw new Error('Obje tespiti başarısız');
            }

            this.updateProgress(60, `${detectResult.detected_objects.length} obje tespit edildi`);
            this.updateStepStatus('detect', 'completed');
            this.updateStepStatus('search', 'active');

            // Step 3: Search similar products
            this.updateProgress(65, 'Benzer ürünler aranıyor...');
            
            const searchResult = await this.searchSimilar();
            
            if (!searchResult.success) {
                throw new Error('Benzer ürün araması başarısız');
            }

            this.updateProgress(100, 'Analiz tamamlandı!');
            this.updateStepStatus('search', 'completed');

            // Show success and redirect
            setTimeout(() => {
                this.showNotification('Analiz başarıyla tamamlandı! Sonuçlar sayfasına yönlendiriliyorsunuz...', 'success');
                
                setTimeout(() => {
                    window.location.href = `/results?file_id=${this.fileId}`;
                }, 1500);
            }, 1000);

        } catch (error) {
            console.error('Analysis error:', error);
            this.showNotification(`Hata: ${error.message}`, 'error');
            this.resetProgress();
        } finally {
            this.updateAnalyzeButton(false);
        }
    }

    async uploadFile() {
        const formData = new FormData();
        formData.append('file', this.selectedFile);

        const response = await fetch(`${API_BASE_URL}/load-file`, {
            method: 'POST',
            body: formData
        });

        return await response.json();
    }

    async detectObjects() {
        const response = await fetch(`${API_BASE_URL}/detect`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ file_id: this.fileId })
        });

        return await response.json();
    }

    async searchSimilar() {
        const response = await fetch(`${API_BASE_URL}/search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ file_id: this.fileId })
        });

        return await response.json();
    }

    showProgressSection() {
        const previewSection = document.getElementById('preview-section');
        const progressSection = document.getElementById('progress-section');

        if (previewSection && progressSection) {
            previewSection.style.opacity = '0.5';
            previewSection.style.pointerEvents = 'none';
            
            progressSection.classList.remove('d-none');
            progressSection.classList.add('fade-in');
        }
    }

    updateProgress(percentage, description) {
        const progressBar = document.getElementById('progress-bar');
        const progressPercentage = document.getElementById('progress-percentage');
        const progressDescription = document.getElementById('progress-description');

        if (progressBar) {
            progressBar.style.width = `${percentage}%`;
        }

        if (progressPercentage) {
            progressPercentage.textContent = `${percentage}%`;
        }

        if (progressDescription) {
            progressDescription.textContent = description;
        }

        this.progress = percentage;
    }

    updateStepStatus(stepKey, status) {
        const step = document.getElementById(`step-${stepKey}`);
        if (!step) return;

        const icon = step.querySelector('.step-icon');
        const stepElement = step;

        // Reset classes
        stepElement.classList.remove('completed', 'active');
        icon.classList.remove('completed', 'active');

        // Add new status
        if (status === 'completed') {
            stepElement.classList.add('completed');
            icon.classList.add('completed');
            icon.innerHTML = '<i class="bi bi-check-lg"></i>';
        } else if (status === 'active') {
            stepElement.classList.add('active');
            icon.classList.add('active');
        }
    }

    updateAnalyzeButton(isLoading) {
        const analyzeBtn = document.getElementById('analyze-btn');
        const analyzeSpinner = document.getElementById('analyze-spinner');

        if (!analyzeBtn) return;

        if (isLoading) {
            analyzeBtn.disabled = true;
            if (analyzeSpinner) {
                analyzeSpinner.classList.remove('d-none');
            }
            analyzeBtn.innerHTML = `
                <span class="spinner me-2"></span>
                Analiz Ediliyor...
            `;
        } else {
            analyzeBtn.disabled = false;
            if (analyzeSpinner) {
                analyzeSpinner.classList.add('d-none');
            }
            analyzeBtn.innerHTML = `
                <i class="bi bi-play-fill me-2"></i>
                Analizi Başlat
            `;
        }
    }

    resetForm() {
        // Reset all state
        this.selectedFile = null;
        this.fileId = null;
        this.isProcessing = false;
        this.progress = 0;

        // Reset UI elements
        const fileInput = document.getElementById('file-input');
        const uploadArea = document.getElementById('upload-area');
        const previewSection = document.getElementById('preview-section');
        const progressSection = document.getElementById('progress-section');

        if (fileInput) {
            fileInput.value = '';
        }

        if (uploadArea) {
            uploadArea.style.display = 'flex';
            uploadArea.classList.remove('dragover');
        }

        if (previewSection) {
            previewSection.classList.add('d-none');
            previewSection.style.opacity = '1';
            previewSection.style.pointerEvents = 'auto';
        }

        if (progressSection) {
            progressSection.classList.add('d-none');
        }

        // Reset steps
        ['upload', 'detect', 'search'].forEach(step => {
            this.resetStepStatus(step);
        });

        // Reset progress
        this.updateProgress(0, '');
        this.updateAnalyzeButton(false);
    }

    resetStepStatus(stepKey) {
        const step = document.getElementById(`step-${stepKey}`);
        if (!step) return;

        const icon = step.querySelector('.step-icon');
        const stepElement = step;

        stepElement.classList.remove('completed', 'active');
        icon.classList.remove('completed', 'active');

        // Reset to default icon
        const iconMap = {
            'upload': 'bi-check-lg',
            'detect': 'bi-search',
            'search': 'bi-collection'
        };

        icon.innerHTML = `<i class="${iconMap[stepKey]}"></i>`;

        // Upload step should always be completed if we're resetting
        if (stepKey === 'upload') {
            stepElement.classList.add('completed');
            icon.classList.add('completed');
        }
    }

    resetProgress() {
        this.isProcessing = false;
        
        const previewSection = document.getElementById('preview-section');
        const progressSection = document.getElementById('progress-section');

        if (previewSection) {
            previewSection.style.opacity = '1';
            previewSection.style.pointerEvents = 'auto';
        }

        if (progressSection) {
            progressSection.classList.add('d-none');
        }

        this.updateProgress(0, '');
        this.updateAnalyzeButton(false);
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <div class="notification-icon">
                    <i class="bi ${this.getNotificationIcon(type)}"></i>
                </div>
                <div class="notification-message">${message}</div>
                <button class="notification-close" onclick="this.parentElement.parentElement.remove()">
                    <i class="bi bi-x"></i>
                </button>
            </div>
        `;

        // Add styles if not already added
        this.addNotificationStyles();

        // Add to page
        document.body.appendChild(notification);

        // Auto remove after 5 seconds
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, 5000);
    }

    getNotificationIcon(type) {
        const icons = {
            'success': 'bi-check-circle-fill',
            'error': 'bi-exclamation-triangle-fill',
            'warning': 'bi-exclamation-circle-fill',
            'info': 'bi-info-circle-fill'
        };
        return icons[type] || icons.info;
    }

    addNotificationStyles() {
        if (document.getElementById('notification-styles')) return;

        const styles = document.createElement('style');
        styles.id = 'notification-styles';
        styles.textContent = `
            .notification {
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 1050;
                min-width: 300px;
                max-width: 500px;
                background: white;
                border-radius: 12px;
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
                border-left: 4px solid var(--primary);
                animation: slideInRight 0.3s ease-out;
            }
            
            .notification-success { border-left-color: var(--success); }
            .notification-error { border-left-color: var(--error); }
            .notification-warning { border-left-color: var(--warning); }
            
            .notification-content {
                display: flex;
                align-items: center;
                padding: 16px;
                gap: 12px;
            }
            
            .notification-icon {
                flex-shrink: 0;
                width: 24px;
                height: 24px;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            .notification-success .notification-icon { color: var(--success); }
            .notification-error .notification-icon { color: var(--error); }
            .notification-warning .notification-icon { color: var(--warning); }
            .notification-info .notification-icon { color: var(--primary); }
            
            .notification-message {
                flex: 1;
                font-size: 14px;
                line-height: 1.4;
                color: var(--gray-700);
            }
            
            .notification-close {
                background: none;
                border: none;
                color: var(--gray-400);
                cursor: pointer;
                padding: 4px;
                border-radius: 4px;
                transition: all 0.2s ease;
            }
            
            .notification-close:hover {
                background: var(--gray-100);
                color: var(--gray-600);
            }
            
            @keyframes slideInRight {
                from {
                    transform: translateX(100%);
                    opacity: 0;
                }
                to {
                    transform: translateX(0);
                    opacity: 1;
                }
            }
        `;
        
        document.head.appendChild(styles);
    }

    // Utility method for smooth scrolling
    smoothScrollTo(element) {
        if (element) {
            element.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    }
}

// Enhanced Results Page Functionality
class ResultsPageApp {
    constructor() {
        this.fileId = null;
        this.initializeResultsPage();
    }

    initializeResultsPage() {
        // Get file_id from URL
        const urlParams = new URLSearchParams(window.location.search);
        this.fileId = urlParams.get('file_id');

        if (this.fileId) {
            this.loadResults();
        }

        this.loadHistory();
    }

    async loadResults() {
        try {
            const response = await fetch(`${API_BASE_URL}/result/${this.fileId}`);
            const data = await response.json();

            if (data.error) {
                this.showError(data.error);
            } else {
                this.displayResults(data);
            }
        } catch (error) {
            console.error('Error loading results:', error);
            this.showError('Sonuçlar yüklenirken hata oluştu.');
        }
    }

    async loadHistory() {
        try {
            const response = await fetch(`${API_BASE_URL}/results/history`);
            const data = await response.json();
            this.displayHistory(data.history || []);
        } catch (error) {
            console.error('Error loading history:', error);
        }
    }

    displayResults(data) {
        // Implementation for displaying results
        // This will be enhanced based on the new API response format
        console.log('Results data:', data);
    }

    displayHistory(history) {
        // Implementation for displaying history
        console.log('History data:', history);
    }

    showError(message) {
        console.error('Error:', message);
    }
}

// Initialize appropriate app based on current page
document.addEventListener('DOMContentLoaded', () => {
    const currentPath = window.location.pathname;
    
    if (currentPath === '/' || currentPath === '/index.html') {
        new FashionAIApp();
    } else if (currentPath === '/results' || currentPath === '/results.html') {
        new ResultsPageApp();
    }
});

// Global utilities
window.FashionAI = {
    showNotification: (message, type) => {
        if (window.fashionApp) {
            window.fashionApp.showNotification(message, type);
        }
    }
};