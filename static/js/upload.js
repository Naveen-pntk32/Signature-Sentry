// Upload functionality and UI enhancements for Signature Fraud Detection System

document.addEventListener('DOMContentLoaded', function() {
    const genuineInput = document.getElementById('genuine_signature');
    const testInput = document.getElementById('test_signature');
    const submitBtn = document.getElementById('submitBtn');
    
    if (genuineInput && testInput && submitBtn) {
        // Initialize upload handlers
        initializeUploadHandler('genuine_signature', 'genuineUploadArea');
        initializeUploadHandler('test_signature', 'testUploadArea');
        
        // Monitor both inputs for submit button state
        [genuineInput, testInput].forEach(input => {
            input.addEventListener('change', updateSubmitButton);
        });
    }
});

function initializeUploadHandler(inputId, uploadAreaId) {
    const input = document.getElementById(inputId);
    const uploadArea = document.getElementById(uploadAreaId);
    
    if (!input || !uploadArea) return;
    
    const placeholder = uploadArea.querySelector('.upload-placeholder');
    const preview = uploadArea.querySelector('.upload-preview');
    const previewImg = preview?.querySelector('img');
    const filename = preview?.querySelector('.filename');
    
    // File input change handler
    input.addEventListener('change', function(e) {
        handleFileSelect(e.target.files[0], uploadArea, placeholder, preview, previewImg, filename);
    });
    
    // Drag and drop handlers
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            const file = files[0];
            if (isValidImageFile(file)) {
                input.files = files;
                handleFileSelect(file, uploadArea, placeholder, preview, previewImg, filename);
            } else {
                showError('Please upload a valid image file (PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP)');
            }
        }
    });
}

function handleFileSelect(file, uploadArea, placeholder, preview, previewImg, filename) {
    if (!file) {
        showPlaceholder(placeholder, preview);
        return;
    }
    
    // Validate file
    const validation = validateFile(file);
    if (!validation.valid) {
        showError(validation.message);
        showPlaceholder(placeholder, preview);
        return;
    }
    
    // Show preview
    const reader = new FileReader();
    reader.onload = function(e) {
        if (previewImg && filename && placeholder && preview) {
            previewImg.src = e.target.result;
            filename.textContent = file.name;
            placeholder.style.display = 'none';
            preview.style.display = 'block';
            
            // Add success styling
            uploadArea.style.borderColor = 'var(--bs-success)';
        }
    };
    reader.readAsDataURL(file);
}

function showPlaceholder(placeholder, preview) {
    if (placeholder && preview) {
        placeholder.style.display = 'block';
        preview.style.display = 'none';
    }
}

function validateFile(file) {
    const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/bmp', 'image/tiff', 'image/webp'];
    const maxSize = 5 * 1024 * 1024; // 5MB
    
    if (!allowedTypes.includes(file.type)) {
        return {
            valid: false,
            message: 'Invalid file type. Please upload an image file (PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP).'
        };
    }
    
    if (file.size > maxSize) {
        return {
            valid: false,
            message: 'File size too large. Maximum size is 5MB.'
        };
    }
    
    return { valid: true };
}

function isValidImageFile(file) {
    const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/bmp', 'image/tiff', 'image/webp'];
    return allowedTypes.includes(file.type);
}

function updateSubmitButton() {
    const genuineInput = document.getElementById('genuine_signature');
    const testInput = document.getElementById('test_signature');
    const submitBtn = document.getElementById('submitBtn');
    
    if (genuineInput && testInput && submitBtn) {
        const bothFilesSelected = genuineInput.files.length > 0 && testInput.files.length > 0;
        submitBtn.disabled = !bothFilesSelected;
        
        if (bothFilesSelected) {
            submitBtn.classList.remove('btn-secondary');
            submitBtn.classList.add('btn-primary');
        } else {
            submitBtn.classList.remove('btn-primary');
            submitBtn.classList.add('btn-secondary');
        }
    }
}

function showError(message) {
    // Create or update error alert
    let existingAlert = document.querySelector('.alert-danger.file-error');
    
    if (existingAlert) {
        existingAlert.remove();
    }
    
    const alertHtml = `
        <div class="alert alert-danger alert-dismissible fade show file-error" role="alert">
            <i class="fas fa-exclamation-triangle me-2"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    
    const container = document.querySelector('.container');
    if (container) {
        container.insertAdjacentHTML('afterbegin', alertHtml);
    }
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
        const alert = document.querySelector('.alert-danger.file-error');
        if (alert) {
            const closeBtn = alert.querySelector('.btn-close');
            if (closeBtn) {
                closeBtn.click();
            }
        }
    }, 5000);
}

function showSuccess(message) {
    // Create success alert
    const alertHtml = `
        <div class="alert alert-success alert-dismissible fade show" role="alert">
            <i class="fas fa-check-circle me-2"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    
    const container = document.querySelector('.container');
    if (container) {
        container.insertAdjacentHTML('afterbegin', alertHtml);
    }
}

// File size formatter
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Image dimensions checker
function checkImageDimensions(file, callback) {
    const img = new Image();
    img.onload = function() {
        callback({
            width: this.width,
            height: this.height,
            valid: this.width >= 50 && this.height >= 50
        });
    };
    img.src = URL.createObjectURL(file);
}

// Enhanced file validation with dimensions
function validateFileEnhanced(file, callback) {
    const basicValidation = validateFile(file);
    
    if (!basicValidation.valid) {
        callback(basicValidation);
        return;
    }
    
    checkImageDimensions(file, function(dimensions) {
        if (!dimensions.valid) {
            callback({
                valid: false,
                message: 'Image dimensions too small. Minimum size is 50x50 pixels.'
            });
        } else {
            callback({
                valid: true,
                message: `Valid image (${dimensions.width}x${dimensions.height})`
            });
        }
    });
}

// Keyboard navigation support
document.addEventListener('keydown', function(e) {
    // Submit form with Ctrl+Enter
    if (e.ctrlKey && e.key === 'Enter') {
        const submitBtn = document.getElementById('submitBtn');
        if (submitBtn && !submitBtn.disabled) {
            submitBtn.click();
        }
    }
});

// Form validation before submit
document.addEventListener('submit', function(e) {
    const form = e.target;
    
    if (form.id === 'uploadForm') {
        const genuineInput = document.getElementById('genuine_signature');
        const testInput = document.getElementById('test_signature');
        
        if (!genuineInput.files.length || !testInput.files.length) {
            e.preventDefault();
            showError('Please select both signature images before submitting.');
            return false;
        }
        
        // Additional validation can be added here
        return true;
    }
});

// Clear any existing loading overlays on page load
window.addEventListener('load', function() {
    const existingOverlay = document.getElementById('loadingOverlay');
    if (existingOverlay) {
        existingOverlay.remove();
    }
});
