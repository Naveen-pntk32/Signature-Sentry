import os
import logging
import uuid
from flask import render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
from app import app
from models import VerificationResult
from signature_model import signature_model
from image_processor import image_processor

logger = logging.getLogger(__name__)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page with upload form"""
    return render_template('index.html')

@app.route('/verify', methods=['POST'])
def verify_signatures():
    """Handle signature verification request"""
    try:
        # Check if files are present in request
        if 'genuine_signature' not in request.files or 'test_signature' not in request.files:
            flash('Please upload both genuine and test signature images.', 'error')
            return redirect(url_for('index'))
        
        genuine_file = request.files['genuine_signature']
        test_file = request.files['test_signature']
        
        # Check if files are selected
        if genuine_file.filename == '' or test_file.filename == '':
            flash('Please select both signature images.', 'error')
            return redirect(url_for('index'))
        
        # Validate file types
        if not (allowed_file(genuine_file.filename) and allowed_file(test_file.filename)):
            flash('Please upload valid image files (PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP).', 'error')
            return redirect(url_for('index'))
        
        # Generate unique filenames
        genuine_filename = secure_filename(f"genuine_{uuid.uuid4().hex}_{genuine_file.filename}")
        test_filename = secure_filename(f"test_{uuid.uuid4().hex}_{test_file.filename}")
        
        # Save uploaded files
        genuine_path = os.path.join(app.config['UPLOAD_FOLDER'], genuine_filename)
        test_path = os.path.join(app.config['UPLOAD_FOLDER'], test_filename)
        
        genuine_file.save(genuine_path)
        test_file.save(test_path)
        
        logger.info(f"Saved files: {genuine_path}, {test_path}")
        
        # Validate uploaded images
        genuine_valid, genuine_error = image_processor.validate_image(genuine_path)
        test_valid, test_error = image_processor.validate_image(test_path)
        
        if not genuine_valid:
            flash(f'Genuine signature image error: {genuine_error}', 'error')
            _cleanup_files([genuine_path, test_path])
            return redirect(url_for('index'))
        
        if not test_valid:
            flash(f'Test signature image error: {test_error}', 'error')
            _cleanup_files([genuine_path, test_path])
            return redirect(url_for('index'))
        
        # Process images
        try:
            genuine_processed = image_processor.preprocess_image(genuine_path)
            test_processed = image_processor.preprocess_image(test_path)
            
            logger.info("Images preprocessed successfully")
            
        except Exception as e:
            logger.error(f"Error preprocessing images: {str(e)}")
            flash(f'Error processing images: {str(e)}', 'error')
            _cleanup_files([genuine_path, test_path])
            return redirect(url_for('index'))
        
        # Perform signature verification
        try:
            similarity_score = signature_model.predict_similarity(genuine_processed, test_processed)
            is_genuine = signature_model.is_genuine(similarity_score)
            
            logger.info(f"Similarity score: {similarity_score}, Is genuine: {is_genuine}")
            
            # Create result object
            result = VerificationResult(
                genuine_image_path=genuine_filename,
                test_image_path=test_filename,
                similarity_score=similarity_score,
                is_genuine=is_genuine,
                threshold=signature_model.threshold
            )
            
            # Clean up uploaded files for security
            _cleanup_files([genuine_path, test_path])
            
            return render_template('results.html', result=result)
            
        except Exception as e:
            logger.error(f"Error during signature verification: {str(e)}")
            flash(f'Error during verification: {str(e)}', 'error')
            _cleanup_files([genuine_path, test_path])
            return redirect(url_for('index'))
    
    except Exception as e:
        logger.error(f"Unexpected error in verify_signatures: {str(e)}")
        flash('An unexpected error occurred. Please try again.', 'error')
        return redirect(url_for('index'))

@app.route('/api/verify', methods=['POST'])
def api_verify_signatures():
    """API endpoint for signature verification"""
    try:
        # Check if files are present in request
        if 'genuine_signature' not in request.files or 'test_signature' not in request.files:
            return jsonify({
                'error': 'Missing signature files',
                'message': 'Please provide both genuine_signature and test_signature files'
            }), 400
        
        genuine_file = request.files['genuine_signature']
        test_file = request.files['test_signature']
        
        # Validate files
        if genuine_file.filename == '' or test_file.filename == '':
            return jsonify({
                'error': 'Empty files',
                'message': 'Please select both signature images'
            }), 400
        
        if not (allowed_file(genuine_file.filename) and allowed_file(test_file.filename)):
            return jsonify({
                'error': 'Invalid file format',
                'message': 'Please upload valid image files',
                'allowed_formats': list(ALLOWED_EXTENSIONS)
            }), 400
        
        # Save and process files
        genuine_filename = secure_filename(f"api_genuine_{uuid.uuid4().hex}_{genuine_file.filename}")
        test_filename = secure_filename(f"api_test_{uuid.uuid4().hex}_{test_file.filename}")
        
        genuine_path = os.path.join(app.config['UPLOAD_FOLDER'], genuine_filename)
        test_path = os.path.join(app.config['UPLOAD_FOLDER'], test_filename)
        
        genuine_file.save(genuine_path)
        test_file.save(test_path)
        
        # Validate images
        genuine_valid, genuine_error = image_processor.validate_image(genuine_path)
        test_valid, test_error = image_processor.validate_image(test_path)
        
        if not genuine_valid or not test_valid:
            _cleanup_files([genuine_path, test_path])
            return jsonify({
                'error': 'Invalid images',
                'genuine_error': genuine_error if not genuine_valid else None,
                'test_error': test_error if not test_valid else None
            }), 400
        
        # Process and verify
        genuine_processed = image_processor.preprocess_image(genuine_path)
        test_processed = image_processor.preprocess_image(test_path)
        
        similarity_score = signature_model.predict_similarity(genuine_processed, test_processed)
        is_genuine = signature_model.is_genuine(similarity_score)
        
        # Create result
        result = VerificationResult(
            genuine_image_path=genuine_filename,
            test_image_path=test_filename,
            similarity_score=similarity_score,
            is_genuine=is_genuine,
            threshold=signature_model.threshold
        )
        
        # Clean up files
        _cleanup_files([genuine_path, test_path])
        
        return jsonify({
            'success': True,
            'result': result.to_dict()
        })
    
    except Exception as e:
        logger.error(f"API verification error: {str(e)}")
        return jsonify({
            'error': 'Verification failed',
            'message': str(e)
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': signature_model.model is not None,
        'upload_folder': app.config['UPLOAD_FOLDER']
    })

def _cleanup_files(file_paths):
    """Clean up temporary files"""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"Cleaned up file: {file_path}")
        except Exception as e:
            logger.warning(f"Could not clean up file {file_path}: {str(e)}")

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    flash('File too large. Maximum size is 16MB.', 'error')
    return redirect(url_for('index'))

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return render_template('index.html'), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    logger.error(f"Server error: {str(e)}")
    flash('An internal server error occurred. Please try again.', 'error')
    return render_template('index.html'), 500
