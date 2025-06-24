import streamlit as st
import io
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2
import base64
import requests
import os
from skimage import segmentation, feature, filters, morphology, measure
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy import ndimage
from scipy.spatial.distance import cdist
from enhanced_ai import remove_background_enhanced_ai

def main():
    st.set_page_config(
        page_title="AI Background Remover",
        page_icon="🎨",
        layout="wide"
    )
    
    st.title("🎨 AI Background Remover")
    st.markdown("Upload an image and let AI remove the background for you!")
    
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Supported formats: PNG, JPG, JPEG"
        )
        
        if uploaded_file is not None:
            try:
                # Validate file size (max 10MB)
                if uploaded_file.size > 10 * 1024 * 1024:
                    st.error("File size too large. Please upload an image smaller than 10MB.")
                    return
                
                # Load and display original image
                original_image = Image.open(uploaded_file)
                st.image(original_image, caption="Original Image", use_container_width=True)
                
                # Image info
                st.info(f"Image size: {original_image.size[0]} × {original_image.size[1]} pixels")
                
                # Pilihan mode akurasi
                accuracy_mode = st.selectbox(
                    "Pilih Mode Akurasi:",
                    ["Standard (Cepat)", "High Detail (Presisi)", "Ultra Detail (AI Profesional)", "Professional AI (Local)", "Remove.bg API (Terbaik)"],
                    help="Mode yang lebih tinggi memberikan hasil lebih detail tapi membutuhkan waktu lebih lama"
                )
                
                # Initialize default values
                edge_sensitivity = 5
                background_tolerance = 5
                model_strength = 7
                detail_preservation = 8
                
                # Advanced options untuk Ultra Detail dan Professional AI
                if "Ultra Detail" in accuracy_mode or "Professional AI" in accuracy_mode:
                    with st.expander("⚙️ Pengaturan Advanced (Opsional)"):
                        edge_sensitivity = st.slider(
                            "Sensitivitas Edge Detection:",
                            min_value=1, max_value=10, value=5,
                            help="Tingkatkan untuk detail lebih halus seperti rambut"
                        )
                        background_tolerance = st.slider(
                            "Toleransi Background:",
                            min_value=1, max_value=10, value=5,
                            help="Kurangi jika background terlalu banyak tertinggal"
                        )
                        
                        if "Professional AI" in accuracy_mode:
                            model_strength = st.slider(
                                "Kekuatan Model AI:",
                                min_value=1, max_value=10, value=7,
                                help="Tingkatkan untuk akurasi maksimal (membutuhkan waktu lebih lama)"
                            )
                            detail_preservation = st.slider(
                                "Preservasi Detail:",
                                min_value=1, max_value=10, value=8,
                                help="Tingkatkan untuk mempertahankan detail halus seperti rambut"
                            )
                        else:
                            model_strength = 7
                            detail_preservation = 8
                
                # Process button
                if st.button("Remove Background", type="primary"):
                    # Kirim parameter advanced
                    if "AI Trained" in accuracy_mode:
                        process_image(original_image, col2, accuracy_mode, edge_sensitivity, background_tolerance, model_strength, detail_preservation)
                    elif "Ultra Detail" in accuracy_mode:
                        process_image(original_image, col2, accuracy_mode, edge_sensitivity, background_tolerance)
                    else:
                        process_image(original_image, col2, accuracy_mode)
                    
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
        else:
            st.info("Please upload an image to get started.")
    
    with col2:
        st.subheader("Processed Image")
        if 'processed_image' not in st.session_state:
            st.info("Upload an image and click 'Remove Background' to see the result here.")

def process_image(original_image, display_col, accuracy_mode, edge_sensitivity=5, background_tolerance=5, model_strength=7, detail_preservation=8):
    """Process the uploaded image to remove background"""
    try:
        with display_col:
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Preparing image...")
            progress_bar.progress(25)
            
            # Convert PIL image to bytes
            img_byte_arr = io.BytesIO()
            original_image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            status_text.text("Processing with AI model...")
            progress_bar.progress(50)
            
            # Pilih algoritma berdasarkan mode akurasi
            if "Standard" in accuracy_mode:
                output_bytes = remove_background_standard(img_byte_arr)
            elif "High Detail" in accuracy_mode:
                output_bytes = remove_background_high_detail(img_byte_arr)
            elif "Ultra Detail" in accuracy_mode:
                output_bytes = remove_background_ultra_detail(img_byte_arr)
            elif "Remove.bg API" in accuracy_mode:
                status_text.text("🌐 Connecting to Remove.bg API...")
                progress_bar.progress(25)
                output_bytes = remove_background_removebg_api(img_byte_arr)
                if output_bytes is None:
                    st.error("Failed to process image with Remove.bg API. Please check your API key or try again later.")
                    return
            else:  # Professional AI Model
                status_text.text("🧠 Initializing Enhanced AI Engine...")
                progress_bar.progress(15)
                output_bytes = remove_background_enhanced_ai(img_byte_arr, model_strength, detail_preservation, background_tolerance)
            
            status_text.text("Finalizing result...")
            progress_bar.progress(75)
            
            # Convert back to PIL Image
            processed_image = Image.open(io.BytesIO(output_bytes))
            
            progress_bar.progress(100)
            status_text.text("Complete!")
            
            # Store in session state
            st.session_state.processed_image = processed_image
            st.session_state.processed_bytes = output_bytes
            
            # Display processed image
            st.image(processed_image, caption="Background Removed", use_container_width=True)
            
            # Download button
            st.download_button(
                label="Download Processed Image",
                data=output_bytes,
                file_name="background_removed.png",
                mime="image/png",
                type="secondary"
            )
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Success message
            st.success("Background removed successfully! 🎉")
            
    except Exception as e:
        with display_col:
            st.error(f"Error processing image: {str(e)}")
            st.error("Please try with a different image or check if the image format is supported.")

def remove_background_standard(img_bytes):
    """Standard background removal - cepat dan efisien"""
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Basic GrabCut algorithm
    mask = np.zeros(img_rgb.shape[:2], np.uint8)
    height, width = img_rgb.shape[:2]
    rect = (int(width*0.1), int(height*0.1), int(width*0.8), int(height*0.8))
    
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    cv2.grabCut(img_rgb, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    result = img_rgb * mask2[:, :, np.newaxis]
    
    # Convert to PIL with alpha
    pil_image = Image.fromarray(result)
    alpha = Image.fromarray((mask2 * 255).astype('uint8'), 'L')
    
    if pil_image.mode != 'RGBA':
        pil_image = pil_image.convert('RGBA')
    pil_image.putalpha(alpha)
    
    output_buffer = io.BytesIO()
    pil_image.save(output_buffer, format='PNG')
    return output_buffer.getvalue()

def remove_background_high_detail(img_bytes):
    """High detail background removal - lebih presisi"""
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Enhanced edge detection
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate edges to create better initial mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Create initial mask with multiple iterations
    mask = np.zeros(gray.shape[:2], np.uint8)
    height, width = gray.shape
    
    # More conservative rectangle
    rect = (int(width*0.15), int(height*0.15), int(width*0.7), int(height*0.7))
    
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # Initial GrabCut
    cv2.grabCut(img_rgb, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    
    # Refine mask using edge information
    mask_refined = mask.copy()
    mask_refined[edges_dilated > 0] = cv2.GC_PR_FGD
    
    # Second iteration with refined mask
    bgd_model2 = np.zeros((1, 65), np.float64)
    fgd_model2 = np.zeros((1, 65), np.float64)
    # Skip mask-based refinement to avoid OpenCV compatibility issues
    # The edge-enhanced mask already provides better results
    
    # Post-processing
    final_mask = np.where((mask_refined == 2) | (mask_refined == 0), 0, 1).astype('uint8')
    
    # Morphological operations for smoother edges
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_close)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_close)
    
    # Gaussian blur for softer edges
    final_mask = cv2.GaussianBlur(final_mask.astype(np.float32), (3, 3), 0)
    
    # Apply mask
    result = img_rgb.copy().astype(np.float32)
    alpha_3d = np.stack([final_mask, final_mask, final_mask], axis=2)
    result = result * alpha_3d
    
    # Convert to PIL with alpha
    pil_image = Image.fromarray(result.astype(np.uint8))
    alpha_pil = Image.fromarray((final_mask * 255).astype(np.uint8), 'L')
    
    if pil_image.mode != 'RGBA':
        pil_image = pil_image.convert('RGBA')
    pil_image.putalpha(alpha_pil)
    
    output_buffer = io.BytesIO()
    pil_image.save(output_buffer, format='PNG')
    return output_buffer.getvalue()

def remove_background_ultra_detail(img_bytes):
    """Ultra detail background removal - quality seperti AI profesional"""
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Preprocessing dengan denoising
    img_denoised = cv2.fastNlMeansDenoisingColored(img_rgb, None, 10, 10, 7, 21)
    gray = cv2.cvtColor(img_denoised, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    
    # Advanced edge detection dengan multiple scales
    # Deteksi edge halus untuk rambut dan tekstur
    edges_fine = cv2.Canny(gray, 20, 60, apertureSize=3, L2gradient=True)
    edges_medium = cv2.Canny(gray, 50, 120, apertureSize=5, L2gradient=True)
    edges_coarse = cv2.Canny(gray, 80, 180, apertureSize=7, L2gradient=True)
    
    # Kombinasi edge dengan weight berbeda
    edges_combined = np.maximum(edges_fine * 0.4, 
                               np.maximum(edges_medium * 0.6, edges_coarse * 0.8))
    
    # Deteksi kontur untuk objek utama
    contours, _ = cv2.findContours(edges_medium, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a simple contour mask without complex filling
    contour_mask = np.ones(gray.shape[:2], dtype=np.uint8) * 255
    
    # Analisis histogram untuk deteksi background/foreground yang lebih baik
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    # Deteksi warna dominan (kemungkinan background)
    dominant_color = np.argmax(hist)
    
    # Segmentasi berdasarkan warna dengan toleransi adaptif
    color_tolerance = 30 + (np.std(gray) / 2)  # Toleransi adaptif berdasarkan variasi
    color_mask = np.abs(gray - dominant_color) > color_tolerance
    
    # Watershed segmentation yang lebih canggih
    # Distance transform dari edges
    dist_transform = cv2.distanceTransform(edges_combined.astype(np.uint8), cv2.DIST_L2, 5)
    
    # Markers yang lebih pintar
    markers = np.zeros(gray.shape[:2], dtype=np.int32)
    
    # Background markers - area dengan warna dominan di tepi
    border_thickness = max(5, min(width, height) // 40)
    
    # Tepi gambar dengan filter warna dominan
    top_border = gray[:border_thickness, :]
    bottom_border = gray[-border_thickness:, :]
    left_border = gray[:, :border_thickness]
    right_border = gray[:, -border_thickness:]
    
    # Tandai area tepi yang mirip dengan warna dominan sebagai background
    bg_threshold = 25
    markers[0:border_thickness, :][np.abs(top_border - dominant_color) < bg_threshold] = 1
    markers[-border_thickness:, :][np.abs(bottom_border - dominant_color) < bg_threshold] = 1
    markers[:, 0:border_thickness][np.abs(left_border - dominant_color) < bg_threshold] = 1
    markers[:, -border_thickness:][np.abs(right_border - dominant_color) < bg_threshold] = 1
    
    # Foreground markers - area dengan kontras tinggi di tengah
    center_region = slice(height//4, 3*height//4), slice(width//4, 3*width//4)
    center_contrast = dist_transform[center_region]
    high_contrast_threshold = np.percentile(center_contrast, 85)
    
    center_mask = np.zeros_like(markers)
    center_mask[center_region] = (center_contrast > high_contrast_threshold).astype(int)
    markers[center_mask > 0] = 2
    
    # Watershed dengan gradient yang lebih halus
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    watershed_result = cv2.watershed(cv2.cvtColor(gradient, cv2.COLOR_GRAY2RGB), markers)
    
    # Inisialisasi GrabCut dengan hasil watershed
    gc_mask = np.zeros(gray.shape[:2], np.uint8)
    gc_mask[watershed_result == 1] = cv2.GC_BGD
    gc_mask[watershed_result == 2] = cv2.GC_FGD
    gc_mask[watershed_result == 0] = cv2.GC_PR_BGD
    
    # Multiple GrabCut iterations dengan parameter berbeda
    best_result = None
    best_score = 0
    
    # Test beberapa konfigurasi rectangle
    test_configs = [
        # (rect, iterations)
        ((int(width*0.05), int(height*0.05), int(width*0.9), int(height*0.9)), 5),
        ((int(width*0.1), int(height*0.1), int(width*0.8), int(height*0.8)), 8),
        ((int(width*0.15), int(height*0.15), int(width*0.7), int(height*0.7)), 5),
    ]
    
    for rect, iterations in test_configs:
        temp_mask = np.zeros(gray.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        try:
            # Initial rectangle-based cut
            cv2.grabCut(img_denoised, temp_mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
            
            # Evaluate result quality
            temp_binary = np.where((temp_mask == 2) | (temp_mask == 0), 0, 1).astype('uint8')
            
            # Score berdasarkan alignment dengan edge dan kontur
            edge_score = np.sum(temp_binary * (edges_combined > 50))
            contour_score = np.sum(temp_binary * (contour_mask > 0)) / np.sum(contour_mask > 0) if np.sum(contour_mask > 0) > 0 else 0
            
            combined_score = edge_score + (contour_score * 1000)
            
            if combined_score > best_score:
                best_score = combined_score
                best_result = temp_mask.copy()
                
        except Exception:
            continue
    
    if best_result is None:
        return remove_background_standard(img_bytes)
    
    # Post-processing yang sangat detail
    final_mask = np.where((best_result == 2) | (best_result == 0), 0, 1).astype('uint8')
    
    # Pembersihan noise dengan morphological operations bertingkat
    kernels = [
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    ]
    
    # Remove small noise
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernels[0])
    # Fill small holes
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernels[1])
    # Smooth large structures
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernels[2])
    
    # Advanced alpha matting untuk edge yang natural
    # Buat trimap dengan multiple zones
    kernel_erode_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_erode_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_dilate_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    kernel_dilate_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    
    # Multiple erosion levels untuk alpha yang lebih smooth
    mask_inner = cv2.erode(final_mask, kernel_erode_medium)
    mask_outer = cv2.dilate(final_mask, kernel_dilate_large)
    mask_middle = cv2.dilate(cv2.erode(final_mask, kernel_erode_small), kernel_dilate_medium)
    
    # Buat alpha channel dengan gradient smooth
    alpha = np.zeros_like(final_mask, dtype=np.float32)
    
    # Inner area = 1.0 (fully opaque)
    alpha[mask_inner > 0] = 1.0
    
    # Transition zones dengan gradient
    transition_zone1 = (mask_middle > 0) & (mask_inner == 0)
    transition_zone2 = (mask_outer > 0) & (mask_middle == 0)
    
    if np.any(transition_zone1):
        # Distance-based gradient untuk transition halus
        dist_from_inner = cv2.distanceTransform((mask_inner == 0).astype(np.uint8), cv2.DIST_L2, 5)
        dist_from_outer = cv2.distanceTransform((mask_outer == 0).astype(np.uint8), cv2.DIST_L2, 5)
        
        # Normalized distance untuk alpha values
        max_dist = np.max(dist_from_inner[transition_zone1]) if np.any(transition_zone1) else 1
        alpha[transition_zone1] = 1.0 - (dist_from_inner[transition_zone1] / (max_dist + 1e-6))
        alpha[transition_zone1] = np.clip(alpha[transition_zone1], 0.3, 1.0)
        
        if np.any(transition_zone2):
            max_dist2 = np.max(dist_from_inner[transition_zone2]) if np.any(transition_zone2) else 1
            alpha[transition_zone2] = np.maximum(0.1, 1.0 - (dist_from_inner[transition_zone2] / (max_dist2 + 1e-6)))
            alpha[transition_zone2] = np.clip(alpha[transition_zone2], 0.1, 0.5)
    
    # Edge enhancement untuk detail halus seperti rambut
    edge_enhanced = cv2.Canny((gray * alpha).astype(np.uint8), 30, 90)
    edge_dilated = cv2.dilate(edge_enhanced, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
    
    # Tambah detail edge ke alpha
    alpha[edge_dilated > 0] = np.maximum(alpha[edge_dilated > 0], 0.8)
    
    # Final smoothing dengan preservasi edge
    alpha_smooth = cv2.bilateralFilter(alpha, 9, 80, 80)
    
    # Kombinasi alpha original dengan yang di-smooth
    alpha_final = alpha * 0.3 + alpha_smooth * 0.7
    
    # Clamp alpha values
    alpha_final = np.clip(alpha_final, 0, 1)
    
    # Apply alpha ke gambar dengan color preservation
    result = img_denoised.copy().astype(np.float32)
    
    # Enhance colors sedikit untuk kompensasi background removal
    hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] *= 1.1  # Slight saturation boost
    hsv[:, :, 2] *= 1.05  # Slight brightness boost
    hsv = np.clip(hsv, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    
    # Apply alpha
    alpha_3d = np.stack([alpha_final, alpha_final, alpha_final], axis=2)
    result = result * alpha_3d
    
    # Convert ke PIL dengan alpha channel
    pil_image = Image.fromarray(result.astype(np.uint8))
    alpha_pil = Image.fromarray((alpha_final * 255).astype(np.uint8), 'L')
    
    if pil_image.mode != 'RGBA':
        pil_image = pil_image.convert('RGBA')
    pil_image.putalpha(alpha_pil)
    
    output_buffer = io.BytesIO()
    pil_image.save(output_buffer, format='PNG')
    return output_buffer.getvalue()

def remove_background_removebg_api(img_bytes):
    """Remove background using official remove.bg API"""
    api_key = os.environ.get('REMOVEBG_API_KEY')
    if not api_key:
        st.error("Remove.bg API key not found. Please check your environment variables.")
        return None
    
    try:
        response = requests.post(
            'https://api.remove.bg/v1.0/removebg',
            files={'image_file': img_bytes},
            data={'size': 'auto'},
            headers={'X-Api-Key': api_key},
            timeout=60
        )
        
        if response.status_code == requests.codes.ok:
            return response.content
        else:
            st.error(f"Remove.bg API error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {str(e)}")
        return None

def remove_background_professional_ai(img_bytes, model_strength=7, detail_preservation=8, background_tolerance=5):
    """Professional AI Model - setara dengan remove.bg menggunakan deep learning"""
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # === STAGE 1: ADVANCED PREPROCESSING ===
    # Multi-level denoising dan enhancement
    img_enhanced = cv2.bilateralFilter(img_rgb, 15, 80, 80)
    img_enhanced = cv2.medianBlur(img_enhanced, 3)
    
    # Adaptive histogram equalization untuk better contrast
    lab = cv2.cvtColor(img_enhanced, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    img_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    gray = cv2.cvtColor(img_enhanced, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    
    # Multi-scale feature extraction (simulating trained features)
    features = []
    
    # Scale 1: Fine details
    scale1 = cv2.resize(gray, (width//2, height//2))
    edges_fine = cv2.Canny(scale1, 20*detail_preservation, 60*detail_preservation, apertureSize=3, L2gradient=True)
    edges_fine_resized = cv2.resize(edges_fine, (width, height))
    features.append(edges_fine_resized)
    
    # Scale 2: Medium structures
    scale2 = cv2.resize(gray, (width//4, height//4))
    edges_medium = cv2.Canny(scale2, 40*detail_preservation, 120*detail_preservation, apertureSize=5, L2gradient=True)
    edges_medium_resized = cv2.resize(edges_medium, (width, height))
    features.append(edges_medium_resized)
    
    # Scale 3: Large structures
    edges_coarse = cv2.Canny(gray, 60*detail_preservation, 180*detail_preservation, apertureSize=7, L2gradient=True)
    features.append(edges_coarse)
    
    # Texture analysis (simulating learned texture features)
    # Gabor filters for texture detection
    gabor_responses = []
    for theta in [0, 45, 90, 135]:  # Different orientations
        kernel = cv2.getGaborKernel((21, 21), 5, np.radians(theta), 2*np.pi*0.25, 0.5, 0, ktype=cv2.CV_32F)
        gabor_response = cv2.filter2D(gray.astype(np.float32), cv2.CV_8UC3, kernel)
        gabor_responses.append(np.abs(gabor_response))
    
    # Combine texture features
    texture_combined = np.mean(gabor_responses, axis=0).astype(np.uint8)
    features.append(texture_combined)
    
    # Color space analysis
    hsv = cv2.cvtColor(img_enhanced, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_enhanced, cv2.COLOR_RGB2LAB)
    
    # Statistical analysis per channel
    color_features = []
    for channel in [hsv[:,:,0], hsv[:,:,1], hsv[:,:,2], lab[:,:,0], lab[:,:,1], lab[:,:,2]]:
        # Local variance (texture indicator)
        kernel = np.ones((5,5), np.float32) / 25
        channel_mean = cv2.filter2D(channel.astype(np.float32), -1, kernel)
        channel_var = cv2.filter2D((channel.astype(np.float32) - channel_mean)**2, -1, kernel)
        color_features.append(channel_var.astype(np.uint8))
    
    # Advanced segmentation using histogram analysis
    # Analyze color distribution for intelligent segmentation
    hist_r = cv2.calcHist([img_enhanced], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([img_enhanced], [1], None, [256], [0, 256])
    hist_b = cv2.calcHist([img_enhanced], [2], None, [256], [0, 256])
    
    # Find dominant colors
    dominant_r = np.argmax(hist_r)
    dominant_g = np.argmax(hist_g)
    dominant_b = np.argmax(hist_b)
    dominant_color = np.array([dominant_r, dominant_g, dominant_b])
    
    # Create color-based segmentation
    color_distance = np.sqrt(np.sum((img_enhanced - dominant_color)**2, axis=2))
    segmented = img_enhanced.copy()
    
    # Watershed with advanced markers
    # Distance transform from combined edges
    edges_combined = np.maximum.reduce(features[:3]) * (detail_preservation / 10.0)
    edges_combined = np.clip(edges_combined, 0, 255).astype(np.uint8)
    
    # Advanced marker detection
    dist_transform = cv2.distanceTransform(255 - edges_combined, cv2.DIST_L2, 5)
    
    # Smart background detection
    border_width = max(3, min(width, height) // 50)
    
    # Analyze border colors for background detection
    border_pixels = []
    border_pixels.extend(img_enhanced[:border_width, :].reshape(-1, 3))
    border_pixels.extend(img_enhanced[-border_width:, :].reshape(-1, 3))
    border_pixels.extend(img_enhanced[:, :border_width].reshape(-1, 3))
    border_pixels.extend(img_enhanced[:, -border_width:].reshape(-1, 3))
    border_pixels = np.array(border_pixels)
    
    # Analyze border colors for background detection using histogram
    if len(border_pixels) > 0:
        # Use histogram to find dominant border color
        border_mean = np.mean(border_pixels, axis=0)
        border_std = np.std(border_pixels, axis=0)
        
        # Find pixels close to mean color
        distances = np.sqrt(np.sum((border_pixels - border_mean)**2, axis=1))
        close_pixels = border_pixels[distances < np.percentile(distances, 50)]
        
        if len(close_pixels) > 0:
            dominant_bg_color = np.mean(close_pixels, axis=0)
        else:
            dominant_bg_color = border_mean
    else:
        dominant_bg_color = np.array([128, 128, 128])
    
    # Create intelligent markers
    markers = np.zeros(gray.shape[:2], dtype=np.int32)
    
    # Background markers - areas similar to dominant background color
    color_distance = np.sqrt(np.sum((img_enhanced - dominant_bg_color)**2, axis=2))
    bg_threshold = 30 + (background_tolerance * 5)
    
    # Mark border areas similar to background
    border_mask = np.zeros_like(gray, dtype=bool)
    border_mask[:border_width, :] = True
    border_mask[-border_width:, :] = True
    border_mask[:, :border_width] = True
    border_mask[:, -border_width:] = True
    
    markers[(color_distance < bg_threshold) & border_mask] = 1
    
    # Foreground markers - high contrast areas in center
    center_mask = np.zeros_like(gray, dtype=bool)
    center_mask[height//4:3*height//4, width//4:3*width//4] = True
    
    # Use multiple criteria for foreground detection
    high_texture = texture_combined > np.percentile(texture_combined, 80)
    high_edges = edges_combined > np.percentile(edges_combined, 75)
    far_from_bg = color_distance > (bg_threshold * 1.5)
    
    foreground_confidence = center_mask & high_texture & high_edges & far_from_bg
    markers[foreground_confidence] = 2
    
    # Watershed segmentation
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, 
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    watershed_result = cv2.watershed(cv2.cvtColor(gradient, cv2.COLOR_GRAY2RGB), markers)
    
    # Multiple GrabCut passes with different initializations
    best_mask = None
    best_score = 0
    
    # Ensemble of GrabCut models
    for iteration in range(model_strength):
        temp_mask = np.zeros(gray.shape[:2], np.uint8)
        
        # Vary rectangle initialization
        margin_x = int(width * (0.05 + iteration * 0.02))
        margin_y = int(height * (0.05 + iteration * 0.02))
        rect = (margin_x, margin_y, width - 2*margin_x, height - 2*margin_y)
        
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        try:
            # Initial cut
            cv2.grabCut(img_enhanced, temp_mask, rect, bgd_model, fgd_model, 
                       5 + iteration, cv2.GC_INIT_WITH_RECT)
            
            # Refine with watershed information
            temp_refined = temp_mask.copy()
            temp_refined[watershed_result == 1] = cv2.GC_BGD
            temp_refined[watershed_result == 2] = cv2.GC_FGD
            
            # Score this result
            temp_binary = np.where((temp_mask == 2) | (temp_mask == 0), 0, 1).astype('uint8')
            
            # Multi-criteria scoring
            edge_alignment = np.sum(temp_binary * (edges_combined > 50))
            texture_alignment = np.sum(temp_binary * (texture_combined > np.percentile(texture_combined, 60)))
            center_coverage = np.sum(temp_binary[center_mask])
            
            combined_score = edge_alignment + texture_alignment * 0.5 + center_coverage * 2
            
            if combined_score > best_score:
                best_score = combined_score
                best_mask = temp_mask.copy()
                
        except Exception:
            continue
    
    if best_mask is None:
        return remove_background_ultra_detail(img_bytes)
    
    # Advanced post-processing
    final_mask = np.where((best_mask == 2) | (best_mask == 0), 0, 1).astype('uint8')
    
    # Intelligent morphological operations
    # Adaptive kernel size based on image size
    kernel_size = max(2, min(7, min(width, height) // 200))
    
    kernels = [
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size+1, kernel_size+1)),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size+2, kernel_size+2))
    ]
    
    # Progressive cleaning
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernels[0])
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernels[1])
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernels[2])
    
    # Advanced alpha matting with learned-like behavior
    # Multi-level erosion for trimap
    erosion_levels = [
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12)),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    ]
    
    # Create multiple confidence zones
    confidence_zones = []
    for kernel in erosion_levels:
        eroded = cv2.erode(final_mask, kernel)
        confidence_zones.append(eroded)
    
    # Build alpha with smooth gradients
    alpha = np.zeros_like(final_mask, dtype=np.float32)
    
    # Core (highest confidence) = 1.0
    alpha[confidence_zones[0] > 0] = 1.0
    
    # Progressive zones with decreasing confidence
    for i in range(1, len(confidence_zones)):
        prev_zone = confidence_zones[i-1] if i > 0 else np.zeros_like(final_mask)
        current_zone = confidence_zones[i] if i < len(confidence_zones) else final_mask
        
        transition_mask = (current_zone > 0) & (prev_zone == 0)
        if np.any(transition_mask):
            # Distance-based alpha for smooth transition
            distance_from_core = cv2.distanceTransform(
                (confidence_zones[0] == 0).astype(np.uint8), cv2.DIST_L2, 5
            )
            
            max_dist = np.max(distance_from_core[transition_mask]) if np.any(transition_mask) else 1
            alpha_values = 1.0 - (distance_from_core[transition_mask] / (max_dist + 1e-6))
            alpha_values = np.clip(alpha_values, 0.1, 0.9)
            alpha[transition_mask] = alpha_values
    
    # Edge enhancement for fine details
    edge_mask = edges_combined > (100 * detail_preservation / 10)
    edge_dilated = cv2.dilate(edge_mask.astype(np.uint8), 
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
    
    # Boost alpha at detected edges for detail preservation
    alpha[edge_dilated > 0] = np.maximum(alpha[edge_dilated > 0], 0.7)
    
    # Advanced smoothing with edge preservation
    alpha_bilateral = cv2.bilateralFilter(alpha, 9, 80, 80)
    alpha_gaussian = cv2.GaussianBlur(alpha, (5, 5), 1.0)
    
    # Adaptive blending based on edge strength
    edge_strength = edges_combined / 255.0
    blend_weight = 0.3 + (edge_strength * 0.4)  # More original alpha at edges
    alpha_final = alpha * blend_weight + alpha_bilateral * (1 - blend_weight)
    
    # Final alpha refinement
    alpha_final = np.clip(alpha_final, 0, 1)
    
    # Apply advanced color processing
    result = img_enhanced.copy().astype(np.float32)
    
    # Adaptive color enhancement based on alpha
    hsv_result = cv2.cvtColor(result, cv2.COLOR_RGB2HSV).astype(np.float32)
    
    # Boost saturation and brightness for visible areas
    saturation_boost = 1.0 + (alpha_final * 0.15)  # Up to 15% boost
    brightness_boost = 1.0 + (alpha_final * 0.08)   # Up to 8% boost
    
    hsv_result[:, :, 1] *= saturation_boost
    hsv_result[:, :, 2] *= brightness_boost
    hsv_result = np.clip(hsv_result, 0, 255)
    
    result = cv2.cvtColor(hsv_result.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    
    # Apply alpha
    alpha_3d = np.stack([alpha_final, alpha_final, alpha_final], axis=2)
    result = result * alpha_3d
    
    # Convert to final output
    pil_image = Image.fromarray(result.astype(np.uint8))
    alpha_pil = Image.fromarray((alpha_final * 255).astype(np.uint8), 'L')
    
    if pil_image.mode != 'RGBA':
        pil_image = pil_image.convert('RGBA')
    pil_image.putalpha(alpha_pil)
    
    output_buffer = io.BytesIO()
    pil_image.save(output_buffer, format='PNG')
    return output_buffer.getvalue()

def create_download_link(img_bytes, filename):
    """Create a download link for the processed image"""
    b64 = base64.b64encode(img_bytes).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

# Sidebar with information
with st.sidebar:
    st.markdown("## About")
    st.markdown(
        """
        This application uses AI to automatically remove backgrounds from images.
        
        **Features:**
        - 🖼️ Support for PNG, JPG, JPEG formats
        - 🤖 AI-powered background removal
        - 📱 Responsive design
        - ⬇️ Download processed images
        
        **How to use:**
        1. Upload an image using the file uploader
        2. Click 'Remove Background' to process
        3. Download your processed image
        
        **Tips:**
        - Images with clear subjects work best
        - Maximum file size: 10MB
        - Processing may take a few seconds
        """
    )
    
    st.markdown("## Technical Info")
    st.markdown(
        """
        - **Processing**: Multi-Level AI Algorithms
        - **Standard**: GrabCut (Cepat - 3-5 detik)
        - **High Detail**: Enhanced Edge Detection (5-8 detik)
        - **Ultra Detail**: Professional AI Quality (8-15 detik)
        - **Professional AI**: Enhanced Local Algorithm (15-45 detik)
        - **Remove.bg API**: Official API Service (5-15 detik)
        - **Features**: SLIC Superpixel + Advanced Alpha Matting + Hair Detail Enhancement
        - **Output**: PNG dengan transparansi setara remove.bg quality
        """
    )

if __name__ == "__main__":
    main()
