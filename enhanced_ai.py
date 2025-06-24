import cv2
import numpy as np
from PIL import Image
import io
from skimage import segmentation, feature, filters, morphology, measure
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy import ndimage

def remove_background_enhanced_ai(img_bytes, model_strength=7, detail_preservation=8, background_tolerance=5):
    """Enhanced AI Model - meniru kualitas remove.bg dengan teknik advanced"""
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    h, w = img_rgb.shape[:2]
    
    # === STAGE 1: ADVANCED PREPROCESSING ===
    # Multi-level noise reduction
    img_denoised = cv2.fastNlMeansDenoisingColored(img_rgb, None, 10, 10, 7, 21)
    img_bilateral = cv2.bilateralFilter(img_denoised, 15, 80, 80)
    
    # Adaptive histogram equalization
    lab = cv2.cvtColor(img_bilateral, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    img_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # === STAGE 2: HUMAN DETECTION & SEGMENTATION ===
    gray = cv2.cvtColor(img_enhanced, cv2.COLOR_RGB2GRAY)
    
    # SLIC superpixel segmentation (seperti yang digunakan remove.bg)
    segments = segmentation.slic(img_enhanced, n_segments=500, compactness=10, sigma=1)
    
    # Multi-scale edge detection
    edges_canny = cv2.Canny(gray, 50, 150, apertureSize=3, L2gradient=True)
    
    # Sobel edge detection
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges_sobel = np.sqrt(sobel_x**2 + sobel_y**2)
    edges_sobel = (edges_sobel / edges_sobel.max() * 255).astype(np.uint8)
    
    # Combine edges
    edges_combined = cv2.bitwise_or(edges_canny, edges_sobel)
    
    # === STAGE 3: INTELLIGENT FOREGROUND DETECTION ===
    # Histogram analysis untuk background detection
    border_width = max(10, min(w//20, h//20))
    
    # Extract border colors
    border_pixels = []
    border_pixels.extend(img_enhanced[:border_width, :].reshape(-1, 3))
    border_pixels.extend(img_enhanced[-border_width:, :].reshape(-1, 3))
    border_pixels.extend(img_enhanced[:, :border_width].reshape(-1, 3))
    border_pixels.extend(img_enhanced[:, -border_width:].reshape(-1, 3))
    border_pixels = np.array(border_pixels)
    
    # K-means clustering untuk background colors
    if len(border_pixels) > 100:
        kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
        border_labels = kmeans.fit_predict(border_pixels)
        bg_colors = kmeans.cluster_centers_
        
        # Find dominant background color
        unique, counts = np.unique(border_labels, return_counts=True)
        dominant_bg_idx = unique[np.argmax(counts)]
        dominant_bg_color = bg_colors[dominant_bg_idx]
    else:
        dominant_bg_color = np.mean(border_pixels, axis=0)
    
    # === STAGE 4: ADVANCED GRABCUT WITH MULTIPLE INITIALIZATIONS ===
    best_mask = None
    best_score = 0
    
    for iteration in range(model_strength):
        mask = np.zeros(gray.shape[:2], np.uint8)
        
        # Dynamic rectangle with varying margins
        margin_x = int(w * (0.05 + iteration * 0.015))
        margin_y = int(h * (0.05 + iteration * 0.015))
        rect = (margin_x, margin_y, w - 2*margin_x, h - 2*margin_y)
        
        if rect[2] > 0 and rect[3] > 0:
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            try:
                # Initial GrabCut
                cv2.grabCut(img_enhanced, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
                
                # Refine dengan user hints
                # Background hints - areas similar to dominant background
                bg_distance = np.sqrt(np.sum((img_enhanced - dominant_bg_color)**2, axis=2))
                bg_threshold = 40 + (background_tolerance * 5)
                
                # Mark definite background
                mask[bg_distance < bg_threshold] = cv2.GC_BGD
                
                # Mark definite foreground (center high-contrast areas)
                center_mask = np.zeros_like(gray, dtype=bool)
                center_mask[h//4:3*h//4, w//4:3*w//4] = True
                high_contrast = edges_combined > np.percentile(edges_combined, 75)
                mask[center_mask & high_contrast & (bg_distance > bg_threshold * 1.5)] = cv2.GC_FGD
                
                # Final GrabCut iteration with mask refinement
                mask_copy = mask.copy()
                try:
                    cv2.grabCut(img_enhanced, mask_copy, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
                    mask = mask_copy
                except:
                    pass  # Use initial mask if refinement fails
                
                # Evaluate mask quality
                binary_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
                
                # Scoring based on edge alignment and center coverage
                edge_score = np.sum(binary_mask * (edges_combined > 50))
                center_score = np.sum(binary_mask[center_mask])
                combined_score = edge_score + center_score * 2
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_mask = mask.copy()
                    
            except:
                continue
    
    if best_mask is None:
        # Fallback to simple thresholding
        gray_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        best_mask = np.where(gray_thresh > 0, 3, 0).astype(np.uint8)
    
    # === STAGE 5: ALPHA MATTING & EDGE REFINEMENT ===
    # Convert to binary mask
    binary_mask = np.where((best_mask == 2) | (best_mask == 0), 0, 1).astype(np.uint8)
    
    # === CONNECTED COMPONENTS ANALYSIS ===
    # Find connected components and keep only the largest ones
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    # Keep only components larger than minimum size
    min_component_size = (w * h) // 1000  # 0.1% of image area
    large_components = []
    
    for i in range(1, num_labels):  # Skip background (label 0)
        if stats[i, cv2.CC_STAT_AREA] > min_component_size:
            large_components.append(i)
    
    # Create clean mask with only large components
    clean_mask = np.zeros_like(binary_mask)
    for comp_id in large_components:
        clean_mask[labels == comp_id] = 1
    
    # If no large components found, use original mask
    if len(large_components) == 0:
        clean_mask = binary_mask
    
    # === ADVANCED MORPHOLOGICAL OPERATIONS ===
    kernel_size = max(3, min(7, min(w, h) // 150))
    
    # Progressive morphological cleaning with multiple kernel sizes
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size+2, kernel_size+2))
    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size+4, kernel_size+4))
    
    # Remove small holes and noise
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel_open)
    # Fill gaps
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_close)
    # Final smoothing
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_smooth)
    
    binary_mask = clean_mask
    
    # === ADVANCED ALPHA CHANNEL GENERATION ===
    alpha = binary_mask.astype(np.float32)
    
    # === IMPROVED TRIMAP GENERATION ===
    # Create more sophisticated trimap with multiple erosion/dilation levels
    kernel_sizes = [3, 5, 8, 12]
    eroded_masks = []
    dilated_masks = []
    
    for size in kernel_sizes:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        eroded = cv2.erode(binary_mask, kernel, iterations=1)
        dilated = cv2.dilate(binary_mask, kernel, iterations=1)
        eroded_masks.append(eroded)
        dilated_masks.append(dilated)
    
    # Create confidence zones
    sure_fg = eroded_masks[1]  # Medium erosion for sure foreground
    sure_bg = 1 - dilated_masks[2]  # Larger dilation for sure background
    unknown = 1 - sure_fg - sure_bg
    
    # === MULTI-LEVEL ALPHA MATTING ===
    # Distance-based alpha for unknown regions with improved blending
    if np.any(unknown):
        # Distance transforms with better normalization
        dist_fg = cv2.distanceTransform(1 - sure_fg, cv2.DIST_L2, 5)
        dist_bg = cv2.distanceTransform(sure_bg, cv2.DIST_L2, 5)
        
        # Gaussian weighting for smoother transitions
        dist_fg_norm = cv2.GaussianBlur(dist_fg, (5, 5), 2.0)
        dist_bg_norm = cv2.GaussianBlur(dist_bg, (5, 5), 2.0)
        
        # Alpha blending with sigmoid function for natural transitions
        unknown_mask = unknown > 0
        total_dist = dist_fg_norm[unknown_mask] + dist_bg_norm[unknown_mask] + 1e-6
        alpha_ratio = dist_bg_norm[unknown_mask] / total_dist
        
        # Apply sigmoid for smoother transitions
        alpha_sigmoid = 1 / (1 + np.exp(-10 * (alpha_ratio - 0.5)))
        alpha[unknown_mask] = alpha_sigmoid
    
    # === PROGRESSIVE EDGE REFINEMENT ===
    # Multi-scale bilateral filtering for edge preservation
    alpha_refined = alpha.copy()
    
    for scale, d, sigma_color, sigma_space in [(5, 50, 50), (9, 75, 75), (13, 100, 100)]:
        alpha_smooth = cv2.bilateralFilter(alpha_refined, d, sigma_color, sigma_space)
        
        # Edge-aware blending
        edge_strength = edges_combined.astype(np.float32) / 255.0
        edge_weight = cv2.GaussianBlur(edge_strength, (3, 3), 1.0)
        blend_factor = 0.2 + edge_weight * 0.6
        
        alpha_refined = alpha_refined * blend_factor + alpha_smooth * (1 - blend_factor)
    
    # === HAIR AND FINE DETAIL ENHANCEMENT ===
    # Multi-scale gradient detection for fine structures
    fine_detail_masks = []
    
    for kernel_size in [2, 3, 4]:
        gradient_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        gradient_mask = cv2.morphologyEx(edges_combined, cv2.MORPH_GRADIENT, gradient_kernel)
        fine_detail_masks.append(gradient_mask.astype(np.float32) / 255.0)
    
    # Combine fine details with different weights
    fine_details_combined = (fine_detail_masks[0] * 0.5 + 
                           fine_detail_masks[1] * 0.3 + 
                           fine_detail_masks[2] * 0.2)
    
    # Enhanced hair detail preservation
    hair_enhancement = fine_details_combined * alpha_refined * 0.6
    alpha_final = np.maximum(alpha_refined, hair_enhancement)
    
    # === FINAL SMOOTHING AND QUALITY ENHANCEMENT ===
    # Multi-pass edge-preserving smoothing
    alpha_smooth1 = cv2.bilateralFilter(alpha_final, 9, 75, 75)
    alpha_smooth2 = cv2.bilateralFilter(alpha_smooth1, 5, 50, 50)
    alpha_final = alpha_final * 0.6 + alpha_smooth1 * 0.3 + alpha_smooth2 * 0.1
    
    # Anti-aliasing with sub-pixel accuracy
    alpha_final = cv2.GaussianBlur(alpha_final, (2, 2), 0.8)
    
    # Ensure valid range
    alpha_final = np.clip(alpha_final, 0, 1)
    
    # === STAGE 6: COLOR ENHANCEMENT & COMPOSITION ===
    # Subtle color enhancement for visible areas
    result_rgb = img_enhanced.astype(np.float32)
    
    # Slight saturation boost for foreground
    hsv = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2HSV)
    saturation_boost = 1.0 + (alpha_final * 0.1)  # Up to 10% boost
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_boost, 0, 255)
    result_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # Create final RGBA image
    result = np.zeros((h, w, 4), dtype=np.uint8)
    result[:, :, :3] = result_rgb.astype(np.uint8)
    result[:, :, 3] = (alpha_final * 255).astype(np.uint8)
    
    # Anti-aliasing pada alpha channel
    result[:, :, 3] = cv2.medianBlur(result[:, :, 3], 3)
    
    # Convert to PIL dan return
    pil_image = Image.fromarray(result, 'RGBA')
    output_buffer = io.BytesIO()
    pil_image.save(output_buffer, format='PNG', optimize=True)
    return output_buffer.getvalue()