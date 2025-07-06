import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image, ImageEnhance
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
import base64
import io
import json
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration # NEW: Import for real-time video
import av # NEW: Import for video frame handling
from streamlit_webrtc import RTCConfiguration

rtc_config = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]}
    ]
})

# =========================
# 🎨 Configuration & Setup
# =========================
st.set_page_config(
    page_title="💅 AI Nail Color Studio",
    page_icon="💅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk tampilan yang lebih menarik
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stButton > button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        color: white;
        border: none;
        font-weight: bold;
    }
    .camera-button {
        background: linear-gradient(90deg, #6c5ce7, #fd79a8) !important;
        color: white !important;
        border: none !important;
        border-radius: 20px !important;
        padding: 0.5rem 1rem !important;
        font-weight: bold !important;
        margin: 0.5rem 0 !important;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .color-palette {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
    }
    .color-box {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        border: 2px solid #ddd;
        cursor: pointer;
    }
    .input-method-selector {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #ff6b6b;
    }
    .camera-preview {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# 🧠 AI Model Management
# =========================
@st.cache_resource
def load_model():
    """Load TFLite Interpreter with proper error handling"""
    try:
        interpreter = tf.lite.Interpreter(model_path="deep_mobilenet.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except FileNotFoundError:
        st.error("❌ Model file 'Nail_Segmentation_MobileNetV2.tflite' not found. Please ensure the model file is in the correct directory.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading TFLite model: {str(e)}")
        return None

def predict_with_tflite(interpreter, input_data):
    """Perform inference using TFLite interpreter"""
    try:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        return output_data
    except Exception as e:
        st.error(f"❌ Error during model inference: {str(e)}")
        return None

@st.cache_data
def get_model_info():
    """Get model metadata and performance metrics"""
    return {
        "model_name": "MobileNetV2",
        "input_size": (256, 256),
        "accuracy": 0.99,
        "precision": 0.97,
        "recall": 0.97,
        "Dice": 0.97,
    }

# =========================
# 📸 Image & Video Processing
# =========================
def capture_image_from_camera():
    """Handle camera capture functionality"""
    camera_image = st.camera_input("📸 Take a photo of your nails")
    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")
        return np.array(image)
    return None

def process_uploaded_image(uploaded_file):
    """Process uploaded image file"""
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        return np.array(image)
    return None

def preprocess_image(image, target_size=(256, 256)):
    """Preprocess image for model prediction"""
    image_resized = cv2.resize(image, target_size)
    image_normalized = image_resized.astype(np.float32) / 255.0
    return np.expand_dims(image_normalized, axis=0)

def postprocess_prediction(pred_mask, original_shape, threshold=0.5):
    """Postprocess model prediction"""
    if len(pred_mask.shape) > 2:
        pred_mask = pred_mask.squeeze()
    binary_mask = (pred_mask > threshold).astype(np.uint8)
    resized_mask = cv2.resize(binary_mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
    kernel = np.ones((3, 3), np.uint8)
    cleaned_mask = cv2.morphologyEx(resized_mask, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
    return cleaned_mask

# =========================
# 🎨 Advanced Color Processing
# =========================
def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple"""
    return tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

def apply_advanced_nail_color(image, mask, color_hex, opacity=0.8, blend_mode='normal'):
    """Apply nail color with advanced blending options"""
    color_rgb = hex_to_rgb(color_hex)
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    colored = np.full_like(image, color_rgb, dtype=np.uint8)
    mask_bool = mask.astype(bool)
    if not np.any(mask_bool):
        return image, 0
    
    output = image.copy()
    
    if blend_mode == 'overlay':
        base = image[mask_bool].astype(np.float32) / 255.0
        overlay_color = np.array(color_rgb, dtype=np.float32) / 255.0
        blended = np.where(base < 0.5, 2 * base * overlay_color, 1 - 2 * (1 - base) * (1 - overlay_color))
        blended = (blended * 255).astype(np.uint8)
        final_color = (opacity * blended + (1 - opacity) * image[mask_bool]).astype(np.uint8)
        output[mask_bool] = final_color
    else:
        final_color = (opacity * np.array(color_rgb) + (1 - opacity) * image[mask_bool]).astype(np.uint8)
        output[mask_bool] = final_color
    
    return output, np.sum(mask_bool)

def enhance_image_quality(image, brightness=1.0, contrast=1.0, saturation=1.0):
    """Enhance image quality with adjustable parameters"""
    pil_image = Image.fromarray(image)
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(brightness)
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(contrast)
    enhancer = ImageEnhance.Color(pil_image)
    pil_image = enhancer.enhance(saturation)
    return np.array(pil_image)

def create_gradient_effect(image_shape, gradient_type="radial", direction="vertical"):
    """Create gradient mask for realistic nail polish effect"""
    h, w = image_shape[:2]
    gradient = np.zeros((h, w), dtype=np.float32)
    
    if gradient_type == "radial":
        # Radial gradient from center
        center_x, center_y = w // 2, h // 2
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        for y in range(h):
            for x in range(w):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                gradient[y, x] = 1.0 - (dist / max_dist)
    
    elif gradient_type == "linear":
        if direction == "vertical":
            for y in range(h):
                gradient[y, :] = 1.0 - (y / h)
        else:  # horizontal
            for x in range(w):
                gradient[:, x] = 1.0 - (x / w)
    
    elif gradient_type == "oval":
        # Oval gradient for nail shape
        center_x, center_y = w // 2, h // 2
        for y in range(h):
            for x in range(w):
                dx = (x - center_x) / (w * 0.4)
                dy = (y - center_y) / (h * 0.6)
                dist = np.sqrt(dx**2 + dy**2)
                gradient[y, x] = np.clip(1.0 - dist, 0, 1)
    
    return gradient

def create_glossy_highlight(image_shape, intensity=0.3):
    """Create glossy highlight effect for nail polish"""
    h, w = image_shape[:2]
    highlight = np.zeros((h, w), dtype=np.float32)
    
    # Create multiple highlight spots
    highlight_spots = [
        (int(w * 0.3), int(h * 0.2), w * 0.15),  # Top-left highlight
        (int(w * 0.7), int(h * 0.3), w * 0.1),   # Top-right highlight
        (int(w * 0.5), int(h * 0.4), w * 0.08),  # Center highlight
    ]
    
    for spot_x, spot_y, radius in highlight_spots:
        for y in range(h):
            for x in range(w):
                dist = np.sqrt((x - spot_x)**2 + (y - spot_y)**2)
                if dist < radius:
                    highlight_val = intensity * (1 - dist / radius) ** 2
                    highlight[y, x] = max(highlight[y, x], highlight_val)
    
    return highlight

def apply_shimmer_effect(image, mask, shimmer_intensity=0.2):
    """Apply shimmer/metallic effect to nail area"""
    # Create random shimmer points
    shimmer_mask = np.random.rand(*mask.shape) < 0.1  # 10% of pixels get shimmer
    shimmer_mask = shimmer_mask & (mask > 0.5)  # Only on nail area
    
    # Apply shimmer
    shimmer_image = image.copy()
    shimmer_boost = np.random.uniform(0.8, 1.2, shimmer_image.shape)
    shimmer_image = (shimmer_image * shimmer_boost).clip(0, 255).astype(np.uint8)
    
    # Blend shimmer
    result = image.copy()
    result[shimmer_mask] = (
        shimmer_intensity * shimmer_image[shimmer_mask] + 
        (1 - shimmer_intensity) * image[shimmer_mask]
    ).astype(np.uint8)
    
    return result

def apply_advanced_nail_color_with_effects(image, mask, color_hex, opacity=0.8, 
                                         blend_mode='normal', effect_type='glossy',
                                         gradient_type='oval', shimmer=False):
    """
    Enhanced nail color application with realistic effects
    
    Args:
        image: Input image
        mask: Nail segmentation mask
        color_hex: Base color in hex format
        opacity: Color opacity (0-1)
        blend_mode: Blending mode ('normal', 'overlay', 'multiply')
        effect_type: Effect type ('matte', 'glossy', 'metallic', 'pearl')
        gradient_type: Gradient type ('none', 'radial', 'linear', 'oval')
        shimmer: Whether to apply shimmer effect
    """
    color_rgb = hex_to_rgb(color_hex)
    
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    mask_bool = mask.astype(bool)
    if not np.any(mask_bool):
        return image, 0
    
    output = image.copy()
    
    # Base color application
    base_color = np.array(color_rgb, dtype=np.float32)
    
    # Apply gradient if requested
    if gradient_type != 'none':
        gradient = create_gradient_effect(image.shape, gradient_type)
        gradient = cv2.resize(gradient, (image.shape[1], image.shape[0]))
        
        # Create gradient color variation
        darker_color = base_color * 0.7  # Darker shade
        lighter_color = np.minimum(base_color * 1.3, 255)  # Lighter shade
        
        # Apply gradient colors
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                if mask_bool[y, x]:
                    grad_val = gradient[y, x]
                    blended_color = grad_val * lighter_color + (1 - grad_val) * darker_color
                    base_color_at_pixel = blended_color
                    
                    # Apply base color with blending
                    if blend_mode == 'overlay':
                        base_pixel = image[y, x].astype(np.float32) / 255.0
                        overlay_color = base_color_at_pixel / 255.0
                        blended = np.where(base_pixel < 0.5, 
                                         2 * base_pixel * overlay_color, 
                                         1 - 2 * (1 - base_pixel) * (1 - overlay_color))
                        final_color = (blended * 255).astype(np.uint8)
                    elif blend_mode == 'multiply':
                        final_color = ((image[y, x].astype(np.float32) / 255.0) * 
                                     (base_color_at_pixel / 255.0) * 255).astype(np.uint8)
                    else:  # normal
                        final_color = base_color_at_pixel.astype(np.uint8)
                    
                    # Apply with opacity
                    output[y, x] = (opacity * final_color + (1 - opacity) * image[y, x]).astype(np.uint8)
    else:
        # Standard color application without gradient
        final_color = (opacity * base_color + (1 - opacity) * image[mask_bool]).astype(np.uint8)
        output[mask_bool] = final_color
    
    # Apply effects based on type
    if effect_type == 'glossy':
        # Add glossy highlight
        highlight = create_glossy_highlight(image.shape, intensity=0.4)
        highlight_mask = (highlight > 0) & mask_bool
        
        # Apply white highlights
        highlight_color = np.array([255, 255, 255], dtype=np.float32)
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                if highlight_mask[y, x]:
                    highlight_strength = highlight[y, x]
                    output[y, x] = (
                        highlight_strength * highlight_color + 
                        (1 - highlight_strength) * output[y, x]
                    ).astype(np.uint8)
    
    elif effect_type == 'metallic':
        # Add metallic shine
        output = apply_shimmer_effect(output, mask, shimmer_intensity=0.3)
        
        # Add metallic gradient
        metallic_gradient = create_gradient_effect(image.shape, "linear", "vertical")
        metallic_gradient = cv2.resize(metallic_gradient, (image.shape[1], image.shape[0]))
        
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                if mask_bool[y, x]:
                    metallic_boost = 1.0 + 0.2 * metallic_gradient[y, x]
                    output[y, x] = np.clip(output[y, x] * metallic_boost, 0, 255).astype(np.uint8)
    
    elif effect_type == 'pearl':
        # Add pearl effect with iridescent colors
        pearl_colors = [
            np.array([255, 192, 203]),  # Pink
            np.array([221, 160, 221]),  # Plum
            np.array([173, 216, 230]),  # Light blue
        ]
        
        # Create pearl pattern
        pearl_pattern = np.sin(np.linspace(0, 4*np.pi, image.shape[0]))[:, np.newaxis] * \
                       np.cos(np.linspace(0, 4*np.pi, image.shape[1]))[np.newaxis, :]
        pearl_pattern = (pearl_pattern + 1) / 2  # Normalize to 0-1
        
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                if mask_bool[y, x]:
                    pattern_val = pearl_pattern[y, x]
                    if pattern_val > 0.7:
                        pearl_color = pearl_colors[int(pattern_val * 3) % 3]
                        output[y, x] = (
                            0.3 * pearl_color + 0.7 * output[y, x]
                        ).astype(np.uint8)
    
    # Apply shimmer if requested
    if shimmer:
        output = apply_shimmer_effect(output, mask, shimmer_intensity=0.25)
    
    return output, np.sum(mask_bool)

# =========================
# 📊 Analytics & Insights
# =========================
def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, np.floating): return float(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    elif isinstance(obj, dict): return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [convert_numpy_types(i) for i in obj]
    else: return obj

def analyze_nail_area(mask):
    """Analyze detected nail area and provide insights"""
    total_pixels = int(mask.shape[0] * mask.shape[1])
    nail_pixels = int(np.sum(mask > 0.5))
    coverage_percentage = float((nail_pixels / total_pixels) * 100)
    return {
        "total_pixels": total_pixels,
        "nail_pixels": nail_pixels,
        "coverage_percentage": coverage_percentage,
        "estimated_nail_count": estimate_nail_count(mask)
    }

def estimate_nail_count(mask):
    """Estimate number of nails detected using contour analysis"""
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    significant_contours = [c for c in contours if cv2.contourArea(c) > 100]
    return int(len(significant_contours))

def create_confidence_plot(pred_mask):
    """Create confidence heatmap visualization"""
    fig = px.imshow(pred_mask, color_continuous_scale='RdYlBu_r', title="AI Confidence Heatmap")
    fig.update_layout(title_x=0.5, height=400, showlegend=False)
    return fig

def create_fallback_mask(image_shape):
    """Create a simple fallback mask when model is not available"""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    h, w = image_shape[:2]
    center, radius = (w//2, h//2), min(w, h) // 8
    cv2.circle(mask, center, radius, 255, -1)
    return mask.astype(np.float32) / 255.0

# =========================
# 🎨 Color Palette Management
# =========================
def get_preset_colors():
    """Get enhanced preset colors with effect recommendations"""
    return {
        "💋 Classic Red (Glossy)": {"color": "#DC143C", "effect": "glossy", "gradient": "oval"},
        "🌸 Soft Pink (Pearl)": {"color": "#FFB6C1", "effect": "pearl", "gradient": "radial"},
        "🪵 Nude Beige (Matte)": {"color": "#F5E6D3", "effect": "matte", "gradient": "none"},
        "🖤 Midnight Black (Glossy)": {"color": "#2C2C2C", "effect": "glossy", "gradient": "linear"},
        "🔮 Royal Purple (Metallic)": {"color": "#663399", "effect": "metallic", "gradient": "oval"},
        "💎 Pearl White (Pearl)": {"color": "#F8F8FF", "effect": "pearl", "gradient": "radial"},
        "💙 Ocean Blue (Glossy)": {"color": "#006994", "effect": "glossy", "gradient": "oval"},
        "💚 Forest Green (Metallic)": {"color": "#228B22", "effect": "metallic", "gradient": "linear"},
        "🧡 Sunset Orange (Glossy)": {"color": "#FF8C00", "effect": "glossy", "gradient": "radial"},
        "💛 Golden Yellow (Metallic)": {"color": "#FFD700", "effect": "metallic", "gradient": "oval"},
        "🤎 Chocolate Brown (Matte)": {"color": "#8B4513", "effect": "matte", "gradient": "none"},
        "🩶 Steel Gray (Metallic)": {"color": "#708090", "effect": "metallic", "gradient": "linear"}
    }

def get_seasonal_colors():
    """Get seasonal color recommendations"""
    return {
        "🌸 Spring": ["#FFB6C1", "#98FB98", "#87CEEB", "#DDA0DD"],
        "☀️ Summer": ["#FF6347", "#32CD32", "#00CED1", "#FF69B4"],
        "🍂 Autumn": ["#CD853F", "#B22222", "#DAA520", "#8B4513"],
        "❄️ Winter": ["#2F4F4F", "#800080", "#DC143C", "#191970"]
    }
   
def get_premium_effects():
    """Get premium effect options"""
    return {
        "✨ Matte": {"description": "Smooth, non-reflective finish", "shimmer": False},
        "💎 Glossy": {"description": "High-shine, reflective finish", "shimmer": False},
        "🌟 Metallic": {"description": "Shimmery, metallic finish", "shimmer": True},
        "🦪 Pearl": {"description": "Iridescent, color-changing finish", "shimmer": True},
        "✨ Shimmer": {"description": "Subtle sparkle effect", "shimmer": True}
    } 
# =========================
# 💾 Session State Management
# =========================
def initialize_session_state():
    """Initialize session state variables"""
    if 'analysis_history' not in st.session_state: st.session_state.analysis_history = []
    if 'current_image' not in st.session_state: st.session_state.current_image = None

def save_to_history(original_image, processed_image, color_used, analysis_data, source_method):
    """Save processing results to history"""
    history_entry = {
        "timestamp": datetime.now().isoformat(), "color_used": color_used,
        "analysis_data": convert_numpy_types(analysis_data), "original_image": original_image,
        "processed_image": processed_image, "source_method": source_method
    }
    st.session_state.analysis_history.append(history_entry)

# =========================
# 🌟 Main Application
# =========================
# =========================
# 🌟 Fixed Main Application
# =========================
def main():
    initialize_session_state()
    
    st.title("💅 AI Nail Color Studio Pro")
    st.markdown("### Transform your nails with AI-powered virtual try-on technology")
    
    model = load_model()

    with st.sidebar:
        st.header("🎨 Studio Controls")
        
        with st.expander("🧠 AI Model Info"):
            model_info = get_model_info()
            for key, value in model_info.items():
                if isinstance(value, float):
                    st.metric(key.replace('_', ' ').title(), f"{value:.2f}")
                else:
                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")
        
        st.subheader("🎨 Color Selection")
        color_mode = st.radio("Choose color mode:", ["Preset Colors", "Seasonal Colors", "Custom Color"], key="color_mode")
        
        selected_color = "#FF69B4"  # Default color
        effect_type = "glossy"  # Default effect
        gradient_type = "oval"  # Default gradient
        
        if color_mode == "Preset Colors":
            preset_colors = get_preset_colors()
            selected_label = st.selectbox("Select color:", list(preset_colors.keys()))
            selected_color = preset_colors[selected_label]["color"]
            effect_type = preset_colors[selected_label]["effect"]
            gradient_type = preset_colors[selected_label]["gradient"]
            
            # Show preset color info
            st.info(f"**Effect:** {effect_type.title()}")
            st.info(f"**Gradient:** {gradient_type.title()}")
            
        elif color_mode == "Seasonal Colors":
            seasonal_colors = get_seasonal_colors()
            selected_season = st.selectbox("Select season:", list(seasonal_colors.keys()))
            season_colors = seasonal_colors[selected_season]
            
            # Create color palette display
            st.markdown("**Available Colors:**")
            color_cols = st.columns(4)
            for i, color in enumerate(season_colors):
                with color_cols[i % 4]:
                    st.markdown(f'<div style="background-color: {color}; height: 30px; border-radius: 15px; margin: 2px; border: 1px solid #ddd;"></div>', unsafe_allow_html=True)
            
            color_options = {f'Color {i+1}': color for i, color in enumerate(season_colors)}
            selected_color_key = st.selectbox("Pick a seasonal color:", list(color_options.keys()))
            selected_color = color_options[selected_color_key]
            
            # Default settings for seasonal colors
            effect_type = "glossy"
            gradient_type = "oval"
            
        else:  # Custom Color
            selected_color = st.color_picker("Pick a custom color:", "#FF69B4")
            
            # Allow full customization for custom colors
            st.subheader("✨ Premium Effects")
            effect_options = get_premium_effects()
            selected_effect_key = st.selectbox("Effect Type:", list(effect_options.keys()))
            effect_config = effect_options[selected_effect_key]
            
            # Extract effect name from key (remove emoji and extra text)
            effect_type = selected_effect_key.split()[-1].lower()
            if effect_type == "shimmer":
                effect_type = "glossy"  # Use glossy as base for shimmer
            
            st.markdown(f"**{effect_config['description']}**")
            
            gradient_type = st.selectbox("Gradient Type:", 
                                    ["none", "oval", "radial", "linear"],
                                    index=1)  # Default to oval
        
        st.subheader("⚙️ Advanced Settings")
        opacity = st.slider("Color Opacity", 0.1, 1.0, 0.8, 0.1)
        blend_mode = st.selectbox("Blend Mode", ["normal", "overlay", "multiply"])
        confidence_threshold = st.slider("Detection Threshold", 0.1, 0.9, 0.5, 0.1)
        
        # Show additional effect options for custom colors
        if color_mode == "Custom Color":
            enable_shimmer = st.checkbox("Add Shimmer", 
                                    value=effect_config.get("shimmer", False))
        else:
            # Auto-enable shimmer for metallic and pearl effects
            enable_shimmer = effect_type in ["metallic", "pearl"]
        
        st.subheader("✨ Image Enhancement")
        brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
        contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
        saturation = st.slider("Saturation", 0.5, 2.0, 1.0, 0.1)
        
        # Display current effect summary
        st.subheader("🎯 Current Setup")
        st.markdown(f"**Effect:** {effect_type.title()}")
        st.markdown(f"**Gradient:** {gradient_type.title()}")
        st.markdown(f"**Shimmer:** {'Yes' if enable_shimmer else 'No'}")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="input-method-selector">', unsafe_allow_html=True)
        st.subheader("📸 Choose Input Method")
        
        tab1, tab2, tab3 = st.tabs(["📂 Upload Image", "📷 Camera Capture", "📹 Real-time Try-on"])
        
        image_np = None
        source_method = ""
        
        with tab1:
            st.markdown("**Upload an image from your device**")
            uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], 
                                           help="Upload a clear image of your hand for best results")
            if uploaded_file is not None:
                image_np = process_uploaded_image(uploaded_file)
                source_method = "Upload"
                st.session_state.current_image = image_np
        
        with tab2:
            st.markdown("**Take a photo using your camera**")
            st.info("📋 **Camera Tips:**\n- Ensure good lighting\n- Keep your hand steady\n- Show nails clearly")
            camera_image = capture_image_from_camera()
            if camera_image is not None:
                image_np = camera_image
                source_method = "Camera"
                st.session_state.current_image = image_np
        
        with tab3:
            st.markdown("**See the magic happen live!**")
            st.info("Allow camera access and place your hand in the frame. The color will be applied in real-time.")

            def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
                img = frame.to_ndarray(format="bgr24")
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                if model is None:
                    return av.VideoFrame.from_ndarray(img, format="bgr24")

                input_tensor = preprocess_image(img_rgb)
                prediction = predict_with_tflite(model, input_tensor)
                
                if prediction is not None:
                    pred_mask = prediction[0, ..., 0] if len(prediction.shape) > 3 else prediction[0]
                    processed_mask = postprocess_prediction(pred_mask, img_rgb.shape, confidence_threshold)
                    
                    # Use simplified effects for real-time performance
                    colored_output_rgb, _ = apply_advanced_nail_color_with_effects(
                        img_rgb, 
                        processed_mask, 
                        selected_color, 
                        opacity, 
                        blend_mode,
                        'glossy' if effect_type in ['metallic', 'pearl'] else effect_type,
                        'oval' if gradient_type == 'none' else gradient_type,
                        False  # Disable shimmer for real-time performance
                    )
                    
                    result_img = cv2.cvtColor(colored_output_rgb, cv2.COLOR_RGB2BGR)
                    return av.VideoFrame.from_ndarray(result_img, format="bgr24")
                else:
                    return av.VideoFrame.from_ndarray(img, format="bgr24")

            webrtc_streamer(
                key="realtime-segmentation",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=rtc_config,
                video_frame_callback=video_frame_callback,
                media_stream_constraints={
                    "video": {"width": 320, "height": 240, "frameRate": {"max": 5}},
                    "audio": False
                },
                async_processing=True,
            )
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Process static images from upload or camera capture
        if image_np is not None:
            enhanced_image = enhance_image_quality(image_np, brightness, contrast, saturation)
            
            with st.spinner("🔮 AI is analyzing your nails..."):
                processed_mask = None
                pred_mask_for_plot = None

                if model is not None:
                    input_tensor = preprocess_image(enhanced_image)
                    prediction = predict_with_tflite(model, input_tensor)
                    if prediction is not None:
                        pred_mask_for_plot = prediction[0, ..., 0] if len(prediction.shape) > 3 else prediction[0]
                        processed_mask = postprocess_prediction(pred_mask_for_plot, image_np.shape, confidence_threshold)
                    else:
                        st.warning("⚠️ Model prediction failed. Using fallback mask.")
                        processed_mask = create_fallback_mask(enhanced_image.shape)
                else:
                    st.warning("⚠️ Model not available. Using fallback mask.")
                    processed_mask = create_fallback_mask(enhanced_image.shape)
                
                # Apply the selected effect with all parameters
                colored_output, affected_pixels = apply_advanced_nail_color_with_effects(
                    enhanced_image, 
                    processed_mask, 
                    selected_color, 
                    opacity, 
                    blend_mode,
                    effect_type,
                    gradient_type,
                    enable_shimmer
                )
                
                analysis_data = analyze_nail_area(processed_mask)
            
            st.subheader("🎯 Results")
            st.markdown(f"**Source:** {source_method} | **Color:** {selected_color} | **Effect:** {effect_type.title()}")
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.markdown("**Original Image**")
                st.image(enhanced_image, use_column_width=True)
            with res_col2:
                st.markdown(f"**With {effect_type.title()} Effect**")
                st.image(colored_output, use_column_width=True)
            
            # Show effect details
            st.subheader("✨ Applied Effects")
            effect_col1, effect_col2, effect_col3 = st.columns(3)
            with effect_col1:
                st.metric("Effect Type", effect_type.title())
            with effect_col2:
                st.metric("Gradient", gradient_type.title())
            with effect_col3:
                st.metric("Shimmer", "Yes" if enable_shimmer else "No")
            
            save_to_history(enhanced_image, colored_output, selected_color, analysis_data, source_method)
            
            st.subheader("📊 AI Analysis")
            an_col1, an_col2 = st.columns(2)
            with an_col1:
                st.metric("Coverage Area", f"{analysis_data['coverage_percentage']:.1f}%")
                st.metric("Detected Nails", analysis_data['estimated_nail_count'])
                st.metric("Affected Pixels", f"{affected_pixels:,}")
            with an_col2:
                if pred_mask_for_plot is not None:
                    fig = create_confidence_plot(pred_mask_for_plot)
                    st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("💾 Download Results")
            def convert_image_to_bytes(img):
                img_pil = Image.fromarray(img)
                buf = io.BytesIO()
                img_pil.save(buf, format='PNG')
                return buf.getvalue()
            
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button(
                    label="📥 Download Colored Image", 
                    data=convert_image_to_bytes(colored_output), 
                    file_name=f"nail_{effect_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", 
                    mime="image/png"
                )
            with dl_col2:
                report = {
                    "timestamp": datetime.now().isoformat(),
                    "source_method": source_method,
                    "color_used": selected_color,
                    "effect_type": effect_type,
                    "gradient_type": gradient_type,
                    "shimmer_enabled": enable_shimmer,
                    "analysis_data": convert_numpy_types(analysis_data),
                    "settings": {
                        "opacity": float(opacity),
                        "blend_mode": blend_mode,
                        "threshold": float(confidence_threshold),
                        "brightness": float(brightness),
                        "contrast": float(contrast),
                        "saturation": float(saturation)
                    }
                }
                st.download_button(
                    label="📊 Download Analysis Report", 
                    data=json.dumps(report, indent=2), 
                    file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 
                    mime="application/json"
                )
        
        else:
            st.markdown("""
            <div class="camera-preview">
                <h3>🎯 Ready to Transform Your Nails?</h3>
                <p>Choose an input method above to get started!</p>
                <p><strong>✨ Try our premium effects:</strong></p>
                <ul>
                    <li>💎 <strong>Glossy</strong> - High-shine finish</li>
                    <li>🌟 <strong>Metallic</strong> - Shimmery metallic look</li>
                    <li>🦪 <strong>Pearl</strong> - Iridescent color-changing effect</li>
                    <li>✨ <strong>Matte</strong> - Smooth, non-reflective finish</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.subheader("🏆 Premium Features")
        features = [
            "🤖 AI-powered nail detection",
            "📹 **Real-time video try-on**",
            "💎 **Premium effects (Glossy, Metallic, Pearl)**",
            "🌈 **Advanced gradient options**",
            "✨ **Shimmer effects**",
            "📷 Professional camera capture",
            "📂 Multi-format file support",
            "🎨 Advanced color blending modes",
            "📊 Detailed AI analysis",
            "🖼️ Image enhancement tools",
            "🌸 Seasonal color collections",
            "💾 Professional export options"
        ]
        for feature in features:
            st.markdown(f"• {feature}")
        
        st.subheader("🎨 Current Selection")
        # Show current color with effect preview
        st.markdown(f'<div style="background-color: {selected_color}; height: 60px; border-radius: 10px; margin: 10px 0; border: 2px solid #ddd; position: relative;"><div style="position: absolute; bottom: 5px; right: 5px; background: rgba(0,0,0,0.7); color: white; padding: 2px 6px; border-radius: 3px; font-size: 10px;">{effect_type.upper()}</div></div>', unsafe_allow_html=True)
        st.code(selected_color)
        
        # Show color palette for current mode
        if color_mode == "Preset Colors":
            st.subheader("🎨 Preset Palette")
            preset_colors = get_preset_colors()
            for name, config in list(preset_colors.items())[:6]:  # Show first 6
                color = config["color"]
                effect = config["effect"]
                st.markdown(f'<div style="background-color: {color}; height: 25px; border-radius: 5px; margin: 2px 0; border: 1px solid #ddd; display: flex; align-items: center; padding-left: 10px; font-size: 12px; color: white; text-shadow: 1px 1px 1px rgba(0,0,0,0.5);">{name.split()[0]} {effect}</div>', unsafe_allow_html=True)
        
        elif color_mode == "Seasonal Colors":
            st.subheader("🌸 Seasonal Palette")
            seasonal_colors = get_seasonal_colors()
            for season, colors in seasonal_colors.items():
                st.markdown(f"**{season}**")
                color_row = st.columns(4)
                for i, color in enumerate(colors):
                    with color_row[i]:
                        st.markdown(f'<div style="background-color: {color}; height: 20px; border-radius: 10px; margin: 1px; border: 1px solid #ddd;"></div>', unsafe_allow_html=True)

    # Enhanced History Section
    if st.session_state.analysis_history:
        st.divider()
        st.subheader("📚 Processing History")
        
        history_col1, history_col2 = st.columns([3, 1])
        with history_col1:
            st.markdown(f"**Total Sessions:** {len(st.session_state.analysis_history)}")
        with history_col2:
            if st.button("🗑️ Clear History"):
                st.session_state.analysis_history = []
                st.rerun()
        
        for i, entry in enumerate(reversed(st.session_state.analysis_history[-5:])):
            source_icon = "📷" if entry.get('source_method') == 'Camera' else "📂"
            effect_info = f" - {entry.get('effect_type', 'unknown').title()}" if 'effect_type' in entry else ""
            
            with st.expander(f"{source_icon} Session {i+1} - {entry['timestamp'][:19]}{effect_info}"):
                hist_col1, hist_col2 = st.columns(2)
                with hist_col1:
                    st.image(entry['original_image'], caption="Original", use_column_width=True)
                with hist_col2:
                    st.image(entry['processed_image'], caption="Processed", use_column_width=True)
                
                # Show effect details if available
                if 'effect_type' in entry:
                    st.markdown(f"**Effect:** {entry['effect_type'].title()}")
                if 'gradient_type' in entry:
                    st.markdown(f"**Gradient:** {entry['gradient_type'].title()}")
                if 'shimmer_enabled' in entry:
                    st.markdown(f"**Shimmer:** {'Yes' if entry['shimmer_enabled'] else 'No'}")
                
                st.json(entry['analysis_data'])

if __name__ == "__main__":
    main()