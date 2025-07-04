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
        interpreter = tf.lite.Interpreter(model_path="Nail_Segmentation_MobileNetV2.tflite")
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
        # Get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        interpreter.invoke()
        
        # Get output tensor
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
# 📸 Camera Functions
# =========================
def capture_image_from_camera():
    """Handle camera capture functionality"""
    camera_image = st.camera_input("📸 Take a photo of your nails")
    
    if camera_image is not None:
        # Convert the uploaded image to PIL format
        image = Image.open(camera_image).convert("RGB")
        return np.array(image)
    
    return None

def process_uploaded_image(uploaded_file):
    """Process uploaded image file"""
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        return np.array(image)
    return None

# =========================
# 🎨 Advanced Color Processing
# =========================
def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple"""
    return tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    """Convert RGB tuple to hex color"""
    return '#%02x%02x%02x' % rgb

def apply_advanced_nail_color(image, mask, color_hex, opacity=0.8, blend_mode='normal'):
    """Apply nail color with advanced blending options"""
    color_rgb = hex_to_rgb(color_hex)
    
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Create colored overlay
    colored = np.full_like(image, color_rgb, dtype=np.uint8)
    mask_bool = mask.astype(bool)
    
    if not np.any(mask_bool):
        return image, 0
    
    output = image.copy()
    
    # Apply different blending modes
    if blend_mode == 'overlay':
        # Overlay blending for more natural look
        base = image[mask_bool].astype(np.float32) / 255.0
        overlay = np.array(color_rgb, dtype=np.float32) / 255.0
        
        # Overlay formula
        blended = np.where(base < 0.5, 2 * base * overlay, 1 - 2 * (1 - base) * (1 - overlay))
        blended = (blended * 255).astype(np.uint8)
        
        # Apply opacity
        final_color = (opacity * blended + (1 - opacity) * image[mask_bool]).astype(np.uint8)
        output[mask_bool] = final_color
    else:
        # Normal blending
        final_color = (opacity * np.array(color_rgb) + (1 - opacity) * image[mask_bool]).astype(np.uint8)
        output[mask_bool] = final_color
    
    return output, np.sum(mask_bool)

def enhance_image_quality(image, brightness=1.0, contrast=1.0, saturation=1.0):
    """Enhance image quality with adjustable parameters"""
    pil_image = Image.fromarray(image)
    
    # Apply enhancements
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(brightness)
    
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(contrast)
    
    enhancer = ImageEnhance.Color(pil_image)
    pil_image = enhancer.enhance(saturation)
    
    return np.array(pil_image)

# =========================
# 📊 Analytics & Insights
# =========================
def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

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
    
    # Filter contours by area to remove noise
    significant_contours = [c for c in contours if cv2.contourArea(c) > 100]
    return int(len(significant_contours))

def create_confidence_plot(pred_mask):
    """Create confidence heatmap visualization"""
    fig = px.imshow(pred_mask, 
                    color_continuous_scale='RdYlBu_r',
                    title="AI Confidence Heatmap")
    fig.update_layout(
        title_x=0.5,
        height=400,
        showlegend=False
    )
    return fig

def create_fallback_mask(image_shape):
    """Create a simple fallback mask when model is not available"""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    # Create a simple circular mask in the center as demonstration
    h, w = image_shape[:2]
    center = (w//2, h//2)
    radius = min(w, h) // 8
    cv2.circle(mask, center, radius, 255, -1)
    return mask.astype(np.float32) / 255.0

# =========================
# 🎨 Color Palette Management
# =========================
def get_preset_colors():
    """Get predefined color palette"""
    return {
        "💋 Classic Red": "#DC143C",
        "🌸 Soft Pink": "#FFB6C1",
        "🪵 Nude Beige": "#F5E6D3",
        "🖤 Midnight Black": "#2C2C2C",
        "🔮 Royal Purple": "#663399",
        "💎 Pearl White": "#F8F8FF",
        "💙 Ocean Blue": "#006994",
        "💚 Forest Green": "#228B22",
        "🧡 Sunset Orange": "#FF8C00",
        "💛 Golden Yellow": "#FFD700",
        "🤎 Chocolate Brown": "#8B4513",
        "🩶 Steel Gray": "#708090"
    }

def get_seasonal_colors():
    """Get seasonal color recommendations"""
    seasons = {
        "🌸 Spring": ["#FFB6C1", "#98FB98", "#87CEEB", "#DDA0DD"],
        "☀️ Summer": ["#FF6347", "#32CD32", "#00CED1", "#FF69B4"],
        "🍂 Autumn": ["#CD853F", "#B22222", "#DAA520", "#8B4513"],
        "❄️ Winter": ["#2F4F4F", "#800080", "#DC143C", "#191970"]
    }
    return seasons

# =========================
# 📸 Image Processing Pipeline
# =========================
def preprocess_image(image, target_size=(256, 256)):
    """Preprocess image for model prediction"""
    image_resized = cv2.resize(image, target_size)
    image_normalized = image_resized.astype(np.float32) / 255.0
    return np.expand_dims(image_normalized, axis=0)

def postprocess_prediction(pred_mask, original_shape, threshold=0.5):
    """Postprocess model prediction"""
    # Handle different prediction output formats
    if len(pred_mask.shape) > 2:
        pred_mask = pred_mask.squeeze()
    
    # Apply threshold
    binary_mask = (pred_mask > threshold).astype(np.uint8)
    
    # Resize to original image size
    resized_mask = cv2.resize(binary_mask, (original_shape[1], original_shape[0]), 
                             interpolation=cv2.INTER_NEAREST)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    cleaned_mask = cv2.morphologyEx(resized_mask, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
    
    return cleaned_mask

# =========================
# 💾 Session State Management
# =========================
def initialize_session_state():
    """Initialize session state variables"""
    if 'processed_images' not in st.session_state:
        st.session_state.processed_images = []
    if 'user_preferences' not in st.session_state:
        st.session_state.user_preferences = {}
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None
    if 'image_source' not in st.session_state:
        st.session_state.image_source = 'upload'

def save_to_history(original_image, processed_image, color_used, analysis_data, source_method):
    """Save processing results to history"""
    history_entry = {
        "timestamp": datetime.now().isoformat(),
        "color_used": color_used,
        "analysis_data": convert_numpy_types(analysis_data),
        "original_image": original_image,
        "processed_image": processed_image,
        "source_method": source_method
    }
    st.session_state.analysis_history.append(history_entry)

# =========================
# 🌟 Main Application
# =========================
def main():
    initialize_session_state()
    
    # Header
    st.title("💅 AI Nail Color Studio Pro")
    st.markdown("### Transform your nails with AI-powered virtual try-on technology")
    
    # Sidebar for controls
    with st.sidebar:
        st.header("🎨 Studio Controls")
        
        # Model information
        with st.expander("🧠 AI Model Info"):
            model_info = get_model_info()
            for key, value in model_info.items():
                if isinstance(value, float):
                    st.metric(key.replace('_', ' ').title(), f"{value:.2f}")
                else:
                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")
        
        # Color selection
        st.subheader("🎨 Color Selection")
        color_mode = st.radio("Choose color mode:", 
                             ["Preset Colors", "Seasonal Colors", "Custom Color"])
        
        if color_mode == "Preset Colors":
            preset_colors = get_preset_colors()
            selected_label = st.selectbox("Select color:", list(preset_colors.keys()))
            selected_color = preset_colors[selected_label]
            
        elif color_mode == "Seasonal Colors":
            seasonal_colors = get_seasonal_colors()
            selected_season = st.selectbox("Select season:", list(seasonal_colors.keys()))
            season_colors = seasonal_colors[selected_season]
            
            # Create color grid
            cols = st.columns(2)
            for i, color in enumerate(season_colors):
                with cols[i % 2]:
                    if st.button(f"Color {i+1}", key=f"season_{i}"):
                        selected_color = color
                    st.markdown(f'<div style="background-color: {color}; height: 30px; border-radius: 15px; margin: 5px;"></div>', 
                               unsafe_allow_html=True)
            
            selected_color = season_colors[0]  # Default selection
            
        else:  # Custom Color
            selected_color = st.color_picker("Pick a custom color:", "#FF69B4")
        
        # Advanced settings
        st.subheader("⚙️ Advanced Settings")
        opacity = st.slider("Color Opacity", 0.1, 1.0, 0.8, 0.1)
        blend_mode = st.selectbox("Blend Mode", ["normal", "overlay"])
        confidence_threshold = st.slider("Detection Threshold", 0.1, 0.9, 0.5, 0.1)
        
        # Image enhancement
        st.subheader("✨ Image Enhancement")
        brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
        contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
        saturation = st.slider("Saturation", 0.5, 2.0, 1.0, 0.1)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Image input method selection
        st.markdown('<div class="input-method-selector">', unsafe_allow_html=True)
        st.subheader("📸 Choose Input Method")
        
        # Tabs for different input methods
        tab1, tab2 = st.tabs(["📂 Upload Image", "📷 Camera Capture"])
        
        image_np = None
        source_method = ""
        
        with tab1:
            st.markdown("**Upload an image from your device**")
            uploaded_file = st.file_uploader(
                "Choose an image file", 
                type=["jpg", "jpeg", "png"],
                help="Upload a clear image of your hand for best results"
            )
            
            if uploaded_file is not None:
                image_np = process_uploaded_image(uploaded_file)
                source_method = "Upload"
                st.success("✅ Image uploaded successfully!")
        
        with tab2:
            st.markdown("**Take a photo using your camera**")
            st.info("📋 **Camera Tips:**\n- Ensure good lighting\n- Keep your hand steady\n- Show nails clearly\n- Avoid shadows")
            
            camera_image = capture_image_from_camera()
            if camera_image is not None:
                image_np = camera_image
                source_method = "Camera"
                st.success("✅ Photo captured successfully!")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Process the image if available
        if image_np is not None:
            # Enhance image quality
            enhanced_image = enhance_image_quality(image_np, brightness, contrast, saturation)
            
            # Load model
            model = load_model()
            
            # Process image
            with st.spinner("🔮 AI is analyzing your nails..."):
                if model is not None:
                    # Preprocess
                    input_tensor = preprocess_image(enhanced_image)
                    
                    # Predict using TFLite
                    prediction = predict_with_tflite(model, input_tensor)
                    
                    if prediction is not None:
                        # Extract mask from prediction
                        pred_mask = prediction[0, ..., 0] if len(prediction.shape) > 3 else prediction[0]
                        
                        # Postprocess
                        processed_mask = postprocess_prediction(pred_mask, image_np.shape, confidence_threshold)
                    else:
                        st.warning("⚠️ Model prediction failed. Using fallback mask.")
                        pred_mask = create_fallback_mask(enhanced_image.shape)
                        processed_mask = postprocess_prediction(pred_mask, image_np.shape, confidence_threshold)
                else:
                    st.warning("⚠️ Model not available. Using fallback mask for demonstration.")
                    pred_mask = create_fallback_mask(enhanced_image.shape)
                    processed_mask = postprocess_prediction(pred_mask, image_np.shape, confidence_threshold)
                
                # Apply color
                colored_output, affected_pixels = apply_advanced_nail_color(
                    enhanced_image, processed_mask, selected_color, opacity, blend_mode
                )
                
                # Analyze results
                analysis_data = analyze_nail_area(processed_mask)
            
            # Display results
            st.subheader("🎯 Results")
            st.markdown(f"**Source:** {source_method} | **Color:** {selected_label if 'selected_label' in locals() else 'Custom'}")
            
            # Image comparison
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                st.markdown("**Original Image**")
                st.image(enhanced_image, use_column_width=True)
                
            with result_col2:
                st.markdown(f"**With {selected_label if 'selected_label' in locals() else 'Selected'} Color**")
                st.image(colored_output, use_column_width=True)
            
            # Save to history
            save_to_history(enhanced_image, colored_output, selected_color, analysis_data, source_method)
            
            # Analysis section
            st.subheader("📊 AI Analysis")
            
            analysis_col1, analysis_col2 = st.columns(2)
            
            with analysis_col1:
                st.metric("Coverage Area", f"{analysis_data['coverage_percentage']:.1f}%")
                st.metric("Detected Nails", analysis_data['estimated_nail_count'])
                st.metric("Affected Pixels", f"{affected_pixels:,}")
                st.metric("Input Method", source_method)
                
            with analysis_col2:
                # Confidence heatmap
                if 'pred_mask' in locals():
                    fig = create_confidence_plot(pred_mask)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Download section
            st.subheader("💾 Download Results")
            
            # Convert to bytes for download
            def convert_image_to_bytes(image):
                img_pil = Image.fromarray(image)
                buf = io.BytesIO()
                img_pil.save(buf, format='PNG')
                return buf.getvalue()
            
            download_col1, download_col2 = st.columns(2)
            
            with download_col1:
                st.download_button(
                    label="📥 Download Colored Image",
                    data=convert_image_to_bytes(colored_output),
                    file_name=f"nail_color_{source_method.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png"
                )
            
            with download_col2:
                # Generate analysis report with proper type conversion
                report = {
                    "timestamp": datetime.now().isoformat(),
                    "source_method": source_method,
                    "color_used": selected_color,
                    "analysis_data": convert_numpy_types(analysis_data),
                    "settings": {
                        "opacity": float(opacity),
                        "blend_mode": blend_mode,
                        "threshold": float(confidence_threshold)
                    }
                }
                
                st.download_button(
                    label="📊 Download Analysis Report",
                    data=json.dumps(report, indent=2),
                    file_name=f"analysis_report_{source_method.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        else:
            # Show instructions when no image is loaded
            st.markdown("""
            <div class="camera-preview">
                <h3>🎯 Ready to Transform Your Nails?</h3>
                <p>Choose one of the options above:</p>
                <ul style="text-align: left; display: inline-block;">
                    <li><strong>📂 Upload Image:</strong> Select a photo from your device</li>
                    <li><strong>📷 Camera Capture:</strong> Take a real-time photo</li>
                </ul>
                <p><em>For best results, ensure good lighting and clear nail visibility</em></p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("🏆 Features")
        
        features = [
            "🤖 AI-powered nail detection",
            "📷 Real-time camera capture",
            "📂 File upload support",
            "🎨 Advanced color blending",
            "📊 Real-time analysis",
            "✨ Image enhancement",
            "🌈 Seasonal color suggestions",
            "💾 Download results",
            "📈 Performance metrics",
            "🔮 Confidence visualization"
        ]
        
        for feature in features:
            st.markdown(f"• {feature}")
        
        # Color preview
        st.subheader("🎨 Current Color")
        st.markdown(f'<div style="background-color: {selected_color}; height: 60px; border-radius: 10px; margin: 10px 0; border: 2px solid #ddd;"></div>', 
                   unsafe_allow_html=True)
        st.code(selected_color)
        
        # Quick tips
        st.subheader("💡 Pro Tips")
        tips = [
            "📷 Use camera for real-time results",
            "💡 Ensure good lighting for better detection",
            "🧼 Keep nails clean and visible",
            "🎨 Try different blend modes for realistic effects",
            "🔧 Adjust opacity for subtle color changes",
            "🌈 Use seasonal colors for trending looks",
            "📱 Mobile camera works great too!"
        ]
        
        for tip in tips:
            st.markdown(f"• {tip}")
    
    # History section
    if st.session_state.analysis_history:
        st.subheader("📚 Processing History")
        
        # History stats
        history_stats = {
            "total_sessions": len(st.session_state.analysis_history),
            "camera_captures": sum(1 for h in st.session_state.analysis_history if h.get('source_method') == 'Camera'),
            "uploads": sum(1 for h in st.session_state.analysis_history if h.get('source_method') == 'Upload')
        }
        
        hist_stat_col1, hist_stat_col2, hist_stat_col3 = st.columns(3)
        with hist_stat_col1:
            st.metric("Total Sessions", history_stats["total_sessions"])
        with hist_stat_col2:
            st.metric("Camera Captures", history_stats["camera_captures"])
        with hist_stat_col3:
            st.metric("File Uploads", history_stats["uploads"])
        
        if st.button("🗑️ Clear History"):
            st.session_state.analysis_history = []
            st.success("History cleared!")
        
        # Display recent history
        for i, entry in enumerate(st.session_state.analysis_history[-5:]):  # Show last 5 entries
            source_icon = "📷" if entry.get('source_method') == 'Camera' else "📂"
            with st.expander(f"{source_icon} Session {i+1} - {entry['timestamp'][:19]} ({entry.get('source_method', 'Unknown')})"):
                hist_col1, hist_col2 = st.columns(2)
                
                with hist_col1:
                    st.image(entry['original_image'], caption="Original", use_column_width=True)
                
                with hist_col2:
                    st.image(entry['processed_image'], caption="Processed", use_column_width=True)
                
                st.json(entry['analysis_data'])

if __name__ == "__main__":
    main()