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
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# 🎨 Configuration & Setup
# =========================
st.set_page_config(
    page_title="💅 AI Nail Color Studio Pro",
    page_icon="💅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for better UI
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
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .error-box {
        background: #ffebee;
        border: 1px solid #f44336;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: #c62828;
    }
    .success-box {
        background: #e8f5e8;
        border: 1px solid #4caf50;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: #2e7d32;
    }
    .debug-info {
        background: #f3f4f6;
        border: 1px solid #d1d5db;
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        font-family: monospace;
        font-size: 0.85em;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# 🧠 Enhanced AI Model Management with Better Error Handling
# =========================
@st.cache_resource
def load_model():
    """Load TFLite Interpreter with comprehensive error handling"""
    try:
        # Check multiple possible model paths
        model_paths = [
            "test.tflite",
            "models/test.tflite", 
            "nail_model.tflite",
            "model.tflite"
        ]
        
        model_found = False
        for path in model_paths:
            if os.path.exists(path):
                try:
                    interpreter = tf.lite.Interpreter(model_path=path)
                    interpreter.allocate_tensors()
                    
                    # Get model details for debugging
                    input_details = interpreter.get_input_details()
                    output_details = interpreter.get_output_details()
                    
                    logger.info(f"Model loaded successfully from {path}")
                    logger.info(f"Input shape: {input_details[0]['shape']}")
                    logger.info(f"Output shape: {output_details[0]['shape']}")
                    
                    st.success(f"✅ Model loaded: {path}")
                    st.info(f"📊 Input shape: {input_details[0]['shape']}")
                    st.info(f"📊 Output shape: {output_details[0]['shape']}")
                    
                    model_found = True
                    return interpreter, input_details, output_details
                    
                except Exception as e:
                    logger.error(f"Error loading model from {path}: {str(e)}")
                    continue
        
        if not model_found:
            st.error("❌ No valid model file found!")
            st.markdown("""
            <div class="error-box">
                <strong>Model Loading Failed!</strong><br>
                Please ensure one of these files exists:
                <ul>
                    <li>test.tflite</li>
                    <li>models/test.tflite</li>
                    <li>nail_model.tflite</li>
                    <li>model.tflite</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            return None, None, None
            
    except Exception as e:
        logger.error(f"Critical error in model loading: {str(e)}")
        st.error(f"❌ Critical error loading model: {str(e)}")
        return None, None, None

# =========================
# 🎥 Enhanced Video Processing with Debug Mode
# =========================
class EnhancedVideoProcessor:
    def __init__(self):
        self.model_input_size = (256, 256)  # Common size for nail segmentation
        self.debug_mode = False
        self.frame_count = 0
        self.successful_predictions = 0
        
    def enable_debug(self):
        self.debug_mode = True
        logger.info("Debug mode enabled")
    
    def preprocess_frame(self, frame):
        """Enhanced preprocessing with debug info"""
        try:
            # Resize frame
            if frame.shape[:2] != self.model_input_size:
                processed = cv2.resize(frame, self.model_input_size)
            else:
                processed = frame.copy()
            
            # Normalize to [0, 1]
            processed = processed.astype(np.float32) / 255.0
            
            # Add batch dimension
            processed = np.expand_dims(processed, axis=0)
            
            if self.debug_mode:
                logger.info(f"Preprocessed shape: {processed.shape}")
                logger.info(f"Value range: {processed.min():.3f} - {processed.max():.3f}")
            
            return processed
            
        except Exception as e:
            logger.error(f"Preprocessing error: {str(e)}")
            return None
    
    def predict_with_model(self, interpreter, input_details, output_details, input_data):
        """Safe model prediction with error handling"""
        try:
            # Set input tensor
            interpreter.set_tensor(input_details[0]['index'], input_data)
            
            # Run inference
            interpreter.invoke()
            
            # Get output
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            self.successful_predictions += 1
            
            if self.debug_mode:
                logger.info(f"Prediction successful! Output shape: {output_data.shape}")
                logger.info(f"Output range: {output_data.min():.3f} - {output_data.max():.3f}")
            
            return output_data
            
        except Exception as e:
            logger.error(f"Model prediction error: {str(e)}")
            return None
    
    def postprocess_prediction(self, prediction, target_shape, threshold=0.5):
        """Enhanced postprocessing with debug"""
        try:
            if prediction is None:
                return np.zeros(target_shape[:2], dtype=np.uint8)
            
            # Handle different prediction shapes
            if len(prediction.shape) == 4:  # (batch, height, width, channels)
                mask = prediction[0, :, :, 0]
            elif len(prediction.shape) == 3:  # (batch, height, width)
                mask = prediction[0, :, :]
            else:
                mask = prediction
            
            # Apply threshold
            binary_mask = (mask > threshold).astype(np.uint8)
            
            # Resize to target shape
            if binary_mask.shape != target_shape[:2]:
                resized_mask = cv2.resize(
                    binary_mask, 
                    (target_shape[1], target_shape[0]), 
                    interpolation=cv2.INTER_NEAREST
                )
            else:
                resized_mask = binary_mask
            
            # Apply morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned_mask = cv2.morphologyEx(resized_mask, cv2.MORPH_CLOSE, kernel)
            cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
            
            if self.debug_mode:
                unique_values = np.unique(cleaned_mask)
                logger.info(f"Mask unique values: {unique_values}")
                logger.info(f"Positive pixels: {np.sum(cleaned_mask > 0)}")
            
            return cleaned_mask
            
        except Exception as e:
            logger.error(f"Postprocessing error: {str(e)}")
            return np.zeros(target_shape[:2], dtype=np.uint8)
    
    def get_stats(self):
        """Get processing statistics"""
        success_rate = (self.successful_predictions / max(self.frame_count, 1)) * 100
        return {
            "frames_processed": self.frame_count,
            "successful_predictions": self.successful_predictions,
            "success_rate": success_rate
        }

# =========================
# 🎨 Enhanced Color Application
# =========================
def apply_nail_color_enhanced(image, mask, color_hex, opacity=0.8):
    """Enhanced color application with better visualization"""
    try:
        # Convert hex to RGB
        color_rgb = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        
        # Ensure mask is proper format
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)
        
        # Create smooth mask
        mask_smooth = cv2.GaussianBlur(mask, (5, 5), 0)
        mask_norm = mask_smooth.astype(np.float32) / 255.0
        
        # Apply color with smooth blending
        colored_image = image.copy()
        
        for i in range(3):  # RGB channels
            colored_image[:, :, i] = (
                image[:, :, i] * (1 - mask_norm * opacity) + 
                color_rgb[i] * mask_norm * opacity
            )
        
        # Count affected pixels
        affected_pixels = np.sum(mask > 0)
        
        return colored_image.astype(np.uint8), affected_pixels
        
    except Exception as e:
        logger.error(f"Color application error: {str(e)}")
        return image, 0

# =========================
# 🎮 Main Application with Enhanced Debugging
# =========================
def main():
    st.title("💅 AI Nail Color Studio Pro - Debug Enhanced")
    st.markdown("### 🔧 Advanced Debugging & Quality Control")
    
    # Initialize session state
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
    if 'video_processor' not in st.session_state:
        st.session_state.video_processor = EnhancedVideoProcessor()
    
    # Load model with detailed feedback
    with st.spinner("🔄 Loading AI Model..."):
        model_data = load_model()
        interpreter, input_details, output_details = model_data
    
    # Sidebar controls
    with st.sidebar:
        st.header("🔧 Debug & Control Panel")
        
        # Debug controls
        with st.expander("🐛 Debug Settings", expanded=True):
            debug_mode = st.checkbox("Enable Debug Mode", value=st.session_state.debug_mode)
            if debug_mode != st.session_state.debug_mode:
                st.session_state.debug_mode = debug_mode
                if debug_mode:
                    st.session_state.video_processor.enable_debug()
            
            show_preprocessing = st.checkbox("Show Preprocessing Steps", value=False)
            show_mask_overlay = st.checkbox("Show Mask Overlay", value=True)
            log_level = st.selectbox("Log Level", ["INFO", "DEBUG", "ERROR"], index=0)
        
        # Model status
        st.subheader("🧠 Model Status")
        if interpreter is not None:
            st.success("✅ Model Loaded Successfully")
            if input_details:
                st.info(f"📊 Expected Input: {input_details[0]['shape']}")
            if output_details:
                st.info(f"📊 Output Shape: {output_details[0]['shape']}")
        else:
            st.error("❌ Model Not Loaded")
            st.stop()
        
        # Color selection
        st.subheader("🎨 Color Selection")
        selected_color = st.color_picker("Choose nail color:", "#FF69B4")
        opacity = st.slider("Color Opacity", 0.1, 1.0, 0.8, 0.1)
        threshold = st.slider("Detection Threshold", 0.1, 0.9, 0.5, 0.1)
        
        # Processing stats
        if st.session_state.video_processor:
            stats = st.session_state.video_processor.get_stats()
            st.subheader("📊 Processing Stats")
            st.metric("Frames Processed", stats["frames_processed"])
            st.metric("Successful Predictions", stats["successful_predictions"])
            st.metric("Success Rate", f"{stats['success_rate']:.1f}%")
    
    # Main content
    st.subheader("📸 Enhanced Testing Interface")
    
    tab1, tab2, tab3 = st.tabs(["🖼️ Static Image Test", "📷 Camera Test", "🎥 Optimized Real-time"])
    
    with tab1:
        st.markdown("**Upload and test with static images**")
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Load and display original image
            image = Image.open(uploaded_file).convert("RGB")
            image_np = np.array(image)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image_np, caption="Original Image", use_column_width=True)
            
            # Process image
            if st.button("🔮 Process Image"):
                with st.spinner("Processing..."):
                    processor = st.session_state.video_processor
                    if debug_mode:
                        processor.enable_debug()
                    
                    # Preprocess
                    processed_input = processor.preprocess_frame(image_np)
                    
                    if processed_input is not None:
                        # Predict
                        prediction = processor.predict_with_model(
                            interpreter, input_details, output_details, processed_input
                        )
                        
                        if prediction is not None:
                            # Postprocess
                            mask = processor.postprocess_prediction(
                                prediction, image_np.shape, threshold
                            )
                            
                            # Apply color
                            colored_result, affected_pixels = apply_nail_color_enhanced(
                                image_np, mask, selected_color, opacity
                            )
                            
                            # Display results
                            with col2:
                                if show_mask_overlay:
                                    # Show mask overlay
                                    overlay = image_np.copy()
                                    overlay[mask > 0] = [255, 0, 0]  # Red overlay
                                    combined = cv2.addWeighted(image_np, 0.7, overlay, 0.3, 0)
                                    st.image(combined, caption="Mask Overlay", use_column_width=True)
                                else:
                                    st.image(colored_result, caption="Colored Result", use_column_width=True)
                            
                            # Show metrics
                            st.subheader("📊 Results")
                            metric_col1, metric_col2, metric_col3 = st.columns(3)
                            with metric_col1:
                                st.metric("Affected Pixels", f"{affected_pixels:,}")
                            with metric_col2:
                                coverage = (affected_pixels / (image_np.shape[0] * image_np.shape[1])) * 100
                                st.metric("Coverage", f"{coverage:.2f}%")
                            with metric_col3:
                                st.metric("Image Size", f"{image_np.shape[1]}x{image_np.shape[0]}")
                            
                            # Debug information
                            if debug_mode:
                                st.subheader("🔍 Debug Information")
                                debug_col1, debug_col2 = st.columns(2)
                                with debug_col1:
                                    st.markdown("**Preprocessing:**")
                                    st.markdown(f"- Input shape: {processed_input.shape}")
                                    st.markdown(f"- Value range: {processed_input.min():.3f} - {processed_input.max():.3f}")
                                with debug_col2:
                                    st.markdown("**Prediction:**")
                                    st.markdown(f"- Output shape: {prediction.shape}")
                                    st.markdown(f"- Prediction range: {prediction.min():.3f} - {prediction.max():.3f}")
                                    st.markdown(f"- Mask pixels: {np.sum(mask > 0)}")
                        else:
                            st.error("❌ Model prediction failed!")
                    else:
                        st.error("❌ Image preprocessing failed!")
    
    with tab2:
        st.markdown("**Camera capture for testing**")
        camera_image = st.camera_input("Take a photo")
        
        if camera_image is not None:
            image = Image.open(camera_image).convert("RGB")
            image_np = np.array(image)
            st.image(image_np, caption="Captured Image", use_column_width=True)
            
            # Add similar processing as tab1
            if st.button("🔮 Process Camera Image"):
                # Similar processing logic as tab1
                st.info("Processing camera image...")
    
    with tab3:
        st.markdown("**⚡ Optimized Real-time Processing**")
        
        # Quality settings
        col1, col2, col3 = st.columns(3)
        with col1:
            fps_target = st.selectbox("Target FPS", [15, 20, 30], index=1)
        with col2:
            resolution = st.selectbox("Resolution", ["480p", "720p", "1080p"], index=0)
        with col3:
            processing_skip = st.selectbox("Process Every N Frames", [1, 2, 3], index=1)
        
        # Performance warning
        st.warning("⚠️ For best real-time performance, use a computer with dedicated GPU!")
        
        # Enhanced real-time callback
        def optimized_callback(frame: av.VideoFrame) -> av.VideoFrame:
            try:
                processor = st.session_state.video_processor
                processor.frame_count += 1
                
                # Convert frame
                img = frame.to_ndarray(format="rgb24")
                
                # Skip processing based on setting
                if processor.frame_count % processing_skip != 0:
                    return av.VideoFrame.from_ndarray(img, format="rgb24")
                
                # Process with model
                processed_input = processor.preprocess_frame(img)
                if processed_input is not None:
                    prediction = processor.predict_with_model(
                        interpreter, input_details, output_details, processed_input
                    )
                    
                    if prediction is not None:
                        mask = processor.postprocess_prediction(prediction, img.shape, threshold)
                        colored_result, _ = apply_nail_color_enhanced(img, mask, selected_color, opacity)
                        return av.VideoFrame.from_ndarray(colored_result, format="rgb24")
                
                return av.VideoFrame.from_ndarray(img, format="rgb24")
                
            except Exception as e:
                logger.error(f"Real-time processing error: {str(e)}")
                return av.VideoFrame.from_ndarray(img, format="rgb24")
        
        # WebRTC configuration for better quality
        rtc_config = RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        })
        
        # Start real-time processing
        ctx = webrtc_streamer(
            key="optimized-realtime",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config,
            video_frame_callback=optimized_callback,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640 if resolution == "480p" else 1280},
                    "height": {"ideal": 480 if resolution == "480p" else 720},
                    "frameRate": {"ideal": fps_target}
                },
                "audio": False
            },
            async_processing=True
        )
        
        if ctx.state.playing:
            st.success("🎥 Real-time processing active!")
        else:
            st.info("👆 Click 'START' to begin real-time processing")
    
    # Footer with tips
    st.markdown("---")
    st.subheader("💡 Troubleshooting Tips")
    
    tips_col1, tips_col2 = st.columns(2)
    with tips_col1:
        st.markdown("""
        **Model Issues:**
        - Ensure model file exists in correct location
        - Check model input/output shapes match expectations
        - Verify model is properly trained for nail segmentation
        - Try different threshold values
        """)
    
    with tips_col2:
        st.markdown("""
        **Performance Issues:**
        - Use GPU-enabled computer for real-time processing
        - Reduce processing frequency (process every 2-3 frames)
        - Lower resolution for better FPS
        - Ensure good lighting conditions
        """)

if __name__ == "__main__":
    main()