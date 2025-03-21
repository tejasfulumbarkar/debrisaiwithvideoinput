import streamlit as st
import joblib
from ultralytics import YOLO
import numpy as np
import os 
import cv2
import time
from PIL import Image
import tempfile

# Load YOLO model
def load_css(file_name:str)->str:
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
yolo_model = YOLO('best3.pt')

# Load Random Forest Classifier
rf_classifier = joblib.load('random_forest_model.pkl')

# Apply custom CSS for better UI
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
        
        .stApp {
            margin: 0;
            padding: 0;
            background: #1E1332;  /* Dark purple background */
            color: #E2E8F0;
            font-family: 'Poppins', sans-serif;
            max-width: 100vw !important;
            overflow-x: hidden;
        }
        
        .title {
            font-size: 3.5rem;
            font-weight: 700;
            text-align: left;
            background: linear-gradient(120deg, #E2A3FF, #A682FF);  /* Purple gradient */
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 2rem 0 0.5rem 0;
            padding: 0;
            line-height: 1.2;
        }
        
        .tagline {
            font-size: 1.1rem;
            text-align: left;
            color: #B4A5FF;  /* Light purple */
            margin-bottom: 2rem;
            font-weight: 400;
            line-height: 1.6;
            max-width: 600px;
        }
        
        .upload-box {
            border-radius: 1rem;
            border: 2px dashed #A682FF;  /* Purple border */
            padding: 2rem;
            background: rgba(30, 19, 50, 0.5);  /* Dark purple with transparency */
            text-align: center;
            margin: 2rem 0;
            backdrop-filter: blur(12px);
        }
        
        .upload-box h3 {
            font-size: 1.3rem;
            margin-bottom: 1rem;
            color: #E2A3FF !important;  /* Light purple */
        }
        
        .result-box {
            background: rgba(30, 19, 50, 0.7);  /* Dark purple with transparency */
            padding: 1.5rem;
            border-radius: 1rem;
            margin: 1rem 0;
            border-left: 4px solid #A682FF;  /* Purple accent */
            font-size: 1rem;
            width: 100%;
            box-sizing: border-box;
        }
        
        /* Full-width image container */
        .image-container {
            width: 100%;
            margin: 0;
            padding: 0;
        }
        
        .image-container img {
            width: 100%;
            border-radius: 1rem;
            margin: 0;
        }
        
        /* Detection info layout */
        .detection-info {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-top: 2rem;
        }
        
        .debug-box {
            background: rgba(30, 19, 50, 0.5);
            padding: 1.5rem;
            border-radius: 1rem;
            margin: 0.5rem 0;
            border: 1px solid #A682FF;
            width: 100%;
            box-sizing: border-box;
        }
        
        /* Custom button styling */
        .stButton>button {
            background: linear-gradient(120deg, #E2A3FF, #A682FF) !important;
            color: white !important;
            border: none !important;
            padding: 0.8rem 2rem !important;
            border-radius: 0.5rem !important;
            font-weight: 500 !important;
            transition: all 0.3s ease !important;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 12px rgba(166, 130, 255, 0.3) !important;
        }
        
        /* Hide Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Responsive design */
        @media (max-width: 768px) {
            .title { font-size: 2.5rem; }
            .tagline { font-size: 1rem; }
            .detection-info {
                grid-template-columns: 1fr;
            }
        }
        
        /* Summary Card Styles */
        .summary-card {
            background: rgba(30, 19, 50, 0.7);
            border-radius: 1rem;
            padding: 2rem;
            margin: 2rem 0;
            border: 1px solid #A682FF;
            box-shadow: 0 4px 20px rgba(166, 130, 255, 0.2);
        }
        
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-top: 1rem;
        }
        
        .summary-item {
            text-align: center;
            padding: 1rem;
            background: rgba(166, 130, 255, 0.1);
            border-radius: 0.5rem;
            transition: transform 0.3s ease;
        }
        
        .summary-item:hover {
            transform: translateY(-5px);
        }
        
        .summary-icon {
            font-size: 2rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(120deg, #E2A3FF, #A682FF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        /* Confidence Meter Styles */
        .confidence-meter {
            width: 100%;
            height: 8px;
            background: rgba(166, 130, 255, 0.2);
            border-radius: 4px;
            margin: 0.5rem 0;
            position: relative;
            overflow: hidden;
        }
        
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #E2A3FF, #A682FF);
            border-radius: 4px;
            transition: width 0.5s ease;
        }
        
        /* Recommendation Box Styles */
        .recommendation-box {
            background: rgba(30, 19, 50, 0.7);
            border-radius: 1rem;
            padding: 1.5rem;
            margin: 1rem 0;
            border: 1px solid #A682FF;
            position: relative;
            overflow: hidden;
        }
        
        .recommendation-box::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 4px;
            background: linear-gradient(90deg, #E2A3FF, #A682FF);
        }
        
        .recommendation-title {
            color: #E2A3FF;
            font-size: 1.2rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .action-steps {
            background: rgba(166, 130, 255, 0.1);
            padding: 1rem;
            border-radius: 0.5rem;
            margin-top: 1rem;
        }
        
        .step-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin: 0.5rem 0;
            padding: 0.5rem;
            border-radius: 0.25rem;
            background: rgba(30, 19, 50, 0.5);
        }
        
        /* Risk Level Indicator */
        .risk-indicator {
            display: inline-flex;
            align-items: center;
            padding: 0.25rem 0.75rem;
            border-radius: 1rem;
            font-size: 0.9rem;
            font-weight: 500;
            margin: 0.5rem 0;
        }
        
        .risk-high {
            background: rgba(255, 86, 86, 0.2);
            color: #FF5656;
        }
        
        .risk-moderate {
            background: rgba(255, 170, 86, 0.2);
            color: #FFAA56;
        }
        
        .risk-low {
            background: rgba(86, 255, 136, 0.2);
            color: #56FF88;
        }
        
        .risk-unknown {
            background: rgba(156, 156, 156, 0.2);
            color: #9C9C9C;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title & Tagline
st.markdown("""
    <div style='text-align: center; margin: 2rem 0;'>
        <h1 style='font-size: 3.5rem; font-weight: 700; margin-bottom: 1rem; 
            background: linear-gradient(120deg, #E2A3FF, #A682FF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;'>
            🔍 Detecting and Classifying Space Debris with AI 🚀
        </h1>
        <p style='font-size: 1.2rem; color: #B4A5FF; margin-bottom: 2rem;'>
            🌍 Using AI to detect, classify, and track space debris for a cleaner orbit ✨
        </p>
    </div>
""", unsafe_allow_html=True)

# Create columns for better layout - adjusted ratios for iframe
col1, col2, col3 = st.columns([0.5, 3, 0.5])

with col2:
    # Upload Box with improved styling and emojis
    st.markdown(
        """
        <div class='upload-box' style='text-align: center; padding: 2rem;'>
            <h3 style='color: #E2A3FF; margin-bottom: 1rem; font-size: 1.5rem;'>
                📡 Upload Satellite Image/Video 🛸
            </h3>
            <p style='color: #B4A5FF; margin: 1rem 0;'>
                🎯 Drop your file here or click to browse 📂
            </p>
            <p style='color: #B4A5FF; font-size: 0.9rem;'>
                📁 Supported formats: JPG, JPEG, PNG, MP4, AVI 📁
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "mp4", "avi"])

def calculate_debris_size(width, height):
    area = width * height
    if area < 1000:
        return "Small"
    elif area < 5000:
        return "Medium"
    else:
        return "Large"

def determine_removal_method(size, confidence):
    """
    Determine the most appropriate removal method based on debris characteristics
    """
    if confidence > 0.5:  # High confidence detections
        if size == "Small":
            return "Laser Ablation"
        elif size == "Medium":
            return "Robotic Capture"
        else:  # Large
            return "Tether System"
    elif confidence > 0.3:  # Medium confidence
        if size == "Small":
            return "Ground Tracking"
        elif size == "Medium":
            return "Laser Tracking"
        else:
            return "Further Analysis"
    else:  # Low confidence
        return "Monitor & Track"

def determine_risk_level(size, confidence):
    """
    Determine risk level based on debris characteristics
    """
    if confidence > 0.5:
        if size == "Large":
            return "High"
        elif size == "Medium":
            return "Moderate"
        else:
            return "Low"
    elif confidence > 0.3:
        if size == "Large":
            return "High"
        else:
            return "Moderate"
    else:
        return "Unknown"

def process_image(image_path):
    # Read image
    image = cv2.imread(image_path)
    original_image = image.copy()
    
    # Process with YOLO
    results = yolo_model(image)
    
    detections = []
    all_detections = []  # Store all detections regardless of confidence
    
    # Class names mapping
    class_names = {
        0: 'Debris',
        1: 'Debris',
        9: 'Person'
    }
    
    # Draw detections on image
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get box coordinates and confidence
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            # Get class name
            class_name = class_names.get(cls, f'Class {cls}')
            
            # Store all detections for debugging
            width = int(x2 - x1)
            height = int(y2 - y1)
            size = calculate_debris_size(width, height)
            
            all_detections.append({
                'confidence': conf,
                'class': cls,
                'class_name': class_name,
                'size': size,
                'position': (int(x1), int(y1), int(x2), int(y2))
            })
            
            # Process debris detections (both class 0 and 1)
            if cls in [0, 1]:  # Check for both debris classes
                # Color coding based on confidence:
                # Green: conf > 0.5
                # Yellow: 0.3 < conf <= 0.5
                # Orange: 0.2 < conf <= 0.3
                if conf > 0.5:
                    color = (0, 255, 0)  # Green for high confidence
                elif conf > 0.3:
                    color = (0, 255, 255)  # Yellow for medium confidence
                else:
                    color = (0, 165, 255)  # Orange for low confidence
                
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"Debris {conf:.2f} ({size})"
                cv2.putText(image, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Store all debris detections with confidence > 0.2
                if conf > 0.2:  # Lowered threshold to catch fainter debris
                    detections.append({
                        'confidence': conf,
                        'size': size,
                        'position': (int(x1), int(y1), int(x2), int(y2))
                    })
    
    # Save both original and processed images
    cv2.imwrite("original_image.jpg", original_image)
    cv2.imwrite("processed_image.jpg", image)
    
    return detections, all_detections

def process_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create temporary directory if it doesn't exist
    if not os.path.exists('temp_videos'):
        os.makedirs('temp_videos')
    
    # Create output paths
    output_path = os.path.join('temp_videos', 'output.mp4')
    
    # Initialize progress bar and status text
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Store frames in memory
    frames = []
    frame_count = 0
    total_detections = 0  # Rename for clarity
    detection_details = []
    
    # Define colors for different confidence levels
    COLOR_HIGH = (0, 255, 0)    # Green
    COLOR_MEDIUM = (0, 255, 255) # Yellow
    COLOR_LOW = (0, 165, 255)    # Orange
    COLOR_VERY_LOW = (128, 0, 255) # Purple for very low confidence
    
    # Set YOLO model parameters for higher sensitivity
    yolo_model.conf = 0.05  # Even lower confidence threshold
    yolo_model.iou = 0.2   # Lower IOU threshold for better detection of close objects
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Add legend to the frame
        legend_height = 100
        legend = np.zeros((legend_height, frame.shape[1], 3), dtype=np.uint8)
        cv2.putText(legend, "Debris Detection Legend:", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(legend, "High Confidence (>0.5)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_HIGH, 2)
        cv2.putText(legend, "Medium Confidence (0.3-0.5)", (250, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_MEDIUM, 2)
        cv2.putText(legend, "Low Confidence (0.2-0.3)", (500, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_LOW, 2)
        cv2.putText(legend, "Very Low Confidence (0.1-0.2)", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_VERY_LOW, 2)
        
        # Add black background for better text visibility
        frame_with_legend = np.vstack([legend, frame])
            
        # Process frame with YOLO with higher sensitivity
        results = yolo_model(frame, conf=0.05, iou=0.2)
        
        # Track detections in current frame
        frame_detections = 0
        
        # Draw detections on frame
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates and confidence
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # Process debris detections (both class 0 and 1)
                if cls in [0, 1]:  # Check for both debris classes
                    frame_detections += 1
                    total_detections += 1
                    
                    # Color coding based on confidence
                    if conf > 0.5:
                        color = COLOR_HIGH
                        conf_text = "High"
                    elif conf > 0.3:
                        color = COLOR_MEDIUM
                        conf_text = "Medium"
                    elif conf > 0.2:
                        color = COLOR_LOW
                        conf_text = "Low"
                    else:
                        color = COLOR_VERY_LOW
                        conf_text = "Very Low"
                    
                    # Calculate size
                    width_box = int(x2 - x1)
                    height_box = int(y2 - y1)
                    size = calculate_debris_size(width_box, height_box)
                    
                    # Draw thicker rectangle
                    cv2.rectangle(frame_with_legend, 
                                (int(x1), int(y1) + legend_height), 
                                (int(x2), int(y2) + legend_height), 
                                color, 3)
                    
                    # Add black background for text
                    label = f"Debris ({conf_text}, {size})"
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame_with_legend, 
                                (int(x1), int(y1) - 25 + legend_height),
                                (int(x1) + label_w, int(y1) + legend_height),
                                (0, 0, 0), -1)
                    
                    # Draw text with better visibility
                    cv2.putText(frame_with_legend, label,
                              (int(x1), int(y1) - 5 + legend_height),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Store detection details
                    detection_details.append({
                        'frame': frame_count,
                        'confidence': conf,
                        'confidence_level': conf_text,
                        'size': size,
                        'position': (int(x1), int(y1), int(x2), int(y2))
                    })
        
        # Store the processed frame
        frames.append(frame_with_legend)
        
        # Update progress
        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        status_text.markdown(f"<p style='color: white; margin: 0; padding: 0;'>Processing frame {frame_count}/{total_frames}</p>", unsafe_allow_html=True)
    
    # Calculate estimated unique debris
    unique_debris = len(set(det['frame'] for det in detection_details))  # Count frames with detections
    
    # Release the capture
    cap.release()
    
    # Save frames as video using cv2.VideoWriter
    if frames:
        # Try different codecs
        codecs = [
            ('avc1', '.mp4'),
            ('H264', '.mp4'),
            ('XVID', '.avi'),
            ('MJPG', '.avi'),
            ('mp4v', '.mp4')
        ]
        
        for codec, ext in codecs:
            try:
                temp_output = os.path.join('temp_videos', f'output{ext}')
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(temp_output, fourcc, fps, 
                                    (frames[0].shape[1], frames[0].shape[0]))
                
                for frame in frames:
                    out.write(frame)
                
                out.release()
                
                # Check if the file was created and is not empty
                if os.path.exists(temp_output) and os.path.getsize(temp_output) > 0:
                    output_path = temp_output
                    break
            except Exception as e:
                continue
    
    # Return video properties along with output path and debris count
    return {
        'output_path': output_path,
        'debris_count': unique_debris,
        'total_detections': total_detections,
        'total_frames': total_frames,
        'fps': fps,
        'width': width,
        'height': height,
        'detection_details': detection_details
    }

if uploaded_file is not None:
    # Create a temporary file to store the uploaded file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.getbuffer())
    temp_file.close()
    
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension in ['mp4', 'avi']:
        # Process video
        st.markdown("<p style='color: white;'>Processing video... This may take a few minutes depending on the video length.</p>", unsafe_allow_html=True)
        
        try:
            # Process the video
            video_results = process_video(temp_file.name)
            
            # Display results
            if video_results['total_detections'] > 0:  # Changed from debris_count to total_detections
                st.success(f"Video processing complete! Detected approximately {video_results['debris_count']} unique debris objects across all frames.")
                st.info(f"Total detection events: {video_results['total_detections']} (includes multiple detections of the same debris)")
            else:
                st.warning("No debris detected in the video.")
            
            # Try different methods to display the video
            try:
                # Method 1: Direct file path
                st.video(video_results['output_path'])
            except Exception as e1:
                try:
                    # Method 2: Read as bytes
                    with open(video_results['output_path'], 'rb') as video_file:
                        video_bytes = video_file.read()
                        st.video(video_bytes)
                except Exception as e2:
                    try:
                        # Method 3: Read as bytes with explicit format
                        with open(video_results['output_path'], 'rb') as video_file:
                            video_bytes = video_file.read()
                            st.video(video_bytes, format='video/mp4')
                    except Exception as e3:
                        st.error("Unable to display the video. Please try a different video file.")
                        st.error(f"Error details: {str(e3)}")
            
            # Display statistics
            st.markdown(
                f"""
                <div class='result-box'>
                    <h3 style='color: #E2A3FF; margin-bottom: 1rem;'>Analysis Results</h3>
                    <p style='margin-bottom: 0.5rem;'>🎥 <strong>Video Length:</strong> {int(video_results['total_frames'] / video_results['fps'])} seconds</p>
                    <p style='margin-bottom: 0.5rem;'>📊 <strong>Total Frames:</strong> {video_results['total_frames']}</p>
                    <p style='margin-bottom: 0.5rem;'>🔍 <strong>Total Debris Detections:</strong> {video_results['total_detections']}</p>
                    <p style='margin-bottom: 0.5rem;'>🎯 <strong>Frames with Debris:</strong> {video_results['debris_count']}</p>
                    <p style='margin-bottom: 0.5rem;'>📈 <strong>Average Detections per Frame:</strong> {video_results['total_detections']/video_results['total_frames']:.2f}</p>
                    <p style='margin-bottom: 0.5rem;'>📐 <strong>Resolution:</strong> {video_results['width']}x{video_results['height']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Display detection details if any were found
            if video_results['detection_details']:
                st.markdown("### Detection Details")
                for det in video_results['detection_details']:
                    st.markdown(f"- Frame {det['frame']}: Confidence {det['confidence']:.2f}")
                    
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            st.info("Please make sure the video format is supported and try again.")
    else:
        # Process image
        # Save uploaded file
        with open("uploaded_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Process the image and get detections
        detections, all_detections = process_image("uploaded_image.jpg")

        # Display full-width image
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image("processed_image.jpg", caption="Analyzed Image with Detections", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add a processing animation
        with st.spinner("🔍 Analyzing debris characteristics..."):
            time.sleep(1)
        
        # Display detection results in a grid layout
        st.markdown('<div class="detection-info">', unsafe_allow_html=True)
        
        # Display debris detections first
        if detections:
            # Display summary in columns
            st.markdown("<h2 style='color: #E2A3FF; margin: 2rem 0;'>✨ Detection Summary ✨</h2>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(
                    """
                    <div class='summary-item'>
                        <div style='font-size: 2.5rem; margin-bottom: 0.5rem;'>🛰️</div>
                        <h3>Total Detections</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.write(f"### {len(detections)}")
                
            with col2:
                st.markdown(
                    """
                    <div class='summary-item'>
                        <div style='font-size: 2.5rem; margin-bottom: 0.5rem;'>📏</div>
                        <h3>Size Range</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.write(f"### {min(d['size'] for d in detections)} - {max(d['size'] for d in detections)}")
                
            with col3:
                st.markdown(
                    """
                    <div class='summary-item'>
                        <div style='font-size: 2.5rem; margin-bottom: 0.5rem;'>🎯</div>
                        <h3>Avg. Confidence</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.write(f"### {sum(d['confidence'] for d in detections)/len(detections):.2f}")

            # Display each detection in a cleaner format
            for i, det in enumerate(detections, 1):
                st.markdown("---")
                st.markdown(f"## 🛸 Debris Detection #{i}")
                
                # Create columns for detection details
                info_col1, info_col2 = st.columns(2)
                
                with info_col1:
                    st.markdown("### ⚠️ Risk Assessment")
                    risk_level = determine_risk_level(det['size'], det['confidence'])
                    risk_emoji = {
                        "High": "🔴",
                        "Moderate": "🟡",
                        "Low": "🟢",
                        "Unknown": "⚪"
                    }
                    st.markdown(f"**Level:** {risk_emoji.get(risk_level, '⚪')} {risk_level}")
                    st.progress(det['confidence'])
                    st.markdown(f"🎯 Confidence Score: {det['confidence']:.3f}")
                    
                with info_col2:
                    st.markdown("### 📊 Debris Characteristics")
                    size_emoji = {
                        "Small": "🔹",
                        "Medium": "🔶",
                        "Large": "💠"
                    }
                    st.markdown(f"**Size:** {size_emoji.get(det['size'], '🔹')} {det['size']}")
                    st.markdown(f"**Position:** 📍 {det['position']}")
                
                # Recommendation section
                st.markdown("### 🎯 Recommended Action")
                action = determine_removal_method(det['size'], det['confidence'])
                action_emoji = {
                    "Laser Ablation": "🔆",
                    "Robotic Capture": "🤖",
                    "Tether System": "🔗",
                    "Ground Tracking": "📡",
                    "Laser Tracking": "🎯",
                    "Further Analysis": "🔍",
                    "Monitor & Track": "👁️"
                }
                st.info(f"**{action_emoji.get(action, '🎯')} {action}**")
                
                # Action steps in an expander
                with st.expander("📋 View Action Steps"):
                    st.markdown("1. 📡 Monitor debris trajectory and velocity")
                    st.markdown("2. 🎯 Calculate optimal interception point")
                    st.markdown(f"3. ⚡ Deploy {action.lower()} system")
        else:
            st.info("🔍 No space debris detected in the image.")
        
        st.markdown('</div>', unsafe_allow_html=True)