import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from io import BytesIO
import base64
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="UPDRS Finger Tapping Test",
    page_icon="🤲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .test-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .camera-container {
        border: 3px solid #667eea;
        border-radius: 15px;
        padding: 1rem;
        background: #f8f9fa;
    }
    
    .progress-container {
        background: #e9ecef;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .emoji-button {
        font-size: 3rem;
        padding: 1rem;
        border: none;
        border-radius: 15px;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .emoji-button:hover {
        transform: scale(1.1);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .emoji-button.selected {
        background: #667eea;
        color: white;
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

class UPDRSStreamlitApp:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Test configuration
        self.test_duration = 10  # seconds
        self.finger_distance_threshold = 80  # pixels for tap detection
        self.tap_cooldown = 0.1  # seconds between taps
        
        # Initialize session state
        if 'session_data' not in st.session_state:
            st.session_state.session_data = {
                'mood_rating': None,
                'right_hand_results': {'taps': 0, 'distances': [], 'timestamps': []},
                'left_hand_results': {'taps': 0, 'distances': [], 'timestamps': []},
                'test_start_time': None,
                'current_hand': None,
                'test_phase': 'questionnaire'  # questionnaire, right_hand, left_hand, results
            }
        
        if 'last_tap_times' not in st.session_state:
            st.session_state.last_tap_times = {'right': 0, 'left': 0}
    
    def find_target_hand(self, results, target_hand):
        """Find the target hand (right or left) from detected hands"""
        if not results.multi_hand_landmarks:
            return None, None
            
        hand_landmarks = results.multi_hand_landmarks
        handedness = results.multi_handedness
        
        if len(hand_landmarks) == 1:
            return hand_landmarks[0], handedness[0]
        
        # Find the target hand based on current test phase
        for landmarks, hand_info in zip(hand_landmarks, handedness):
            if target_hand == "right" and hand_info.classification[0].label == "Right":
                return landmarks, hand_info
            elif target_hand == "left" and hand_info.classification[0].label == "Left":
                return landmarks, hand_info
        
        # If target hand not found, return the first hand
        return hand_landmarks[0], handedness[0]
    
    def get_finger_tips(self, landmarks):
        """Extract thumb and index finger tip positions"""
        if landmarks is None:
            return None
        
        # MediaPipe hand landmarks: Thumb tip: 4, Index finger tip: 8
        thumb_tip = landmarks.landmark[4]
        index_tip = landmarks.landmark[8]
        
        return {
            'thumb': (thumb_tip.x, thumb_tip.y),
            'index': (index_tip.x, index_tip.y)
        }
    
    def detect_finger_tap(self, finger_positions, target_hand):
        """Detect if thumb and index finger are close enough to count as a tap"""
        if finger_positions is None:
            return False, 0
        
        thumb_pos = finger_positions['thumb']
        index_pos = finger_positions['index']
        
        # Calculate distance between thumb and index finger (normalized coordinates)
        distance = np.sqrt((thumb_pos[0] - index_pos[0])**2 + (thumb_pos[1] - index_pos[1])**2)
        
        # Convert to pixel distance (assuming 640x480 camera)
        pixel_distance = distance * 640  # Scale to pixel units
        
        # Record distance and timestamp
        current_time = time.time()
        if target_hand == "right":
            st.session_state.session_data['right_hand_results']['distances'].append(pixel_distance)
            st.session_state.session_data['right_hand_results']['timestamps'].append(current_time)
        else:
            st.session_state.session_data['left_hand_results']['distances'].append(pixel_distance)
            st.session_state.session_data['left_hand_results']['timestamps'].append(current_time)
        
        # Check if fingers are close enough for a tap
        if pixel_distance < self.finger_distance_threshold:
            # Check cooldown
            last_tap_time = st.session_state.last_tap_times[target_hand]
            if current_time - last_tap_time > self.tap_cooldown:
                st.session_state.last_tap_times[target_hand] = current_time
                
                # Update tap count
                if target_hand == "right":
                    st.session_state.session_data['right_hand_results']['taps'] += 1
                else:
                    st.session_state.session_data['left_hand_results']['taps'] += 1
                
                return True, pixel_distance
        
        return False, pixel_distance
    
    def process_frame(self, frame, target_hand):
        """Process camera frame for hand tracking"""
        if frame is None:
            return None, None, None
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.hands.process(rgb_frame)
        
        # Find target hand and get finger positions
        hand_landmarks, handedness = self.find_target_hand(results, target_hand)
        finger_positions = None
        
        if hand_landmarks is not None:
            finger_positions = self.get_finger_tips(hand_landmarks)
            
            # Detect finger taps
            tap_detected, distance = self.detect_finger_tap(finger_positions, target_hand)
        
        # Draw hand landmarks and finger markers
        if results.multi_hand_landmarks:
            for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Only draw the target hand
                if ((target_hand == "right" and hand_info.classification[0].label == "Right") or
                    (target_hand == "left" and hand_info.classification[0].label == "Left")):
                    
                    # Draw hand landmarks
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Draw finger tip markers
                    if finger_positions:
                        for finger, pos in finger_positions.items():
                            x = int(pos[0] * frame.shape[1])
                            y = int(pos[1] * frame.shape[0])
                            
                            color = (0, 255, 255) if finger == 'thumb' else (255, 0, 255)  # Yellow for thumb, Magenta for index
                            cv2.circle(frame, (x, y), 8, color, -1)
                            cv2.circle(frame, (x, y), 10, (255, 255, 255), 2)
                        
                        # Draw connection line
                        thumb_pos = finger_positions['thumb']
                        index_pos = finger_positions['index']
                        thumb_x = int(thumb_pos[0] * frame.shape[1])
                        thumb_y = int(thumb_pos[1] * frame.shape[0])
                        index_x = int(index_pos[0] * frame.shape[1])
                        index_y = int(index_pos[1] * frame.shape[0])
                        
                        # Calculate pixel distance
                        pixel_distance = np.sqrt((thumb_pos[0] - index_pos[0])**2 + (thumb_pos[1] - index_pos[1])**2) * 640
                        
                        # Draw line with color based on distance
                        if pixel_distance < self.finger_distance_threshold:
                            line_color = (0, 255, 0)  # Green when close enough for tap
                            line_thickness = 5
                        else:
                            line_color = (0, 0, 255)  # Red when too far
                            line_thickness = 3
                        
                        cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), line_color, line_thickness)
                        
                        # Draw distance text
                        cv2.putText(frame, f"Distance: {pixel_distance:.0f}px", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(frame, f"Threshold: {self.finger_distance_threshold}px", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame, finger_positions, results
    
    def generate_chart(self, distances, timestamps, title, color):
        """Generate a distance chart using Plotly"""
        if not distances or len(distances) < 2:
            return None
        
        # Convert timestamps to relative time
        if timestamps:
            start_time = timestamps[0]
            relative_times = [(t - start_time) for t in timestamps]
        else:
            relative_times = list(range(len(distances)))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=relative_times,
            y=distances,
            mode='lines',
            name=title,
            line=dict(color=color, width=2)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Time (seconds)',
            yaxis_title='Distance (pixels)',
            height=300,
            showlegend=False,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return fig
    
    def generate_report(self):
        """Generate PDF report"""
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        
        # Title
        c.setFont("Helvetica-Bold", 20)
        c.drawString(50, height - 50, "UPDRS Finger Tapping Test Report")
        
        # Date and time
        c.setFont("Helvetica", 12)
        c.drawString(50, height - 80, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Mood rating
        c.drawString(50, height - 110, f"Mood Rating: {st.session_state.session_data['mood_rating']}/5")
        
        # Results
        y_pos = height - 150
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_pos, "Test Results:")
        
        y_pos -= 30
        c.setFont("Helvetica", 12)
        right_taps = st.session_state.session_data['right_hand_results']['taps']
        left_taps = st.session_state.session_data['left_hand_results']['taps']
        right_rate = right_taps / self.test_duration
        left_rate = left_taps / self.test_duration
        
        c.drawString(50, y_pos, f"Right Hand: {right_taps} taps ({right_rate:.1f} taps/sec)")
        y_pos -= 20
        c.drawString(50, y_pos, f"Left Hand: {left_taps} taps ({left_rate:.1f} taps/sec)")
        
        c.save()
        buffer.seek(0)
        return buffer

# Initialize the app
@st.cache_resource
def get_updrs_app():
    return UPDRSStreamlitApp()

updrs_app = get_updrs_app()

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤲 UPDRS Finger Tapping Test</h1>
        <p>Professional Assessment Tool for Motor Function Evaluation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation
    phase = st.session_state.session_data['test_phase']
    
    if phase == 'questionnaire':
        show_questionnaire()
    elif phase == 'right_hand':
        show_hand_test('right')
    elif phase == 'left_hand':
        show_hand_test('left')
    elif phase == 'results':
        show_results()

def show_questionnaire():
    st.markdown("""
    <div class="test-card">
        <h2>📋 How are you feeling today?</h2>
        <p>Please select the emoji that best represents your current mood and energy level</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("😴\n\nVery Tired", key="mood_1", help="Rating: 1"):
            st.session_state.session_data['mood_rating'] = 1
            st.rerun()
    
    with col2:
        if st.button("😔\n\nLow Energy", key="mood_2", help="Rating: 2"):
            st.session_state.session_data['mood_rating'] = 2
            st.rerun()
    
    with col3:
        if st.button("😐\n\nNeutral", key="mood_3", help="Rating: 3"):
            st.session_state.session_data['mood_rating'] = 3
            st.rerun()
    
    with col4:
        if st.button("😊\n\nGood", key="mood_4", help="Rating: 4"):
            st.session_state.session_data['mood_rating'] = 4
            st.rerun()
    
    with col5:
        if st.button("🤩\n\nExcellent", key="mood_5", help="Rating: 5"):
            st.session_state.session_data['mood_rating'] = 5
            st.rerun()
    
    # Show selected mood
    if st.session_state.session_data['mood_rating']:
        st.success(f"Selected mood rating: {st.session_state.session_data['mood_rating']}/5")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("← Back to Home", type="secondary"):
                st.session_state.session_data['test_phase'] = 'questionnaire'
                st.rerun()
        
        with col2:
            if st.button("Continue to Test →", type="primary"):
                st.session_state.session_data['test_phase'] = 'right_hand'
                st.rerun()
    
    # Information section
    with st.expander("ℹ️ Why do we ask about your mood?"):
        st.write("""
        Your emotional state can influence motor performance. By tracking your mood alongside 
        your test results, we can provide more comprehensive insights into your motor function 
        and identify patterns that may be related to your overall well-being.
        """)

def show_hand_test(hand):
    hand_name = "Right" if hand == "right" else "Left"
    
    st.markdown(f"""
    <div class="test-card">
        <h2>🤲 {hand_name} Hand Test</h2>
        <p>Tap your thumb and index finger together repeatedly for 10 seconds</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Camera feed
    st.markdown("### 📹 Camera Feed")
    camera_placeholder = st.empty()
    
    # Test controls
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button(f"Start {hand_name} Hand Test", type="primary", key=f"start_{hand}"):
            st.session_state.session_data['current_hand'] = hand
            st.session_state.session_data['test_start_time'] = time.time()
            
            # Reset results for this hand
            if hand == 'right':
                st.session_state.session_data['right_hand_results'] = {'taps': 0, 'distances': [], 'timestamps': []}
            else:
                st.session_state.session_data['left_hand_results'] = {'taps': 0, 'distances': [], 'timestamps': []}
            
            st.rerun()
    
    with col2:
        if st.button("← Back", type="secondary"):
            if hand == 'right':
                st.session_state.session_data['test_phase'] = 'questionnaire'
            else:
                st.session_state.session_data['test_phase'] = 'right_hand'
            st.rerun()
    
    # Test status
    if st.session_state.session_data['test_start_time']:
        elapsed = time.time() - st.session_state.session_data['test_start_time']
        remaining = max(0, updrs_app.test_duration - elapsed)
        
        # Progress bar
        progress = (updrs_app.test_duration - remaining) / updrs_app.test_duration
        st.progress(progress)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Time Remaining", f"{remaining:.1f}s")
        
        with col2:
            if hand == 'right':
                taps = st.session_state.session_data['right_hand_results']['taps']
            else:
                taps = st.session_state.session_data['left_hand_results']['taps']
            st.metric("Taps", taps)
        
        with col3:
            if remaining > 0:
                st.metric("Status", "Testing")
            else:
                st.metric("Status", "Complete")
        
        # Camera feed simulation (in real deployment, you'd use actual camera)
        if remaining > 0:
            # Simulate camera frame processing
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            processed_frame, finger_positions, results = updrs_app.process_frame(frame, hand)
            
            if processed_frame is not None:
                # Convert BGR to RGB for display
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                camera_placeholder.image(rgb_frame, caption=f"{hand_name} Hand Tracking", use_column_width=True)
        
        # Check if test is complete
        if remaining <= 0:
            st.success(f"{hand_name} hand test completed!")
            
            if hand == 'right':
                if st.button("Continue to Left Hand →", type="primary"):
                    st.session_state.session_data['test_phase'] = 'left_hand'
                    st.rerun()
            else:
                if st.button("View Results →", type="primary"):
                    st.session_state.session_data['test_phase'] = 'results'
                    st.rerun()
    
    # Instructions
    with st.expander("📋 Instructions"):
        st.write("""
        1. Position your hand clearly in front of the camera
        2. Tap your thumb and index finger together
        3. Keep tapping for the full 10 seconds
        4. Try to maintain a steady rhythm
        5. Make sure your hand is well-lit and visible
        """)

def show_results():
    st.markdown("""
    <div class="test-card">
        <h2>📊 Test Results</h2>
        <p>Your UPDRS Finger Tapping Test Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mood Rating", f"{st.session_state.session_data['mood_rating']}/5")
    
    with col2:
        right_taps = st.session_state.session_data['right_hand_results']['taps']
        right_rate = right_taps / updrs_app.test_duration
        st.metric("Right Hand", f"{right_taps} taps", f"{right_rate:.1f}/sec")
    
    with col3:
        left_taps = st.session_state.session_data['left_hand_results']['taps']
        left_rate = left_taps / updrs_app.test_duration
        st.metric("Left Hand", f"{left_taps} taps", f"{left_rate:.1f}/sec")
    
    with col4:
        total_taps = right_taps + left_taps
        avg_rate = total_taps / (2 * updrs_app.test_duration)
        st.metric("Average Rate", f"{avg_rate:.1f}/sec")
    
    # Charts
    st.markdown("### 📈 Distance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.session_data['right_hand_results']['distances']:
            right_chart = updrs_app.generate_chart(
                st.session_state.session_data['right_hand_results']['distances'],
                st.session_state.session_data['right_hand_results']['timestamps'],
                "Right Hand Distance Over Time",
                '#3B82F6'
            )
            if right_chart:
                st.plotly_chart(right_chart, use_container_width=True)
    
    with col2:
        if st.session_state.session_data['left_hand_results']['distances']:
            left_chart = updrs_app.generate_chart(
                st.session_state.session_data['left_hand_results']['distances'],
                st.session_state.session_data['left_hand_results']['timestamps'],
                "Left Hand Distance Over Time",
                '#EF4444'
            )
            if left_chart:
                st.plotly_chart(left_chart, use_container_width=True)
    
    # Detailed results
    with st.expander("📋 Detailed Results"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Right Hand")
            right_data = st.session_state.session_data['right_hand_results']
            st.write(f"**Total Taps:** {right_data['taps']}")
            st.write(f"**Taps per Second:** {right_data['taps'] / updrs_app.test_duration:.2f}")
            if right_data['distances']:
                st.write(f"**Average Distance:** {np.mean(right_data['distances']):.1f}px")
                st.write(f"**Min Distance:** {np.min(right_data['distances']):.1f}px")
                st.write(f"**Max Distance:** {np.max(right_data['distances']):.1f}px")
        
        with col2:
            st.subheader("Left Hand")
            left_data = st.session_state.session_data['left_hand_results']
            st.write(f"**Total Taps:** {left_data['taps']}")
            st.write(f"**Taps per Second:** {left_data['taps'] / updrs_app.test_duration:.2f}")
            if left_data['distances']:
                st.write(f"**Average Distance:** {np.mean(left_data['distances']):.1f}px")
                st.write(f"**Min Distance:** {np.min(left_data['distances']):.1f}px")
                st.write(f"**Max Distance:** {np.max(left_data['distances']):.1f}px")
    
    # Actions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("← Back to Test", type="secondary"):
            st.session_state.session_data['test_phase'] = 'left_hand'
            st.rerun()
    
    with col2:
        # Generate and download PDF report
        report_buffer = updrs_app.generate_report()
        st.download_button(
            label="📄 Download PDF Report",
            data=report_buffer.getvalue(),
            file_name=f'updrs_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf',
            mime='application/pdf',
            type="primary"
        )
    
    with col3:
        if st.button("🔄 Start New Test", type="secondary"):
            # Reset session data
            st.session_state.session_data = {
                'mood_rating': None,
                'right_hand_results': {'taps': 0, 'distances': [], 'timestamps': []},
                'left_hand_results': {'taps': 0, 'distances': [], 'timestamps': []},
                'test_start_time': None,
                'current_hand': None,
                'test_phase': 'questionnaire'
            }
            st.session_state.last_tap_times = {'right': 0, 'left': 0}
            st.rerun()

if __name__ == "__main__":
    main()
