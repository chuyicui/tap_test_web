from flask import Flask, render_template, request, jsonify, send_file
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
import threading

app = Flask(__name__)

class UPDRSWebApp:
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
        self.finger_distance_threshold = 80  # pixels for tap detection (increased for easier detection)
        self.tap_cooldown = 0.1  # seconds between taps (reduced for more responsive detection)
        
        # Session data
        self.session_data = {
            'mood_rating': None,
            'right_hand_results': {'taps': 0, 'distances': [], 'timestamps': []},
            'left_hand_results': {'taps': 0, 'distances': [], 'timestamps': []},
            'test_start_time': None,
            'current_hand': None,
            'test_phase': 'questionnaire'  # questionnaire, right_hand, left_hand, results
        }
        
        # Camera setup
        self.cap = None
        self.setup_camera()
    
    def setup_camera(self):
        """Initialize camera"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Error: Could not open camera")
                self.cap = None
        except Exception as e:
            print(f"Camera setup error: {e}")
            self.cap = None
    
    def get_camera_frame(self):
        """Get current camera frame"""
        if self.cap is None:
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        return frame
    
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
            self.session_data['right_hand_results']['distances'].append(pixel_distance)
            self.session_data['right_hand_results']['timestamps'].append(current_time)
        else:
            self.session_data['left_hand_results']['distances'].append(pixel_distance)
            self.session_data['left_hand_results']['timestamps'].append(current_time)
        
        # Check if fingers are close enough for a tap
        if pixel_distance < self.finger_distance_threshold:
            # Check cooldown
            last_tap_time = getattr(self, f'last_{target_hand}_tap_time', 0)
            if current_time - last_tap_time > self.tap_cooldown:
                setattr(self, f'last_{target_hand}_tap_time', current_time)
                
                # Update tap count
                if target_hand == "right":
                    self.session_data['right_hand_results']['taps'] += 1
                else:
                    self.session_data['left_hand_results']['taps'] += 1
                
                print(f"Tap detected! {target_hand} hand, distance: {pixel_distance:.1f}px, taps: {self.session_data[f'{target_hand}_hand_results']['taps']}")
                return True, pixel_distance
        
        return False, pixel_distance
    
    def process_frame(self, target_hand):
        """Process camera frame for hand tracking"""
        frame = self.get_camera_frame()
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
        
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return frame_base64, finger_positions, results
    
    def generate_chart(self, distances, timestamps, title, color):
        """Generate a distance chart"""
        if not distances or len(distances) < 2:
            return None
        
        plt.figure(figsize=(10, 4))
        plt.plot(timestamps, distances, color=color, linewidth=2)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('Distance (pixels)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save to BytesIO
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    
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
        c.drawString(50, height - 110, f"Mood Rating: {self.session_data['mood_rating']}/5")
        
        # Results
        y_pos = height - 150
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_pos, "Test Results:")
        
        y_pos -= 30
        c.setFont("Helvetica", 12)
        right_taps = self.session_data['right_hand_results']['taps']
        left_taps = self.session_data['left_hand_results']['taps']
        right_rate = right_taps / self.test_duration
        left_rate = left_taps / self.test_duration
        
        c.drawString(50, y_pos, f"Right Hand: {right_taps} taps ({right_rate:.1f} taps/sec)")
        y_pos -= 20
        c.drawString(50, y_pos, f"Left Hand: {left_taps} taps ({left_rate:.1f} taps/sec)")
        
        # Charts
        y_pos -= 50
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y_pos, "Distance Charts:")
        
        # Right hand chart
        if self.session_data['right_hand_results']['distances']:
            right_chart = self.generate_chart(
                self.session_data['right_hand_results']['distances'],
                self.session_data['right_hand_results']['timestamps'],
                "Right Hand Distance Over Time",
                'blue'
            )
            if right_chart:
                img_data = base64.b64decode(right_chart)
                img_buffer = BytesIO(img_data)
                c.drawImage(ImageReader(img_buffer), 50, y_pos - 200, width=500, height=150)
        
        # Left hand chart
        if self.session_data['left_hand_results']['distances']:
            left_chart = self.generate_chart(
                self.session_data['left_hand_results']['distances'],
                self.session_data['left_hand_results']['timestamps'],
                "Left Hand Distance Over Time",
                'red'
            )
            if left_chart:
                img_data = base64.b64decode(left_chart)
                img_buffer = BytesIO(img_data)
                c.drawImage(ImageReader(img_buffer), 50, y_pos - 400, width=500, height=150)
        
        c.save()
        buffer.seek(0)
        return buffer

# Global app instance
updrs_app = UPDRSWebApp()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/questionnaire')
def questionnaire():
    return render_template('questionnaire.html')

@app.route('/test')
def test():
    return render_template('test.html')

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/api/submit_mood', methods=['POST'])
def submit_mood():
    data = request.json
    updrs_app.session_data['mood_rating'] = data.get('mood_rating')
    updrs_app.session_data['test_phase'] = 'right_hand'
    return jsonify({'success': True})

@app.route('/api/start_test', methods=['POST'])
def start_test():
    data = request.json
    hand = data.get('hand')
    updrs_app.session_data['current_hand'] = hand
    updrs_app.session_data['test_start_time'] = time.time()
    
    # Reset results for this hand
    if hand == 'right':
        updrs_app.session_data['right_hand_results'] = {'taps': 0, 'distances': [], 'timestamps': []}
    else:
        updrs_app.session_data['left_hand_results'] = {'taps': 0, 'distances': [], 'timestamps': []}
    
    return jsonify({'success': True})

@app.route('/api/camera_feed')
def camera_feed():
    if updrs_app.session_data['current_hand'] and updrs_app.session_data['test_start_time']:
        # Check if test is still active
        elapsed = time.time() - updrs_app.session_data['test_start_time']
        if elapsed < updrs_app.test_duration:
            frame, finger_positions, results = updrs_app.process_frame(updrs_app.session_data['current_hand'])
            return jsonify({
                'frame': frame,
                'finger_positions': finger_positions,
                'success': True
            })
        else:
            # Test is complete, return empty frame
            return jsonify({
                'frame': None,
                'finger_positions': None,
                'success': False,
                'test_complete': True
            })
    return jsonify({'success': False})

@app.route('/api/test_status')
def test_status():
    if updrs_app.session_data['test_start_time']:
        elapsed = time.time() - updrs_app.session_data['test_start_time']
        remaining = max(0, updrs_app.test_duration - elapsed)
        
        current_hand = updrs_app.session_data['current_hand']
        if current_hand == 'right':
            taps = updrs_app.session_data['right_hand_results']['taps']
        else:
            taps = updrs_app.session_data['left_hand_results']['taps']
        
        # Debug info
        print(f"Test status - Hand: {current_hand}, Elapsed: {elapsed:.1f}s, Remaining: {remaining:.1f}s, Taps: {taps}")
        
        return jsonify({
            'remaining_time': remaining,
            'taps': taps,
            'test_complete': remaining <= 0,
            'current_hand': current_hand,
            'elapsed_time': elapsed
        })
    return jsonify({'success': False})

@app.route('/api/complete_test', methods=['POST'])
def complete_test():
    current_hand = updrs_app.session_data['current_hand']
    
    # Stop the current test
    updrs_app.session_data['test_start_time'] = None
    updrs_app.session_data['current_hand'] = None
    
    # Move to next phase
    if current_hand == 'right':
        updrs_app.session_data['test_phase'] = 'left_hand'
    else:
        updrs_app.session_data['test_phase'] = 'results'
    
    return jsonify({'success': True, 'next_phase': updrs_app.session_data['test_phase']})

@app.route('/api/get_results')
def get_results():
    # Generate charts
    right_chart = None
    left_chart = None
    
    if updrs_app.session_data['right_hand_results']['distances']:
        right_chart = updrs_app.generate_chart(
            updrs_app.session_data['right_hand_results']['distances'],
            updrs_app.session_data['right_hand_results']['timestamps'],
            "Right Hand Distance Over Time",
            '#3B82F6'
        )
    
    if updrs_app.session_data['left_hand_results']['distances']:
        left_chart = updrs_app.generate_chart(
            updrs_app.session_data['left_hand_results']['distances'],
            updrs_app.session_data['left_hand_results']['timestamps'],
            "Left Hand Distance Over Time",
            '#EF4444'
        )
    
    return jsonify({
        'mood_rating': updrs_app.session_data['mood_rating'],
        'right_hand': updrs_app.session_data['right_hand_results'],
        'left_hand': updrs_app.session_data['left_hand_results'],
        'right_chart': right_chart,
        'left_chart': left_chart,
        'test_duration': updrs_app.test_duration
    })

@app.route('/api/download_report')
def download_report():
    report_buffer = updrs_app.generate_report()
    return send_file(
        report_buffer,
        as_attachment=True,
        download_name=f'updrs_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf',
        mimetype='application/pdf'
    )

@app.route('/api/reset')
def reset():
    updrs_app.session_data = {
        'mood_rating': None,
        'right_hand_results': {'taps': 0, 'distances': [], 'timestamps': []},
        'left_hand_results': {'taps': 0, 'distances': [], 'timestamps': []},
        'test_start_time': None,
        'current_hand': None,
        'test_phase': 'questionnaire'
    }
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
