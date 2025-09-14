import cv2
import mediapipe as mp
import pygame
import numpy as np
import time
import sys

class UPDRSGame:
    def __init__(self):
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Initialize Pygame
        pygame.init()
        self.screen_width = 1000
        self.screen_height = 700
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("UPDRS Finger Tapping Test")
        
        # Video display settings
        self.show_video = True
        self.video_width = 500
        self.video_height = 400
        self.video_x = self.screen_width // 2 - self.video_width // 2
        self.video_y = 150
        
        # Game state
        self.game_started = False
        self.test_duration = 10  # seconds
        self.start_time = None
        self.tap_count = 0
        self.last_tap_time = 0
        self.tap_cooldown = 0.2  # seconds between taps
        self.finger_distance_threshold = 50  # pixels for tap detection
        
        # Hand testing
        self.current_hand = "right"  # "right" or "left"
        self.right_hand_results = {"taps": 0, "distances": []}
        self.left_hand_results = {"taps": 0, "distances": []}
        self.test_phase = "instructions"  # "instructions", "right_hand", "left_hand", "results"
        
        # Modern sophisticated color palette
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        
        # Soft, muted colors
        self.SOFT_PINK = (255, 240, 245)
        self.LAVENDER = (230, 230, 250)
        self.MAUVE = (224, 176, 255)
        self.DUSTY_PINK = (220, 208, 255)
        self.PALE_PURPLE = (240, 230, 250)
        
        # Neutral grays
        self.CHARCOAL = (45, 45, 45)
        self.SLATE_GRAY = (112, 128, 144)
        self.LIGHT_SLATE = (200, 200, 210)
        self.SOFT_GRAY = (180, 180, 190)
        
        # Accent colors (muted)
        self.MUTED_GREEN = (120, 180, 120)
        self.MUTED_BLUE = (120, 150, 200)
        self.MUTED_ORANGE = (200, 150, 100)
        
        # Modern fonts
        try:
            # Try to use system fonts for better appearance
            self.font = pygame.font.SysFont("SF Pro Display", 32, bold=True)
            self.small_font = pygame.font.SysFont("SF Pro Text", 20)
            self.large_font = pygame.font.SysFont("SF Pro Display", 48, bold=True)
        except:
            # Fallback to default fonts
            self.font = pygame.font.Font(None, 36)
            self.small_font = pygame.font.Font(None, 24)
            self.large_font = pygame.font.Font(None, 48)
        
    def find_target_hand(self, results):
        """Find the target hand (right or left) from detected hands"""
        if not results.multi_hand_landmarks:
            return None, None
            
        # Get hand landmarks and handedness
        hand_landmarks = results.multi_hand_landmarks
        handedness = results.multi_handedness
        
        if len(hand_landmarks) == 1:
            # Only one hand detected, use it
            return hand_landmarks[0], handedness[0]
        
        # Find the target hand based on current test phase
        for i, (landmarks, hand_info) in enumerate(zip(hand_landmarks, handedness)):
            # Check if this is the hand we want to test
            if self.current_hand == "right" and hand_info.classification[0].label == "Right":
                return landmarks, hand_info
            elif self.current_hand == "left" and hand_info.classification[0].label == "Left":
                return landmarks, hand_info
        
        # If target hand not found, return the first hand
        return hand_landmarks[0], handedness[0]
    
    def get_finger_tips(self, landmarks):
        """Extract thumb and index finger tip positions"""
        # MediaPipe hand landmarks
        # Thumb tip: 4, Index finger tip: 8
        thumb_tip = landmarks.landmark[4]
        index_tip = landmarks.landmark[8]
        
        return {
            'thumb': (int(thumb_tip.x * self.screen_width), int(thumb_tip.y * self.screen_height)),
            'index': (int(index_tip.x * self.screen_width), int(index_tip.y * self.screen_height))
        }
    
    def detect_finger_tap(self, finger_positions):
        """Detect if thumb and index finger are close enough to count as a tap"""
        if finger_positions is None:
            return False, 0
        
        thumb_pos = finger_positions['thumb']
        index_pos = finger_positions['index']
        
        # Calculate distance between thumb and index finger
        distance = np.sqrt((thumb_pos[0] - index_pos[0])**2 + (thumb_pos[1] - index_pos[1])**2)
        
        # Always record the distance for tracking
        if self.current_hand == "right":
            self.right_hand_results["distances"].append(distance)
        else:
            self.left_hand_results["distances"].append(distance)
        
        # Check if fingers are close enough and enough time has passed
        current_time = time.time()
        if (distance < self.finger_distance_threshold and 
            current_time - self.last_tap_time > self.tap_cooldown):
            self.last_tap_time = current_time
            self.tap_count += 1
            
            # Update the appropriate hand results
            if self.current_hand == "right":
                self.right_hand_results["taps"] = self.tap_count
            else:
                self.left_hand_results["taps"] = self.tap_count
            
            return True, distance
        
        return False, distance
    
    def draw_gradient_background(self):
        """Draw a smooth gradient background from light pink to purple"""
        for y in range(self.screen_height):
            # Create gradient from soft pink at top to pale purple at bottom
            ratio = y / self.screen_height
            
            # Interpolate between soft pink and pale purple
            r = int(255 * (1 - ratio) + 240 * ratio)
            g = int(240 * (1 - ratio) + 230 * ratio)
            b = int(245 * (1 - ratio) + 250 * ratio)
            
            color = (r, g, b)
            pygame.draw.line(self.screen, color, (0, y), (self.screen_width, y))
    
    def draw_video_feed(self, frame, finger_positions):
        """Draw the video feed with finger tip markers"""
        # Only draw video if enabled
        if self.show_video:
            # Resize frame to fit display area
            resized_frame = cv2.resize(frame, (self.video_width, self.video_height))
            
            # Draw finger tip markers
            if finger_positions:
                for finger, pos in finger_positions.items():
                    # Convert screen coordinates to video coordinates
                    video_x = int(pos[0] * self.video_width / self.screen_width)
                    video_y = int(pos[1] * self.video_height / self.screen_height)
                    
                    # Draw colored circles for finger tips
                    color = (200, 150, 100) if finger == 'thumb' else (120, 150, 200)  # Orange for thumb, Blue for index
                    cv2.circle(resized_frame, (video_x, video_y), 6, color, -1)
                    cv2.circle(resized_frame, (video_x, video_y), 8, (255, 255, 255), 1)
                
                # Draw connection line between fingers
                thumb_pos = finger_positions['thumb']
                index_pos = finger_positions['index']
                thumb_video_x = int(thumb_pos[0] * self.video_width / self.screen_width)
                thumb_video_y = int(thumb_pos[1] * self.video_height / self.screen_height)
                index_video_x = int(index_pos[0] * self.video_width / self.screen_width)
                index_video_y = int(index_pos[1] * self.video_height / self.screen_height)
                
                # Draw line with different colors based on distance
                distance = np.sqrt((thumb_pos[0] - index_pos[0])**2 + (thumb_pos[1] - index_pos[1])**2)
                if distance < self.finger_distance_threshold:
                    cv2.line(resized_frame, (thumb_video_x, thumb_video_y), (index_video_x, index_video_y), (120, 180, 120), 3)  # Green when close
                else:
                    cv2.line(resized_frame, (thumb_video_x, thumb_video_y), (index_video_x, index_video_y), (180, 180, 180), 2)  # Gray when far
            
            # Convert BGR to RGB for pygame
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            
            # Convert to pygame surface
            frame_surface = pygame.surfarray.make_surface(rgb_frame.swapaxes(0, 1))
            
            # Draw subtle border with modern colors
            pygame.draw.rect(self.screen, self.SOFT_GRAY, 
                            (self.video_x - 1, self.video_y - 1, 
                             self.video_width + 2, self.video_height + 2), 1)
            
            # Blit video to screen
            self.screen.blit(frame_surface, (self.video_x, self.video_y))
        else:
            # Draw modern placeholder when video is off
            placeholder_rect = pygame.Rect(self.video_x, self.video_y, self.video_width, self.video_height)
            pygame.draw.rect(self.screen, self.WHITE, placeholder_rect)
            pygame.draw.rect(self.screen, self.SOFT_GRAY, placeholder_rect, 1)
    
    def draw_game_info(self):
        """Draw modern game information on screen"""
        if self.test_phase == "instructions":
            # Modern start screen
            title_text = self.large_font.render("UPDRS Finger Tapping Test", True, self.CHARCOAL)
            title_rect = title_text.get_rect(center=(self.screen_width // 2, 80))
            self.screen.blit(title_text, title_rect)
            
            start_text = self.font.render("Press SPACE to start with RIGHT hand!", True, self.CHARCOAL)
            start_rect = start_text.get_rect(center=(self.screen_width // 2, 120))
            self.screen.blit(start_text, start_rect)
            
            # Instructions
            instruction_text = self.small_font.render("Tap your thumb and index finger together repeatedly for 10 seconds", True, self.SLATE_GRAY)
            instruction_rect = instruction_text.get_rect(center=(self.screen_width // 2, 580))
            self.screen.blit(instruction_text, instruction_rect)
            
        elif self.test_phase == "right_hand":
            # Right hand test
            hand_text = self.font.render("RIGHT HAND TEST", True, self.CHARCOAL)
            hand_rect = hand_text.get_rect(center=(self.screen_width // 2, 80))
            self.screen.blit(hand_text, hand_rect)
            
            # Modern game info panel
            panel_x = 50
            panel_y = 50
            panel_width = 300
            panel_height = 150
            
            # Draw clean info panel background with subtle transparency effect
            pygame.draw.rect(self.screen, self.WHITE, (panel_x, panel_y, panel_width, panel_height))
            pygame.draw.rect(self.screen, self.SOFT_GRAY, (panel_x, panel_y, panel_width, panel_height), 1)
            
            # Test timer
            elapsed_time = time.time() - self.start_time
            remaining_time = max(0, self.test_duration - elapsed_time)
            timer_text = self.font.render(f"Time: {remaining_time:.1f}s", True, self.CHARCOAL)
            self.screen.blit(timer_text, (panel_x + 15, panel_y + 15))
            
            # Tap counter
            tap_text = self.font.render(f"Taps: {self.tap_count}", True, self.CHARCOAL)
            self.screen.blit(tap_text, (panel_x + 15, panel_y + 55))
            
            # Instructions
            if remaining_time > 0:
                instruction_text = self.small_font.render("Tap thumb and index finger together!", True, self.SLATE_GRAY)
                self.screen.blit(instruction_text, (panel_x + 15, panel_y + 105))
                
        elif self.test_phase == "left_hand":
            if not self.game_started:
                # Show instruction to start left hand test
                hand_text = self.font.render("LEFT HAND TEST", True, self.CHARCOAL)
                hand_rect = hand_text.get_rect(center=(self.screen_width // 2, 80))
                self.screen.blit(hand_text, hand_rect)
                
                start_text = self.font.render("Press SPACE to start LEFT hand test!", True, self.CHARCOAL)
                start_rect = start_text.get_rect(center=(self.screen_width // 2, 120))
                self.screen.blit(start_text, start_rect)
                
                # Show right hand results
                right_results_text = self.small_font.render(f"Right Hand: {self.right_hand_results['taps']} taps", True, self.SLATE_GRAY)
                right_results_rect = right_results_text.get_rect(center=(self.screen_width // 2, 160))
                self.screen.blit(right_results_text, right_results_rect)
            else:
                # Left hand test in progress
                hand_text = self.font.render("LEFT HAND TEST", True, self.CHARCOAL)
                hand_rect = hand_text.get_rect(center=(self.screen_width // 2, 80))
                self.screen.blit(hand_text, hand_rect)
                
                # Modern game info panel
                panel_x = 50
                panel_y = 50
                panel_width = 300
                panel_height = 150
                
                # Draw clean info panel background with subtle transparency effect
                pygame.draw.rect(self.screen, self.WHITE, (panel_x, panel_y, panel_width, panel_height))
                pygame.draw.rect(self.screen, self.SOFT_GRAY, (panel_x, panel_y, panel_width, panel_height), 1)
                
                # Test timer
                elapsed_time = time.time() - self.start_time
                remaining_time = max(0, self.test_duration - elapsed_time)
                timer_text = self.font.render(f"Time: {remaining_time:.1f}s", True, self.CHARCOAL)
                self.screen.blit(timer_text, (panel_x + 15, panel_y + 15))
                
                # Tap counter
                tap_text = self.font.render(f"Taps: {self.tap_count}", True, self.CHARCOAL)
                self.screen.blit(tap_text, (panel_x + 15, panel_y + 55))
                
                # Instructions
                if remaining_time > 0:
                    instruction_text = self.small_font.render("Tap thumb and index finger together!", True, self.SLATE_GRAY)
                    self.screen.blit(instruction_text, (panel_x + 15, panel_y + 105))
    
    def check_test_end(self):
        """Check if test should end"""
        if not self.game_started:
            return False
        
        elapsed_time = time.time() - self.start_time
        return elapsed_time >= self.test_duration
    
    def draw_distance_graph(self, distances, x_offset, y_offset, width, height, color, label):
        """Draw a simple distance graph"""
        if not distances:
            return
        
        # Draw graph background
        pygame.draw.rect(self.screen, self.WHITE, (x_offset, y_offset, width, height))
        pygame.draw.rect(self.screen, self.SOFT_GRAY, (x_offset, y_offset, width, height), 1)
        
        # Draw label
        label_text = self.small_font.render(label, True, self.CHARCOAL)
        self.screen.blit(label_text, (x_offset, y_offset - 25))
        
        if len(distances) < 2:
            return
        
        # Normalize distances to fit graph
        max_dist = max(distances)
        min_dist = min(distances)
        if max_dist == min_dist:
            max_dist = min_dist + 1
        
        # Draw graph lines
        for i in range(1, len(distances)):
            x1 = x_offset + int((i-1) * width / len(distances))
            y1 = y_offset + height - int((distances[i-1] - min_dist) / (max_dist - min_dist) * height)
            x2 = x_offset + int(i * width / len(distances))
            y2 = y_offset + height - int((distances[i] - min_dist) / (max_dist - min_dist) * height)
            
            pygame.draw.line(self.screen, color, (x1, y1), (x2, y2), 2)
        
        # Draw axes labels
        max_text = self.small_font.render(f"{max_dist:.0f}", True, self.SLATE_GRAY)
        min_text = self.small_font.render(f"{min_dist:.0f}", True, self.SLATE_GRAY)
        self.screen.blit(max_text, (x_offset + width + 5, y_offset))
        self.screen.blit(min_text, (x_offset + width + 5, y_offset + height - 15))
    
    def show_test_result(self):
        """Display test result with graphs"""
        # Calculate taps per second for both hands
        right_taps_per_second = self.right_hand_results["taps"] / self.test_duration
        left_taps_per_second = self.left_hand_results["taps"] / self.test_duration
        
        # Modern result panel
        panel_x = 50
        panel_y = 50
        panel_width = 900
        panel_height = 600
        
        # Draw result panel background
        pygame.draw.rect(self.screen, self.WHITE, (panel_x, panel_y, panel_width, panel_height))
        pygame.draw.rect(self.screen, self.SOFT_GRAY, (panel_x, panel_y, panel_width, panel_height), 2)
        
        # Result text
        result_text = self.large_font.render("Test Complete!", True, self.CHARCOAL)
        result_rect = result_text.get_rect(center=(panel_x + panel_width//2, panel_y + 30))
        self.screen.blit(result_text, result_rect)
        
        # Right hand results
        right_text = self.font.render("Right Hand Results", True, self.CHARCOAL)
        self.screen.blit(right_text, (panel_x + 20, panel_y + 80))
        
        right_taps_text = self.small_font.render(f"Taps: {self.right_hand_results['taps']} ({right_taps_per_second:.1f}/sec)", True, self.CHARCOAL)
        self.screen.blit(right_taps_text, (panel_x + 20, panel_y + 110))
        
        # Left hand results
        left_text = self.font.render("Left Hand Results", True, self.CHARCOAL)
        self.screen.blit(left_text, (panel_x + 20, panel_y + 150))
        
        left_taps_text = self.small_font.render(f"Taps: {self.left_hand_results['taps']} ({left_taps_per_second:.1f}/sec)", True, self.CHARCOAL)
        self.screen.blit(left_taps_text, (panel_x + 20, panel_y + 180))
        
        # Draw distance graphs
        graph_width = 400
        graph_height = 200
        
        # Right hand graph
        self.draw_distance_graph(
            self.right_hand_results["distances"], 
            panel_x + 20, panel_y + 220, 
            graph_width, graph_height, 
            self.MUTED_BLUE, "Right Hand Distance"
        )
        
        # Left hand graph
        self.draw_distance_graph(
            self.left_hand_results["distances"], 
            panel_x + 460, panel_y + 220, 
            graph_width, graph_height, 
            self.MUTED_ORANGE, "Left Hand Distance"
        )
        
        # Restart instruction
        restart_text = self.small_font.render("Press R to restart", True, self.SLATE_GRAY)
        restart_rect = restart_text.get_rect(center=(panel_x + panel_width//2, panel_y + 550))
        self.screen.blit(restart_text, restart_rect)
    
    def reset_test(self):
        """Reset test state"""
        self.game_started = False
        self.start_time = None
        self.tap_count = 0
        self.last_tap_time = 0
        self.current_hand = "right"
        self.right_hand_results = {"taps": 0, "distances": []}
        self.left_hand_results = {"taps": 0, "distances": []}
        self.test_phase = "instructions"
    
    def run(self):
        """Main game loop"""
        clock = pygame.time.Clock()
        test_ended = False
        
        while True:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.cleanup()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        if self.test_phase == "instructions":
                            # Start right hand test
                            self.test_phase = "right_hand"
                            self.game_started = True
                            self.start_time = time.time()
                            self.tap_count = 0
                            self.current_hand = "right"
                            test_ended = False
                        elif self.test_phase == "left_hand" and not self.game_started:
                            # Start left hand test
                            self.game_started = True
                            self.start_time = time.time()
                            self.tap_count = 0
                            self.current_hand = "left"
                            test_ended = False
                    elif event.key == pygame.K_r and self.test_phase == "results":
                        self.reset_test()
                        test_ended = False
                    elif event.key == pygame.K_v:
                        # Toggle video display
                        self.show_video = not self.show_video
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left mouse button
                        # Check if toggle button was clicked
                        button_width = 100
                        button_height = 35
                        button_x = self.video_x + self.video_width - button_width
                        button_y = self.video_y + self.video_height + 15
                        
                        if (button_x <= event.pos[0] <= button_x + button_width and 
                            button_y <= event.pos[1] <= button_y + button_height):
                            self.show_video = not self.show_video
            
            # Read camera frame
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process hands
            results = self.hands.process(rgb_frame)
            
            # Draw gradient background
            self.draw_gradient_background()
            
            # Draw game info
            self.draw_game_info()
            
            # Find target hand and get finger positions
            hand_landmarks, handedness = self.find_target_hand(results)
            finger_positions = None
            
            if hand_landmarks is not None:
                # Get finger tip positions
                finger_positions = self.get_finger_tips(hand_landmarks)
                
                if self.game_started and not test_ended:
                    # Detect finger taps
                    self.detect_finger_tap(finger_positions)
            
            # Draw video feed with finger markers
            self.draw_video_feed(frame, finger_positions)
            
            if self.game_started and not test_ended:
                # Check test end
                test_ended = self.check_test_end()
                if test_ended:
                    # Move to next phase
                    if self.test_phase == "right_hand":
                        self.test_phase = "left_hand"
                        self.game_started = False  # Reset game state for next phase
                    elif self.test_phase == "left_hand":
                        self.test_phase = "results"
                        self.game_started = False  # Reset game state for results
            
            # Show test result if in results phase
            if self.test_phase == "results":
                self.show_test_result()
            
            # Update display
            pygame.display.flip()
            clock.tick(60)
    
    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    game = UPDRSGame()
    try:
        game.run()
    except KeyboardInterrupt:
        game.cleanup()
