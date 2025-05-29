import cv2
import mediapipe as mp
import pyttsx3
import threading
import time
import json
import logging
import numpy as np

# Configure error logging
logging.basicConfig(filename='error_log.txt', level=logging.ERROR,
                    format='%(asctime)s: %(levelname)s: %(message)s')

# Gesture Detector Class
class GestureDetector:
    def __init__(self):
        """Initialize the webcam and MediaPipe Hands model."""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Camera not accessible")
        except Exception as e:
            logging.error(f"Camera initialization failed: {str(e)}")
            raise

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils

    def detect_gesture(self):
        """Capture frame, detect hand landmarks, and analyze gesture."""
        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logging.warning("Frame capture failed")
                return None, np.zeros((480, 640, 3), dtype=np.uint8), 0.0

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            gesture = None
            confidence = 1.0  # Default confidence if hand is detected

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    label = handedness.classification[0].label  # 'Left' or 'Right'
                    gesture = self.analyze_gesture(hand_landmarks, label)
                    break  # Only process one hand

            return gesture, frame, confidence
        except Exception as e:
            logging.error(f"Gesture detection error: {str(e)}")
            return None, np.zeros((480, 640, 3), dtype=np.uint8), 0.0

    def analyze_gesture(self, hand_landmarks, label):
        """Analyze hand landmarks to determine the gesture based on finger positions."""
        landmarks = hand_landmarks.landmark
        if len(landmarks) < 21:
            return None

        # Define key landmarks
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        index_pip = landmarks[6]
        middle_pip = landmarks[10]
        ring_pip = landmarks[14]
        pinky_pip = landmarks[18]

        # Check finger extension (tip.y < pip.y means extended, assuming upright hand)
        is_index_extended = index_tip.y < index_pip.y
        is_middle_extended = middle_tip.y < middle_pip.y
        is_ring_extended = ring_tip.y < ring_pip.y
        is_pinky_extended = pinky_tip.y < pinky_pip.y
        is_thumb_extended = (label == 'Right' and thumb_tip.x < landmarks[5].x - 0.05) or \
                            (label == 'Left' and thumb_tip.x > landmarks[5].x + 0.05)

        # Calculate finger tip positions for additional checks
        finger_tips_y = [index_tip.y, middle_tip.y, ring_tip.y, pinky_tip.y]
        min_y = min(finger_tips_y)
        max_y = max(finger_tips_y)
        finger_tips_x = [index_tip.x, middle_tip.x, ring_tip.x, pinky_tip.x]
        std_x = np.std(finger_tips_x)

        # Gesture detection logic (order matters for specificity)
        if thumb_tip.y < min_y - 0.1 and not is_index_extended and not is_middle_extended and \
           not is_ring_extended and not is_pinky_extended:
            return "thumbs_up"
        elif thumb_tip.y > max_y + 0.1 and not is_index_extended and not is_middle_extended and \
             not is_ring_extended and not is_pinky_extended:
            return "thumbs_down"
        elif not is_index_extended and not is_middle_extended and not is_ring_extended and not is_pinky_extended:
            return "fist"
        elif is_index_extended and not is_middle_extended and not is_ring_extended and not is_pinky_extended:
            return "one"
        elif is_index_extended and is_middle_extended and not is_ring_extended and not is_pinky_extended:
            return "two"  # Also used for language_switch
        elif is_index_extended and is_middle_extended and is_ring_extended and not is_pinky_extended:
            return "three"
        elif is_index_extended and is_middle_extended and is_ring_extended and is_pinky_extended:
            if std_x < 0.03:  # Fingers close together
                return "palm_together"
            else:  # Fingers spread apart
                return "four"
        elif is_thumb_extended and is_pinky_extended and not is_index_extended and \
             not is_middle_extended and not is_ring_extended:
            return "shaka"
        elif is_index_extended and is_pinky_extended and is_thumb_extended and \
             not is_middle_extended and not is_ring_extended:
            return "iloveyou"
        elif ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5 < 0.05 and \
             is_middle_extended and is_ring_extended and is_pinky_extended:
            return "ok"
        return None

# Gesture Interpreter Class
class GestureInterpreter:
    def __init__(self, config_file='gestures.json'):
        """Initialize gesture-to-message mappings."""
        self.gesture_to_message = {
            "thumbs_up": "Yes",
            "thumbs_down": "No",
            "fist": "Stop",
            "palm_together": "Hello",
            "one": "One",
            "two": "Two",  # Also triggers language switch
            "three": "Three",
            "four": "Four",
            "shaka": "Call me",
            "iloveyou": "I love you",
            "ok": "Okay",
            "language_switch": "Switch Language"
        }
        self.load_config(config_file)

    def load_config(self, config_file):
        """Load custom gesture mappings from a JSON file."""
        try:
            with open(config_file, 'r') as f:
                custom_mappings = json.load(f)
                self.gesture_to_message.update(custom_mappings)
        except Exception as e:
            logging.warning(f"Config file load failed: {str(e)}")

    def interpret(self, gesture):
        """Map detected gesture to a message."""
        return self.gesture_to_message.get(gesture, None)

# Output Handler Class
class OutputHandler:
    def __init__(self, language='en'):
        """Initialize text-to-speech and logging."""
        self.engine = pyttsx3.init()
        self.languages = ['en', 'es', 'fr']
        self.current_language = language
        self.set_language(language)
        self.last_spoken = None
        self.log_file = open("communication_log.txt", "a", encoding='utf-8')

    def set_language(self, lang_code):
        """Set the speech language."""
        try:
            voices = self.engine.getProperty('voices')
            for voice in voices:
                if lang_code in str(voice.languages).lower():
                    self.engine.setProperty('voice', voice.id)
                    self.current_language = lang_code
                    break
        except Exception as e:
            logging.error(f"Set language error: {str(e)}")

    def switch_language(self):
        """Cycle to the next supported language."""
        current_idx = self.languages.index(self.current_language)
        next_idx = (current_idx + 1) % len(self.languages)
        self.set_language(self.languages[next_idx])
        return self.languages[next_idx]

    def display_message(self, message, frame, confidence):
        """Display message and confidence on the frame."""
        if message:
            cv2.putText(frame, f"{message} ({self.current_language})", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence * 100:.1f}%", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "Show a gesture to communicate", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Updated gesture guide with new gestures
        cv2.putText(frame, "Gestures: 👍 Yes | 👎 No | ✊ Stop | 🖐 Hello | ☝ One | ✌ Two | 🤟 I love you | 👌 Okay | 🤙 Call me",
                    (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 255), 2)

    def speak(self, message):
        """Speak the message in a separate thread and log it."""
        def speak_thread():
            try:
                self.engine.stop()
                self.engine.say(message)
                self.engine.runAndWait()
            except Exception as e:
                logging.error(f"Speak error: {str(e)}")

        threading.Thread(target=speak_thread).start()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.log_file.write(f"{timestamp}: {message} ({self.current_language})\n")

    def cleanup(self):
        """Clean up resources."""
        try:
            self.engine.stop()
            self.log_file.close()
        except Exception as e:
            logging.error(f"Cleanup error: {str(e)}")

# Communication Assistant Class
class CommunicationAssistant:
    def __init__(self):
        """Initialize all components."""
        self.detector = GestureDetector()
        self.interpreter = GestureInterpreter()
        self.output = OutputHandler(language='en')
        self.prev_gesture = None
        self.gesture_counter = 0
        self.threshold = 10
        self.has_spoken = False
        self.start_time = time.time()
        self.frame_count = 0

    def adjust_threshold(self, frame):
        """Adjust gesture confirmation threshold based on frame brightness."""
        try:
            brightness = np.mean(frame)
            if brightness < 50:
                self.threshold = 15
            elif brightness > 200:
                self.threshold = 7
            else:
                self.threshold = 10
        except:
            self.threshold = 10

    def run(self):
        """Main loop to detect gestures and provide output."""
        while True:
            try:
                gesture, frame, confidence = self.detector.detect_gesture()
                self.frame_count += 1
                fps = self.frame_count / (time.time() - self.start_time + 0.001)
                self.adjust_threshold(frame)

                message = self.interpreter.interpret(gesture)
                self.output.display_message(message, frame, confidence)

                cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 100, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                # Gesture persistence logic
                if gesture == self.prev_gesture and gesture is not None:
                    self.gesture_counter += 1
                else:
                    self.prev_gesture = gesture
                    self.gesture_counter = 1
                    self.has_spoken = False

                # Speak message if gesture persists
                if self.gesture_counter >= self.threshold and not self.has_spoken:
                    if message:
                        if gesture == "two":  # Using "two" for language switch
                            new_lang = self.output.switch_language()
                            self.output.speak(f"Language switched to {new_lang}")
                        else:
                            self.output.speak(message)
                        self.has_spoken = True

                cv2.imshow("Gesture Communication Assistant", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception as e:
                logging.error(f"Run loop error: {str(e)}")
                break

        self.detector.cap.release()
        cv2.destroyAllWindows()
        self.output.cleanup()

# Main Execution
if __name__ == "__main__":
    try:
        assistant = CommunicationAssistant()
        assistant.run()
    except Exception as e:
        logging.error(f"Main application failure: {str(e)}")
        print("Application crashed. See error_log.txt for details.")