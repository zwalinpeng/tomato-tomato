import cv2
import glob
import os
import time
from pathlib import Path
from dotenv import load_dotenv

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import numpy as np

load_dotenv()
 
# hand landmarks --------------------------------
WRIST = 0
THUMB_TIP, THUMB_IP, THUMB_MCP = 4, 3, 2
INDEX_TIP, INDEX_PIP, INDEX_MCP = 8, 6, 5
MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP = 12, 10, 9
RING_TIP, RING_PIP, RING_MCP = 16, 14, 13
PINKY_TIP, PINKY_PIP, PINKY_MCP = 20, 18, 17
 
FINGER_TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_PIPS = [THUMB_IP, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
FINGER_MCPS = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]

# model paths -----------------------------------
landmark_model_path = os.getenv('PROJECT_DIR') + '/models/hand_landmarker.task'
detector_model_path = os.getenv('PROJECT_DIR') + '/models/blaze_face_short_range.tflite'
recognizer_model_path = os.getenv('PROJECT_DIR') + '/models/gesture_recognizer.task'

# mediapipe constants ---------------------------
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
# hand landmarker
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
# face detector
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
FaceDetectorResult = mp.tasks.vision.FaceDetectorResult
# gesture recognizer
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult

# hand landmarker setup
# def print_landmarker_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
#     print('hand landmarker result: {}'.format(result))

# landmarker_options = HandLandmarkerOptions(
#     base_options=BaseOptions(model_asset_path=landmark_model_path),
#     num_hands=2,
#     running_mode=VisionRunningMode.LIVE_STREAM,
#     result_callback=print_landmarker_result)

#  class for managing tomato overlay ------------
class Tomato:
    def __init__(self, frames, tomatoing=False, frame_i=0):
        self.frames = frames        # initialize frames
        self.tomatoing = tomatoing  # whether overlay is playing
        self.frame_i = frame_i      # next frame for overlay
        self.x = 0                  # x position of overlay
        self.y = 0                  # y position of overlay
        self.overlay_height, self.overlay_width = frames[0].shape[:2]   # height and width of overlay
        self.center = (0, 0)
    
    # getters ---------------
    def get_x(self) -> int:
        return self.x

    def get_y(self) -> int:
        return self.y
    
    def is_tomatoing(self) -> bool:
        return self.tomatoing

    # generate position of overlay
    def generate_position(self, origin_x: int, origin_y: int, width: int, height: int) -> None:
        self.x = np.random.randint(origin_x, origin_x + width) - (self.overlay_width // 2)
        self.y = np.random.randint(origin_y, origin_y + height) - (self.overlay_height // 2)
        self.center = (self.x + (self.overlay_width // 2), self.y + (self.overlay_height // 2))

    # initialize tomato for throwing
    def start_tomato(self, face: list[int]) -> None:
        if not self.tomatoing:
            self.tomatoing = True
            origin_x, origin_y, width, height = face
            self.generate_position(origin_x, origin_y, width, height)

    # overlay tomato on given frame
    def overlay(self, frame: np.ndarray, draw: bool) -> np.ndarray:
        # find region of interest
        x = self.x
        y = self.y

        max_y = min(y + self.overlay_height, frame.shape[0])
        max_x = min(x + self.overlay_width, frame.shape[1])
        roi = frame[y:max_y, x:max_x]
        
        overlay_bgra = self.frames[self.frame_i]
        
        # crop overlay
        if roi.shape[:2] != overlay_bgra.shape[:2]:
            overlay_bgra = overlay_bgra[:roi.shape[0], :roi.shape[1], :]

        # split into color (BGR) and alpha channels
        overlay_bgr = overlay_bgra[:, :, :3]
        alpha = np.ceil(overlay_bgra[:, :, 3].astype(np.float32) / 255.0)
        alpha_3ch = cv2.merge([alpha, alpha, alpha])

        # overlay
        blended = (overlay_bgr.astype(float) * alpha_3ch + roi.astype(float) * (1 - alpha_3ch)).astype(np.uint8)  
        frame[y:max_y, x:max_x] = blended
        if draw:
            cv2.rectangle(frame, (x, y), (max_x, max_y), (0, 0, 255), 3)
            cv2.circle(frame, self.center, 3, (0, 0, 255), 3)

        return frame

    def play(self, frame: np.ndarray, draw: bool) -> np.ndarray:
        if not self.tomatoing:
            return frame
        
        res = self.overlay(frame, draw)
        self.frame_i += 1

        # overlay is done playing
        if self.frame_i >= len(self.frames):
            self.tomatoing = False
            self.frame_i = 0
        
        return res


# HELPER FUNCTIONS ------------------------------
 
# random rps generator
def rps() -> str:
    hand = np.random.rand()

    # probabilities based off of random reddit post that rock is thrown 35.4% of the time, paper 35.0% and scissors 29.4%
    # https://www.reddit.com/r/dataisbeautiful/comments/xtnsj3/oc_how_to_mathematically_win_at_rock_paper/
    if hand <= 0.354:
        return 'rock'
    elif hand <= .704:
        return 'paper'
    else:
        return 'scissors'

# -----------------------------------------------
def main() -> None:
    # webcam
    cap = cv2.VideoCapture(0)  
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # process tomato frames
    frame_paths = sorted(glob.glob('frames/*.png'))
    tomato_frames = [cv2.imread(p, cv2.IMREAD_UNCHANGED) for p in frame_paths]
    resized_overlays = [cv2.resize(f, None, fx=1.5, fy=1.5) for f in tomato_frames]
    overlay_height, overlay_width = resized_overlays[0].shape[:2]

    # initialize tomato
    tomato = Tomato(resized_overlays)

    # track results
    face =[0, 0, int(width - overlay_width), int(height - overlay_height)] # [origin_x, origin_y, width, height]
    gesture = [None, None]  # [handedness, gesture_label]
    draw = False

    # face detector setup
    def process_detector_result(result: FaceDetectorResult, output_image: mp.Image, timestamp_ms: int) -> None:
        # update face tracker if face is detected
        if result.detections:
            box = result.detections[0].bounding_box
            # print(f'x: {box.origin_x}\t\t y: {box.origin_y}\nwidth: {box.width}\t height: {box.height}\n')
            nonlocal face
            face = [box.origin_x, box.origin_y, box.width, box.height]

    detector_options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=detector_model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=process_detector_result)

    # gesture recognizer setup
    def process_recognizer_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int) -> None:
        # update gesture tracker if gesture is detected
        if result.handedness and result.gestures:
            # print(f'{result.handedness[0][0].category_name}: {result.gestures[0][0].category_name}')
            nonlocal gesture
            gesture = [result.handedness[0][0].category_name, result.gestures[0][0].category_name]
            # play tomato if thumbs down
            if result.gestures[0][0].category_name == 'Thumb_Down' and not tomato.is_tomatoing():
                tomato.start_tomato(face)

    recognizer_options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=recognizer_model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=process_recognizer_result)

    # video feed loop
    with FaceDetector.create_from_options(detector_options) as detector, GestureRecognizer.create_from_options(recognizer_options) as recognizer:
        while True:
            ok, frame = cap.read()

            if not ok:
                print('could not get webcam feed')
                break

            timestamp = int(round(time.time()*1000))

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            # landmarker.detect_async(mp_image, timestamp)
            detector.detect_async(mp_image, timestamp)
            recognizer.recognize_async(mp_image, timestamp)

            # label face
            if draw:
                cv2.rectangle(frame, (face[0], face[1]), (face[0] + face[2], face[1] + face[3]), (0, 255, 0), 3)

            # overlay tomato
            tomato.play(frame, draw)
            
            frame = cv2.flip(frame, 1) # mirror

            # label gesture
            cv2.putText(
                frame,
                f"{gesture[0]}: {gesture[1]}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA)
            
            cv2.imshow('oh hey', frame) 

            key = cv2.waitKey(1) & 0xFF;

            if key == ord('b'):     # 'b' to draw hitboxes
                draw = not draw
            elif key == ord('q'):   # 'q' to exit
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()