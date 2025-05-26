import cv2
from ultralytics import YOLO
from fer import FER
from datetime import datetime, timedelta
import random

class EmotionObjectDetector:
    def __init__(self, model_path="yolov8n.pt", video_source=0):
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(video_source)
        self.emotion_detector = FER()
        self.class_colors = {}
        self.default_color = (180, 180, 180)

        # สำหรับล็อกข้อมูลคน
        self.person_data = None
        self.person_data_time = None

        # ผี
        self.ghost_active = False
        self.ghost_size = (150, 250)
        self.ghost_data = {
            "age": random.randint(250, 600),
            "gender": "Male",
            "emotion": "angry"
        }
        self.ghost_pos_x = 0
        self.ghost_start_y = 0
        self.ghost_stopped = False
        self.ghost_position = "bottom_left"

        # === โหลด Age & Gender Detection Model ===
        self.age_net = cv2.dnn.readNetFromCaffe(
            'age_deploy.prototxt',
            'age_net.caffemodel'
        )
        self.gender_net = cv2.dnn.readNetFromCaffe(
            'gender_deploy.prototxt',
            'gender_net.caffemodel'
        )
        self.age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.gender_list = ['Male', 'Female']

    def get_color(self, label):
        if label not in self.class_colors:
            self.class_colors[label] = (
                random.randint(100, 255),
                random.randint(100, 255),
                random.randint(100, 255)
            )
        return self.class_colors[label]

    def draw_rounded_rectangle(self, img, start, end, color, radius=15, thickness=2):
        x1, y1 = start
        x2, y2 = end
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)

    def draw_label(self, img, text, pos, color):
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        thickness = 2
        (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
        x, y = pos
        cv2.rectangle(img, (x, y - h - 10), (x + w + 10, y), (0, 0, 0), -1)
        cv2.putText(img, text, (x + 5, y - 5), font, scale, color, thickness, lineType=cv2.LINE_AA)

    def draw_timestamp(self, frame):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        (w, h), _ = cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (10, 5), (10 + w + 10, 5 + h + 10), (50, 50, 50), -1)
        cv2.putText(frame, timestamp, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    def detect_emotion(self, face_img):
        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        results = self.emotion_detector.detect_emotions(face_img_rgb)
        if results:
            emotions = results[0]["emotions"]
            if emotions:
                top_emotion = max(emotions, key=emotions.get)
                return top_emotion
        return None

    def predict_age_gender(self, face_img):
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), 
                                     (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        
        # Gender prediction
        self.gender_net.setInput(blob)
        gender_preds = self.gender_net.forward()
        gender = self.gender_list[gender_preds[0].argmax()]

        # Age prediction
        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()
        age_range = self.age_list[age_preds[0].argmax()]

        return gender, age_range

    def spawn_ghost(self, frame_width, frame_height):
        self.ghost_start_y = random.randint(50, frame_height - self.ghost_size[1] - 50)

    def set_ghost_position(self, frame_width, frame_height):
        if self.ghost_position == "top_left":
            self.ghost_pos_x = 20
            self.ghost_start_y = 50
        elif self.ghost_position == "top_right":
            self.ghost_pos_x = frame_width - self.ghost_size[0] - 20
            self.ghost_start_y = 50
        elif self.ghost_position == "bottom_left":
            self.ghost_pos_x = 20
            self.ghost_start_y = frame_height - self.ghost_size[1] - 50
        elif self.ghost_position == "bottom_right":
            self.ghost_pos_x = frame_width - self.ghost_size[0] - 20
            self.ghost_start_y = frame_height - self.ghost_size[1] - 50

    def process_frame(self, frame):
        frame_height, frame_width = frame.shape[:2]
        results = self.model(frame, stream=True)

        person_detected = False

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                label = self.model.names[cls]
                color = self.get_color(label)

                if label == "person":
                    person_detected = True
                    face_img = frame[y1:y2, x1:x2]

                    if self.person_data is None or datetime.now() > self.person_data_time:
                        gender, age_range = self.predict_age_gender(face_img)
                        emotion = self.detect_emotion(face_img) or "neutral"
                        self.person_data = {
                            "gender": gender,
                            "age": age_range,
                            "emotion": emotion
                        }
                        self.person_data_time = datetime.now() 

                        self.ghost_active = True
                        self.spawn_ghost(frame_width, frame_height)

                    mem = self.person_data
                    label_text = f"{label} {conf:.2f} ({mem['emotion']}) {mem['gender']} {mem['age']}"
                    self.draw_rounded_rectangle(frame, (x1, y1), (x2, y2), color)
                    self.draw_label(frame, label_text, (x1, y1), color)

                else:
                    label_text = f"{label} {conf:.2f}"
                    self.draw_rounded_rectangle(frame, (x1, y1), (x2, y2), color)
                    self.draw_label(frame, label_text, (x1, y1), color)

        if self.ghost_active:
            self.set_ghost_position(frame_width, frame_height)

            ghost_x1 = self.ghost_pos_x
            ghost_y1 = self.ghost_start_y
            ghost_x2 = ghost_x1 + self.ghost_size[0]
            ghost_y2 = ghost_y1 + self.ghost_size[1]

            ghost_color = (255, 100, 255)
            self.draw_rounded_rectangle(frame, (ghost_x1, ghost_y1), (ghost_x2, ghost_y2), ghost_color)
            ghost_label = f"???? {self.ghost_data['gender']} {self.ghost_data['age']}yr ({self.ghost_data['emotion']})"
            self.draw_label(frame, ghost_label, (ghost_x1, ghost_y1), ghost_color)

        if not person_detected:
            self.ghost_active = False

        self.draw_timestamp(frame)

        return frame

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                breakq

            frame = cv2.resize(frame, (640, 480))
            processed_frame = self.process_frame(frame)

            cv2.imshow("YOLOv8 + Emotion + Ghost (Realtime)", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

# ==== Main ==== 
if __name__ == "__main__":
    detector = EmotionObjectDetector()
    detector.run()
