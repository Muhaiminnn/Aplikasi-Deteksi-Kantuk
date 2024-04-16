from kivy.uix.screenmanager import ScreenManager, Screen, SlideTransition
from kivymd.app import MDApp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.label import MDLabel
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
import numpy as np
import mediapipe as mp
import pygame

def open_len(arr):
    y_arr = []

    for _, y in arr:
        y_arr.append(y)

    min_y = min(y_arr)
    max_y = max(y_arr)

    return max_y - min_y

mp_face_mesh = mp.solutions.face_mesh

RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
MOUTH_OUTLINE = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]

class PoseDetectionApp(MDApp):
    def build(self):
        self.screen_manager = ScreenManager()
        self.main_layout = MainLayout(name='main_layout')
        self.layout_2 = Layout_2(name='layout_2')
        self.screen_manager.add_widget(self.main_layout)
        self.screen_manager.add_widget(self.layout_2)
        return self.screen_manager

class MainLayout(Screen):
    def __init__(self, **kwargs):
        super(MainLayout, self).__init__(**kwargs)

        self.MDlabel = MDLabel(
            text="Aplikasi Deteksi Kantuk\nBy. Muhaimin",
            font_size="20sp",
            bold=True,
            size_hint=(None, None),
            size=(300, 50),
            pos_hint={'center_x': 0.5, 'center_y': 0.9},
            halign="center",
            valign="middle"
        )
        self.add_widget(self.MDlabel)

        image = Image(
            source="assets/logo/logofix.png",
            size_hint=(None, None),
            size=(100, 100),
            pos_hint={'center_x': 0.5, 'center_y': 0.1}
        )
        self.add_widget(image)

        # Button widget for starting the detection
        self.start_button = MDRaisedButton(
            text='Start',
            size_hint=(None, None),
            size=(150, 50),
            pos_hint={'center_x': 0.5, 'center_y': 0.5},
            md_bg_color=(0, 1, 0, 1)  # Green background color
        )
        self.start_button.bind(on_press=self.switch_to_layout_2)
        self.add_widget(self.start_button)

    def switch_to_layout_2(self, instance):
        #layout_2 = Layout_2(name='layout_2')
        #self.manager.add_widget(layout_2)
        self.manager.transition = SlideTransition(direction='left')
        self.manager.current = 'layout_2'

class Layout_2(Screen):
    def __init__(self, **kwargs):
        super(Layout_2, self).__init__(**kwargs)
        
        # Image widget for displaying camera feed
        self.image = Image()
        self.add_widget(self.image)
        self.image.pos_hint = {'center_x': 0.5, 'center_y': 0.5}

        self.stop_alarm = MDRaisedButton(
            text='Matikan Alarm',
            size_hint=(None, None),
            size=(150, 50),
            pos_hint={'center_x': 0.3, 'center_y': 0.05},
            md_bg_color=(0, 0, 1, 1)  # Red background color
        )
        self.stop_alarm.bind(on_press=self.toggle_alarm)
        self.add_widget(self.stop_alarm)

        self.startt_button = MDRaisedButton(
            text='Nyalakan Kamera',
            size_hint=(None, None),
            size=(150, 50),
            pos_hint={'center_x': 0.5, 'center_y': 0.05},
            md_bg_color=(1, 0, 0, 1)  # Blue background color
        )
        self.startt_button.bind(on_press=self.start_update)
        self.add_widget(self.startt_button)

        self.back_button = MDRaisedButton(
            text='Kembali',
            size_hint=(None, None),
            size=(150, 50),
            pos_hint={'center_x': 0.7, 'center_y': 0.05},
            md_bg_color=(1, 0.5, 0, 1)  # Black background color
        )
        self.back_button.bind(on_press=self.back_stop)
        self.add_widget(self.back_button)
        

        # Mediapipe parameters
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Handle for the webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        # Pygame initialization for sound
        pygame.mixer.init()
        self.sound = pygame.mixer.Sound("assets/audio/Alarm.mp3")

        # Drowsiness detection parameters
        self.drowsy_frames = 0
        self.drowsy_frames_mouth = 0
        self.max_left = 0
        self.max_right = 0
        self.max_mouth = 85

        #Start frame processing
        #self.update()
        #self.update_enabled =Clock.schedule_once(self.start_update, 0)  # Delayed call to start update
        
    def start_update(self, dt):
        Clock.schedule_once(self.update, 0)
        Clock.schedule_interval(self.update, 1.0 / 30.0)  # 30 FPS
    
    def back_stop(self, dt):
        # Stop pemanggilan fungsi update
        Clock.unschedule(self.update)
        self.manager.transition = SlideTransition(direction='right')
        self.manager.current = 'main_layout'
    

    def update(self, dt):
        # Get frame from webcam
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]

        # Process frame with Mediapipe
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            # Tidak ada wajah yang terdeteksi, aktifkan alarm
            cv2.putText(img=frame, text='Wajah Tidak Terdeteksi !!!', org=(100, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 165, 255), thickness=3)
            self.sound.play()
            # Menghentikan semua frame drowsiness
            self.drowsy_frames = 0
            self.drowsy_frames_mouth = 0

        #if results.multi_face_landmarks:
        else:
            all_landmarks = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])

            right_eye = all_landmarks[RIGHT_EYE]
            left_eye = all_landmarks[LEFT_EYE]
            mouth = all_landmarks[MOUTH_OUTLINE]

            cv2.polylines(frame, [left_eye], True, (0,255,0), 1, cv2.LINE_AA)
            cv2.polylines(frame, [right_eye], True, (0,255,0), 1, cv2.LINE_AA)
            cv2.polylines(frame, [mouth], True, (0,255,0), 1, cv2.LINE_AA) 

            len_left = open_len(right_eye)
            len_right = open_len(left_eye)
            len_mouth = open_len(mouth)

            cv2.putText(img=frame, text='Max: ' + str(self.max_left)  + ' Left Eye: ' + str(len_left), fontFace=0, org=(10, 30), fontScale=0.5, color=(0, 255, 0))
            cv2.putText(img=frame, text='Max: ' + str(self.max_right)  + ' Right Eye: ' + str(len_right), fontFace=0, org=(200, 30), fontScale=0.5, color=(0, 255, 0))
            cv2.putText(img=frame, text='Max: ' + str(self.max_mouth)  + ' Mouth: ' + str(len_mouth), fontFace=0, org=(400, 30), fontScale=0.5, color=(0, 255, 0))

            if len_left > self.max_left:
                self.max_left = len_left

            if len_right > self.max_right:
                self.max_right = len_right

            if len_mouth > self.max_mouth:
                self.max_mouth = len_mouth

            if (len_left <= int(self.max_left / 2) + 1 and len_right <= int(self.max_right / 2) + 1):
                self.drowsy_frames += 1
            else:
                self.drowsy_frames = 0

            if (len_mouth >= int(self.max_mouth/ 1.5) + 1):
                self.drowsy_frames_mouth += 1
            else:
                self.drowsy_frames_mouth = 0

            if (self.drowsy_frames > 20):
                cv2.putText(img=frame, text='Mata Tertutup!', org=(100, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.8, color=(0, 0, 255), thickness=3)
                self.sound.play()
            if (self.drowsy_frames_mouth > 20):
                cv2.putText(img=frame, text='Anda Menguap!', org=(100, 400), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.8, color=(0, 0, 255), thickness=3)
                self.sound.play()

        buf1 = cv2.flip(frame, 0)
        buf = buf1.tobytes()
        texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = texture1

    #def switch_to_main_layout(self, instance):
    #    self.manager.current = 'main_layout'

    def toggle_alarm(self, instance):
        self.sound.stop()

    def on_stop(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    PoseDetectionApp().run()
