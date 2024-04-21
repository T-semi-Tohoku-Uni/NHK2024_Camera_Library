import threading
import queue
from enum import Enum
import datetime
from ultralytics import YOLO
from .camera import UpperCamera,LowerCamera,RearCamera
from .detect import DetectObj, OUTPUT_ID

class MainProcess:
    def __init__(self,model_path='/home/pi/NHK2024/NHK2024_R2_Raspi/src/NHK2024_Camera_Library/models/20240109best.pt', show=False, save_movie=False):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]
        self.ucam = UpperCamera(f"{timestamp}")
        self.lcam = LowerCamera(f"{timestamp}")
        self.rcam = RearCamera(f"{timestamp}")
        #self.detector = DetectObj(f"{lib_path}/models/20240109best.pt")
        self.detector = DetectObj(model_path)


        self.thread_upper_capture = threading.Thread()
        self.thread_lower_capture = threading.Thread()
        self.thread_rear_capture = threading.Thread()
        self.thread_front_detector = threading.Thread()
        self.thread_rear_detector = threading.Thread()
        
        # キューの辞書の宣言(上部カメラ画像のキュー，下部カメラ画像のキュー，Realsense画像のキュー，ロボット前の処理した画像のキュー，ロボット後ろの処理した画像のキュー)
        self.q_upper_in = queue.Queue(maxsize=1)
        self.q_lower_in = queue.Queue(maxsize=1)
        self.q_rear_in = queue.Queue(maxsize=1)
        self.q_out = queue.Queue(maxsize=3)
        
        # 画像表示するかどうか（q_outに画像とidを入れる）
        self.detector.show = show
        # 動画保存するかどうか
        self.detector.save_movie = save_movie

    # カメラからの画像取得と画像処理、推論(デプス無し)をスレッドごとに分けて実行      
    def thread_start(self):
        self.thread_upper_capture = threading.Thread(target=self.detector.capturing, args=(self.q_upper_in,self.ucam), daemon=True)
        self.thread_lower_capture = threading.Thread(target=self.detector.capturing, args=(self.q_lower_in,self.lcam), daemon=True)
        self.thread_rear_capture = threading.Thread(target=self.detector.capturing, args=(self.q_rear_in,self.rcam),daemon=True)
        self.thread_front_detector = threading.Thread(target=self.detector.detecting_front, args=(self.ucam,self.lcam,self.rcam,self.q_upper_in,self.q_lower_in,self.q_rear_in,self.q_out),daemon=True)
        self.thread_rear_detector = threading.Thread(target=self.detector.detecting_rear, args=(self.ucam,self.lcam,self.rcam,self.q_upper_in,self.q_lower_in,self.q_rear_in,self.q_out),daemon=True)
        
        self.thread_upper_capture.start()
        self.thread_lower_capture.start()
        self.thread_rear_capture.start()
        self.thread_front_detector.start()
        self.thread_rear_detector.start()
        
    def update_ball_camera_out(self):
        return self.detector.ball_camera_out
    
    def update_silo_camera_out(self):
        return self.detector.silo_camera_out
    
    def update_line_camera_out(self):
        return self.detector.line_camera_out
        
    
    # キューを空にする
    def terminate_queue(self):
        for q in (self.q_upper_in,self.q_lower_in,self.q_rear_in,self.q_out):
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
        print("All Queue Empty")

    # release capture
    def terminate_camera(self):
        for cam in (self.ucam,self.lcam,self.rcam):
            cam.release()

    def finish(self):
        self.terminate_camera()
        self.terminate_queue()

                
        