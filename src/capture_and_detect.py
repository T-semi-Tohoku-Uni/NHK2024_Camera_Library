import threading
import queue
from enum import Enum
from ultralytics import YOLO
from .camera import UpperCamera,LowerCamera,RearCamera
from .detect import DetectObj

class OUTPUT_ID(Enum):
    BALL = 0
    SILO = 1
    LINE = 2


class MainProcess:
    def __init__(self,model_path):
        self.ucam = UpperCamera()
        self.lcam = LowerCamera(0)
        self.rcam = RearCamera()
        self.object_detector = DetectObj(model_path)

        self.thread_upper_capturing = threading.Thread()
        self.thread_lower_capturing = threading.Thread()
        self.thread_front_detecting = threading.Thread()
        self.thread_rear_capturing = threading.Thread()
        self.thread_rear_detecting = threading.Thread()
        
        # キューの辞書の宣言(上部カメラ画像のキュー，下部カメラ画像のキュー，Realsense画像のキュー，ロボット前の処理した画像のキュー，ロボット後ろの処理した画像のキュー)
        self.q_upper_in = queue.Queue(maxsize=1)
        self.q_lower_in = queue.Queue(maxsize=1)
        self.q_rear_in = queue.Queue(maxsize=1)
        self.q_out = queue.Queue(maxsize=1)

    # カメラからの画像取得と画像処理、推論(デプス無し)をスレッドごとに分けて実行      
    def thread_start(self):
        self.thread_upper_capturing = threading.Thread(target=self.object_detector.capturing, args=(self.q_upper_in,self.ucam), daemon=True)
        self.thread_lower_capturing = threading.Thread(target=self.object_detector.capturing, args=(self.q_lower_in,self.lcam), daemon=True)
        self.thread_front_detecting = threading.Thread(target=self.object_detector.detecting_ball_or_line, args=(OUTPUT_ID.BALL,OUTPUT_ID.LINE,self.ucam.params,self.lcam.params,self.q_upper_in,self.q_lower_in,self.q_out),daemon=True)
        self.thread_rear_capturing = threading.Thread(target=self.object_detector.capturing, args=(self.q_rear_in,self.rcam),daemon=True)
        self.thread_rear_detecting = threading.Thread(target=self.object_detector.inference_for_silo, args=(OUTPUT_ID.SILO,self.q_rear_in,self.q_out),daemon=True)
        
        self.thread_upper_capturing.start()
        self.thread_lower_capturing.start()
        self.thread_front_detecting.start()
        self.thread_rear_capturing.start()
        self.thread_rear_detecting.start()

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

                
        