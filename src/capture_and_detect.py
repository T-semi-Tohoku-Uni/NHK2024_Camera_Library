import threading
import queue
from enum import Enum
import datetime
from ultralytics import YOLO
from .camera import ImageSharedMemory, RealsenseObject
from .detect import DetectObj, OUTPUT_ID
from typing import Callable
import numpy as np
import torch
import os

class MainProcess:
    def __init__(
            self,
            ball_model_path,
            silo_model_path,
            show=False, 
            save_movie=False,
        ):
        print("[MainProcess.init]: start")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]
        
        # Get target device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[MainProcess.init]: device is {device}")
        
        # 最大使用率を設定
        # max_usage = 0.45

        # # CUDAの設定を調整
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用するGPUを指定（0番目のGPU）
        # torch.cuda.set_per_process_memory_fraction(max_usage, 0)

        
        # Create YOLO model
        ball_model = YOLO(ball_model_path).to(device)
        silo_model = YOLO(silo_model_path).to(device)
        # Create Detect Object
        self.detector = DetectObj(
            ball_model=ball_model, 
            silo_model=silo_model
       )
        
        # self.ucam = UpperCamera(f"{timestamp}")
        # self.ucam = RealsenseObject(
        #     timestamp=f"{timestamp}",
        #     serial_number='944122072123',
        #     focal_length=270,
        #     pos_x=70,
        #     pos_y=465,
        #     pos_z=800,
        #     theta_x = 40*np.pi/180,
        #     theta_y = 0,
        #     theta_z = 0,
        #     saved_image_dir="image_log/"
        # )
        self.ucam = RealsenseObject(
            timestamp=f"{timestamp}",
            serial_number='944122072123',
            focal_length=270,
            pos_x=-155,
            pos_y=430,
            pos_z=100,
            theta_x = 10*np.pi/180,
            theta_y = 0,
            theta_z = 0,
            saved_image_dir="image_log/"
        )
        # self.lcam = LowerCamera(f"{timestamp}")
        # 上についてるカメラ（ボール認識のみに使用）
        self.lcam = RealsenseObject(
            timestamp=f"{timestamp}",
            serial_number='242622071603',
            focal_length=270,
            pos_x=70,
            pos_y=800,
            pos_z=200,
            theta_x = 35*np.pi/180,
            theta_y = 0,
            theta_z = 0,
            saved_image_dir="image_log/"
        )

        self.thread_upper_capture = threading.Thread()
        self.thread_lower_capture = threading.Thread()
        self.thread_front_detector = threading.Thread()
        
        # キューの辞書の宣言(上部カメラ画像のキュー，下部カメラ画像のキュー，Realsense画像のキュー，ロボット前の処理した画像のキュー，ロボット後ろの処理した画像のキュー)
        self.q_upper_in = queue.Queue(maxsize=1)
        self.q_lower_in = queue.Queue(maxsize=1)
        self.q_out = queue.Queue(maxsize=3)
        
        # 画像表示するかどうか（q_outに画像とidを入れる）
        self.detector.show = show
        # 動画保存するかどうか
        self.detector.save_movie = save_movie
        
    # カメラからの画像取得と画像処理、推論(デプス無し)をスレッドごとに分けて実行      
    def thread_start(self):
        print("[MainProcess.thread_start]: start")
        # self.thread_upper_capture = threading.Thread(target=self.detector.capturing, args=(self.q_upper_in,self.ucam), daemon=True)
        # self.thread_upper_capture = self.ucam.capture_thread
        self.thread_upper_capture = threading.Thread(target=self.ucam.capture, daemon=True).start()
        # self.thread_lower_capture = threading.Thread(target=self.detector.capturing, args=(self.q_lower_in,self.lcam), daemon=True)
        self.thread_lower_capture = threading.Thread(target=self.lcam.capture, daemon=True).start()
        # self.thread_front_detector = threading.Thread(
        #     target=self.detector.detecting_ball, 
        #     args=(
        #         self.ucam,
        #         self.lcam,
        #         self.q_out
        #     ),
        #     daemon=True
        # )
        # self.thread_silo_detector = threading.Thread(
        #     target=self.detector.detecting_silo,
        #     args=(
        #         self.ucam,
        #         self.lcam,
        #         self.q_out
        #     ),
        #     daemon=True
        # )
        
        self.thread_detector = threading.Thread(
            target=self.detector.detecting,
            args=(
                self.ucam,
                self.lcam,
                self.q_out
            ),
            daemon=True
        ).start()
        
        # self.thread_upper_capture.start()
        # self.thread_lower_capture.start()
        # self.thread_detector.start()
        # self.thread_front_detector.start()
        # self.thread_silo_detector.start()
        print("[MainProcess.thread_start]: complete")
        
    def update_ball_camera_out(self):
        return self.detector.ball_camera_out
    
    def update_silo_camera_out(self):
        return self.detector.silo_camera_out
    
    def update_line_camera_out(self):
        return self.detector.line_camera_out
        
    
    # キューを空にする
    def terminate_queue(self):
        for q in (self.q_upper_in,self.q_lower_in,self.q_out):
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
        print("All Queue Empty")

    # release capture
    def terminate_camera(self):
        for cam in (self.ucam,self.lcam):
            cam.release()

    def finish(self):
        self.terminate_camera()
        self.terminate_queue()

        # TODO: カメラのプロセスの開放
                
        