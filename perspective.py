import numpy as np
import cv2
import threading
import queue
from enum import Enum
from ultralytics import YOLO
from src import OUTPUT_ID,UpperCamera,LowerCamera,RearCamera,DetectObj

U_MERGIN = 80
L_MERGIN = 105
R_MERGIN = 80
W=320
H=240

def perspective_transform(q_in, q_out, point, id):
    while True:
        try:
            frame = q_in.get()
            src = point
            dst = np.array([[W,H],[W,0],[0,0],[0,H]],dtype=np.float32)
            M = cv2.getPerspectiveTransform(src, dst)
            result = cv2.warpPerspective(frame,M,(320,240))
            q_out.put((frame,result, id, point))
        except KeyboardInterrupt:
            break


class MainProcessForPerspective:
    def __init__(self,model_path):
        self.ucam = UpperCamera()
        self.lcam = LowerCamera(0)
        self.rcam = RearCamera()
        self.detector = DetectObj(model_path)

        self.thread_upper_capture = threading.Thread()
        self.thread_lower_capture = threading.Thread()
        self.thread_rear_capture = threading.Thread()
        self.thread_upper_detector = threading.Thread()
        self.thread_lower_detector = threading.Thread()
        self.thread_rear_detector = threading.Thread()
        
        # キューの辞書の宣言(上部カメラ画像のキュー，下部カメラ画像のキュー，Realsense画像のキュー，ロボット前の処理した画像のキュー，ロボット後ろの処理した画像のキュー)
        self.q_upper_in = queue.Queue(maxsize=1)
        self.q_lower_in = queue.Queue(maxsize=1)
        self.q_rear_in = queue.Queue(maxsize=1)
        self.q_out = queue.Queue(maxsize=3)
        
        self.upper_point = np.array([[W,H],[W-U_MERGIN,H*2/3],[U_MERGIN,H*2/3],[0,H]], dtype=np.float32)
        self.lower_point = np.array([[W,H],[W-L_MERGIN,0],[L_MERGIN,0],[0,H]], dtype=np.float32)
        self.rear_point = np.array([[W,H],[W-R_MERGIN,H*2/3],[R_MERGIN,H*2/3],[0,H]], dtype=np.float32)

    # カメラからの画像取得と画像処理、推論(デプス無し)をスレッドごとに分けて実行      
    def thread_start(self):
        self.thread_upper_capture = threading.Thread(target=self.detector.capturing, args=(self.q_upper_in,self.ucam), daemon=True)
        self.thread_lower_capture = threading.Thread(target=self.detector.capturing, args=(self.q_lower_in,self.lcam), daemon=True)
        self.thread_rear_capture = threading.Thread(target=self.detector.capturing, args=(self.q_rear_in,self.rcam),daemon=True)
        self.thread_upper_detector = threading.Thread(target=perspective_transform, args=(self.q_upper_in, self.q_out, self.upper_point, "up") ,daemon=True)
        self.thread_lower_detector = threading.Thread(target=perspective_transform, args=(self.q_lower_in, self.q_out, self.lower_point, "low") ,daemon=True)
        self.thread_rear_detector = threading.Thread(target=perspective_transform, args=(self.q_rear_in, self.q_out, self.rear_point, "rear") ,daemon=True)
        
        self.thread_upper_capture.start()
        self.thread_lower_capture.start()
        self.thread_rear_capture.start()
        self.thread_upper_detector.start()
        self.thread_lower_detector.start()
        self.thread_rear_detector.start()
        
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

if __name__ == "__main__":
    model_path = 'models/20240109best.pt'
    
    # メインプロセスを実行するクラス
    mainprocess_for_perspective = MainProcessForPerspective(model_path)
    
    # マルチスレッドの実行
    mainprocess_for_perspective.thread_start()
    
    while True:
        try:
            frame, result, id, point = mainprocess_for_perspective.q_out.get()
            cv2.polylines(frame,np.int32([point]),False,(0,0,255),thickness=2)
            cv2.imshow(f'{id}', np.hstack((frame,result)))
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
            
        except KeyboardInterrupt:
            break
    mainprocess_for_perspective.finish()
    
    cv2.destroyAllWindows()
