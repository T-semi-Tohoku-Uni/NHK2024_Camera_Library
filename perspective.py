import numpy as np
import cv2
import threading
import queue
from enum import Enum
import datetime
from ultralytics import YOLO
from src import OUTPUT_ID,UpperCamera,LowerCamera,RearCamera,DetectObj
# mergin80, h*2/3
U_MERGIN = 120
L_MERGIN = 105
R_MERGIN = 120
W=320
H=240

def perspective_transform(q_in, q_out, point, id):
    while True:
        try:
            frame = q_in.get()
            x1 = point[0][0]
            x2 = point[1][0]
            x3 = point[2][0]
            x4 = point[3][0]
            dst = np.array([[W,H],[W,0],[0,0],[0,H]],dtype=np.float32)
            M = cv2.getPerspectiveTransform(src, dst)
            M_inv = cv2.getPerspectiveTransform(dst,src)
            result = cv2.warpPerspective(frame,M,(W,H),flags=(cv2.INTER_LINEAR))
            
            cv2.drawMarker(frame,(int(point[0][0]),int(point[0][1])),(255,0,255))
            print(f"{M=}")
            x,y,_ = M_inv @ np.array([int(point[0][0]),int(point[0][1]),1])
            cv2.drawMarker(frame,(int(x),int(y)),(255,255,0))
            print(f"[W],[H]:{x=},{y=}")
            x,y,_ = M_inv @ np.array([int(point[1][0]),int(point[1][1]),1])
            cv2.drawMarker(frame,(int(x),int(y)),(255,255,0))
            print(f"[W],[0]:{x=},{y=}")
            x,y,_ = M_inv @ np.array([int(point[2][0]),int(point[2][1]),1])
            cv2.drawMarker(frame,(int(x),int(y)),(255,255,0))
            print(f"[0],[0]:{x=},{y=}")
            x,y,_ = M_inv @ np.array([int(point[3][0]),int(point[3][1]),1])
            cv2.drawMarker(frame,(int(x),int(y)),(255,255,0))
            print(f"[0],[H]:{x=},{y=}")
            #M_inv = cv2.getPerspectiveTransform(dst,src)
            
            
            #inv_x,inv_y,_ = M_inv @ np.array([point[1][0],point[1][1],1])
            #cv2.drawMarker(frame,(int(inv_x),int(inv_y)),(255,255,0))
            
            q_out.put((frame,result, id, point))
        except KeyboardInterrupt:
            break


class MainProcessForPerspective:
    def __init__(self,lib_path='/home/pi/NHK2024/NHK2024_R2_Raspi/src/NHK2024_Camera_Library', show=False, save_movie=False):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]
        self.ucam = UpperCamera(f"{lib_path}/camlogs/{timestamp}")
        self.lcam = LowerCamera(f"{lib_path}/camlogs/{timestamp}",0)
        self.rcam = RearCamera(f"{lib_path}/camlogs/{timestamp}")
        self.detector = DetectObj(f"{lib_path}/models/20240109best.pt")

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
        
        self.upper_point = np.array([[W,H],[W-U_MERGIN,H/2],[U_MERGIN,H/2],[0,H]], dtype=np.float32)
        self.lower_point = np.array([[W,H],[W-L_MERGIN,0],[L_MERGIN,0],[0,H]], dtype=np.float32)
        self.rear_point = np.array([[W,H],[W-R_MERGIN,H/2],[R_MERGIN,H/2],[0,H]], dtype=np.float32)

    # カメラからの画像取得と画像処理、推論(デプス無し)をスレッドごとに分けて実行      
    def thread_start(self):
        self.thread_upper_capture = threading.Thread(target=self.detector.capturing, args=(self.q_upper_in,self.ucam), daemon=True)
        self.thread_lower_capture = threading.Thread(target=self.detector.capturing, args=(self.q_lower_in,self.lcam), daemon=True)
        self.thread_rear_capture = threading.Thread(target=self.detector.capturing, args=(self.q_rear_in,self.rcam),daemon=True)
        self.thread_upper_detector = threading.Thread(target=perspective_transform, args=(self.q_upper_in, self.q_out, self.upper_point, "up") ,daemon=True)
        self.thread_lower_detector = threading.Thread(target=perspective_transform, args=(self.q_lower_in, self.q_out, self.lower_point, "low") ,daemon=True)
        self.thread_rear_detector = threading.Thread(target=perspective_transform, args=(self.q_rear_in, self.q_out, self.rear_point, "rear") ,daemon=True)
        
        self.thread_upper_capture.start()
        #self.thread_lower_capture.start()
        #self.thread_rear_capture.start()
        self.thread_upper_detector.start()
        #self.thread_lower_detector.start()
        #self.thread_rear_detector.start()
        
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
    lib_path = '.'
    
    # メインプロセスを実行するクラス
    mainprocess_for_perspective = MainProcessForPerspective(lib_path,True,False)
    
    # マルチスレッドの実行
    mainprocess_for_perspective.thread_start()
    
    while True:
        try:
            frame, result, id, point = mainprocess_for_perspective.q_out.get()
            cv2.polylines(frame,np.int32([point]),False,(0,0,255),thickness=2)
            cv2.imshow(id, np.hstack((frame,result)))
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
            
        except KeyboardInterrupt:
            break
    mainprocess_for_perspective.finish()
    
    cv2.destroyAllWindows()
