import numpy as np
import cv2
from ultralytics import YOLO
import threading
import queue

FRAME_WIDTH = 320
FRAME_HEIGHT = 240
INFERRED_WIDTH = 320
INFERRED_HEIGHT = 256
PADDY_RICE_RADIUS = 200.0
DETECTABLE_MAX_DIS = 10000.0
ROBOT_POS_X = 100
ROBOT_POS_Y = 400
ROBOT_POS_Z = 150
theta_x = 30*np.pi/180
theta_y = 0
theta_z = 0
OBTAINABE_AREA_CENTER_X = 0
OBTAINABE_AREA_CENTER_Y = 550
OBTAINABE_AREA_RADIUS = 100

def calc_distance(r :float) -> float:
    """
    球の半径から距離を計算する
    
    Parameters
    ----------
    r : float
    フレーム中の球の半径(ピクセル)
    
    Returns
    -------
    dis : float
        カメラから球までの距離[mm](キャリブレーションを用いた値)
    """
    pxl = INFERRED_HEIGHT
    # y方向の焦点距離(inrofの時のBufferlo web camera)
    # fy = 470
    # y方向の焦点距離(Logi C615n)
    fy = 270
    # WebカメラのCMOSセンサー(1/4インチと仮定)の高さ[mm]
    camy = 2.7
    try:
        r = r * camy / pxl
        fy = fy * camy / pxl
        dis = PADDY_RICE_RADIUS *  fy / r
    except ZeroDivisionError:
        dis = DETECTABLE_MAX_DIS
    return dis

def coordinate_transformation(w, h, dis):
    """
    画像座標（ピクセル値）からロボット座標（mm）へ変換する:
    
    Parameters
    -----------
    w : int
        幅 [pxl]
    h : int
        高さ [pxl]
    dis : int
        奥行方向 [mm]
    
    Returns
    -----------
    int(coordinate[0,0]) : int
        水平方向 [mm]
    int(coordinate[2,0]) : int
        奥行方向 [mm]
    int(coordinate[1,0]) : int
        垂直方向 [mm]
    
    """
    internal_param_inv = np.array([[0.0037037, 0, 0], [0, 0.0037037, 0], [0, 0, 1] ,[0, 0, 1/dis]])
    external_param = np.array([[np.cos(theta_z)*np.cos(theta_y), np.cos(theta_z)*np.sin(theta_y)*np.sin(theta_x)-np.sin(theta_z)*np.cos(theta_x), np.cos(theta_z)*np.sin(theta_y)*np.cos(theta_x)+np.sin(theta_z)*np.sin(theta_x), ROBOT_POS_X],
                      [np.sin(theta_z)*np.cos(theta_y), np.sin(theta_z)*np.sin(theta_y)*np.sin(theta_x)+np.cos(theta_z)*np.cos(theta_x), np.sin(theta_z)*np.sin(theta_y)*np.cos(theta_x)-np.cos(theta_z)*np.sin(theta_x), ROBOT_POS_Y],
                      [-np.sin(theta_y), np.cos(theta_y)*np.sin(theta_x), np.cos(theta_y)*np.cos(theta_x), ROBOT_POS_Z],
                      [0, 0, 0, 1]])
    
    # Opencvの座標でいう(INFERRED_WIDTH/2, INFERRED_HEIGHT/2)が(0,0)になるよう平行移動
    Target = np.array([[(w-INFERRED_WIDTH/2)*dis], [(-h+INFERRED_HEIGHT/2)*dis], [dis]])
    
    coordinate = external_param @ internal_param_inv @ Target 
    
    # 水平方向，奥行方向，垂直方向の順に返す
    return int(coordinate[0,0]), int(coordinate[2,0]), int(coordinate[1,0])

class FrontCamera:
    def __init__(self, device_id):
        # Camera Settings
        self.cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
    def read(self):
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        self.cap.release()
    
    def isOpened(self):
        return self.cap.isOpened()
            
    def __del__(self):
        self.cap.release()
        print("Closed Capturing Device")

class MainProcess:
    def __init__(self, model_path):
        # Load the YOLOv8 model
        # self.model = YOLO(ncnn_model_path, task='detect')
        self.model = YOLO(model_path)
        self.q_frames = queue.Queue(maxsize=10)
        self.q_results = queue.Queue(maxsize=10)
        # maskの値を設定する
        self.lower_mask = np.array([90, 50, 50])
        self.upper_mask = np.array([170, 255, 255])
    
    # 画像を取得してキューに入れる
    def capturing(self, q_frames, cap):
        while True:
            try:
                ret, frame = cap.read()
                if not ret:
                    #frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3))
                    break  
                q_frames.put(frame)
                print(f"read frame")
            except KeyboardInterrupt:
                break
    """      
    # 画像処理をしてキューに入れる
    def image_processing(self, q_frames, q_results):
        while True:
            try:
                frame = q_frames.get()
                # 画像をHSVに変換
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV_FULL)
                
                # 指定したHSVの値でマスクを作成する
                mask = cv2.inRange(hsv, self.lower_mask, self.upper_mask)
                
                # メディアンフィルタを適用する。
                mask = cv2.medianBlur(mask, ksize=5)
                
                # ハフ変換
                circles = cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)
                if circles != None:
                    circles = np.uint16(np.around(circles))
                else:
                    circles = np.array([[]])
                
                paddy_rice_x = 0
                paddy_rice_y = 0
                paddy_rice_z = DETECTABLE_MAX_DIS
                for i in circles[0,:]:
                    z = calc_distance(i[2])
                    if z < paddy_rice_z:
                        (paddy_rice_x, paddy_rice_y, paddy_rice_z) = coordinate_transformation(int(i[0]), int(i[1]), z)
                is_obtainable = (paddy_rice_x-OBTAINABE_AREA_CENTER_X)**2 + (paddy_rice_y-OBTAINABE_AREA_CENTER_Y)**2 < OBTAINABE_AREA_RADIUS**2
                
                #デバッグ用　画面に円を表示する準備
                cv2.circle(frame,(int(paddy_rice_x),int(paddy_rice_y)),int(paddy_rice_z),(0,255,0),2)
                
                # 検出したボールの座標をキューに送信 (xは水平，yは奥行方向)
                q_results.put((frame, len(circles[0]), paddy_rice_x, paddy_rice_y, paddy_rice_z, is_obtainable))
                print(f"image processing")
                
            except KeyboardInterrupt:
                break
    """ 
    # 推論してキューに入れる
    def inference(self, q_frames, q_results):
        while True:
            try:
                frame = q_frames.get()
                results = self.model.predict(frame, imgsz=320, conf=0.5, verbose=True)
                #annotated_frame = results[0].plot()
                names = results[0].names
                classes = results[0].boxes.cls
                boxes = results[0].boxes
                
                paddy_rice_x = 0
                paddy_rice_y = 0
                paddy_rice_z = DETECTABLE_MAX_DIS
                for box, cls in zip(boxes, classes):
                    name = names[int(cls)]
                    if(name == "blueball"):
                        x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
                        # 長方形の長辺を籾の半径とする
                        r = max(abs(x1-x2), abs(y1-y2))
                        z = calc_distance(r)
                        # 籾が複数ある場合は最も近いものの座標を返す
                        if z < paddy_rice_z:
                            (paddy_rice_x, paddy_rice_y, paddy_rice_z) = coordinate_transformation(int((x1+x2)/2), int((y1+y2)/2), z)
                is_obtainable = (paddy_rice_x-OBTAINABE_AREA_CENTER_X)**2 + (paddy_rice_y-OBTAINABE_AREA_CENTER_Y)**2 < OBTAINABE_AREA_RADIUS**2
            
                # 検出したボールの座標をキューに送信 (xは水平，yは奥行方向)
                q_results.put((len(boxes), paddy_rice_x, paddy_rice_y, paddy_rice_z, is_obtainable))
                print(f"inference")
                
            except KeyboardInterrupt:
                break
    
    
    """
    # 画像の読み込みと推論を実行してキューに入れる 
    def capturing_and_inference(self, q_results, cap):
        while True:
            try:
                ret, frame = cap.read()
                if not ret:
                    #frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3))
                    break  
                results = self.model.predict(frame, imgsz=320, conf=0.5, verbose=True)
                #annotated_frame = results[0].plot()
                names = results[0].names
                classes = results[0].boxes.cls
                boxes = results[0].boxes
                
                paddy_rice_x = 0
                paddy_rice_y = 0
                paddy_rice_z = DETECTABLE_MAX_DIS
                for box, cls in zip(boxes, classes):
                    name = names[int(cls)]
                    if(name == "blueball"):
                        x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
                        # 長方形の長辺を籾の半径とする
                        r = max(abs(x1-x2), abs(y1-y2))
                        z = calc_distance(r)
                        # 籾が複数ある場合は最も近いものの座標を返す
                        if z < paddy_rice_z:
                            (paddy_rice_x, paddy_rice_y, paddy_rice_z) = coordinate_transformation(int((x1+x2)/2), int((y1+y2)/2), z)
                is_obtainable = (paddy_rice_x-OBTAINABE_AREA_CENTER_X)**2 + (paddy_rice_y-OBTAINABE_AREA_CENTER_Y)**2 < OBTAINABE_AREA_RADIUS**2
            
                # 検出したボールの座標をキューに送信 (xは水平，yは奥行方向)
                q_results.put((len(boxes), paddy_rice_x, paddy_rice_y, paddy_rice_z, is_obtainable))
                print(f"read frame and inference")
                
            except KeyboardInterrupt:
                break
    """
    """
    # マルチプロセスで実行  
    def process_start(self, q_results, cap):
        processes = [multiprocessing.Process(target=self.capturing_and_inference, args=(q_results, cap), daemon=True) for i in range(self.process_num)]
        for process in processes:
            process.start()
            print(f"{process} start")
    """
    
    # カメラからの画像取得と推論をスレッドごとに分けて実行      
    def thread_start(self, cap):
        thread1 = threading.Thread(target=self.capturing, args=(self.q_frames, cap), daemon=True)
        thread2 = threading.Thread(target=self.inference, args=(self.q_frames, self.q_results), daemon=True)
        thread1.start()
        print("thread1 start")
        thread2.start()
        print("thread2 start")
        
        
        

"""
def camera_reader(_cap, out_buf, buf1_ready):
    
    カメラから画像を読みだしてバッファにためる
    
    Parameters
    -----------
    _cap : 
        カメラのキャプチャ
    out_buf : 
        読みだした画像
    buf1_ready : 
        out_bufのイベントオブジェクト
    
    Returns
    -----------

  
    def DetectedObjectCounter(self) -> int:
        
        検出したオブジェクト数を返す

        Returns:
            len(self.boxes) : int
                検知数
        
        
        try:
            self.buf1_ready.wait()
            img = np.reshape(self.buf1, (FRAME_HEIGHT, FRAME_WIDTH, 3))
            self.buf1_ready.clear()
            results = self.model.track(img, save=False, imgsz=320, conf=0.5, persist=True, verbose=False)
            self.annotated_frame = results[0].plot()
            self.names = results[0].names
            self.classes = results[0].boxes.cls
            self.boxes = results[0].boxes
        finally:
            return len(self.boxes)



    def ObjectPosition(self):
        
        籾の位置を返す

        Returns:
            int(self.paddy_rice_x): int
                ロボット座標の水平方向（カメラ側を前とした時の右が正）[mm]
            int(self.paddy_rice_y): int
                ロボット座標の垂直方向（カメラ側を前とした時の上が正）[mm]
            int(self.paddy_rice_z): int
                ロボット座標の奥行方向（カメラ側を前とした時の前が正）[mm]
        
        try:
            self.paddy_rice_x = 0
            self.paddy_rice_y = 0
            self.paddy_rice_z = DETECTABLE_MAX_DIS
            for box, cls in zip(self.boxes, self.classes):
                name = self.names[int(cls)]
                if(name == "blueball"):
                    x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
                    # 長方形の長辺を籾の半径とする
                    r = max(abs(x1-x2), abs(y1-y2))
                    z = calc_distance(r)
                    # 籾が複数ある場合は最も近いものの座標を返す
                    if z < self.paddy_rice_z:
                        (self.paddy_rice_x, self.paddy_rice_y, self.paddy_rice_z) = coordinate_transformation(int((x1+x2)/2), int((y1+y2)/2), z)
        finally:
            return int(self.paddy_rice_x), int(self.paddy_rice_y), int(self.paddy_rice_z)


    def IsObtainable(self):
        
        籾をピックアップできる領域内に籾が存在するかどうか

        Returns
        ----------
        領域内ならばTrue
        そうでないならばFalse
        
    """