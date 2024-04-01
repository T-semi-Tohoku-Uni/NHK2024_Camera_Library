import numpy as np
import cv2
from ultralytics import YOLO
import threading
import queue
import subprocess

# カメラからの画像の幅と高さ[pxl]
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# 籾の半径[mm]
PADDY_RICE_RADIUS = 100.0

# 検出可能最大距離[mm]
DETECTABLE_MAX_DIS = 10000.0

# 検出した輪郭の最小面積(手前側のカメラ)[pxl]
MIN_CONTOUR_AREA_THRESHOLD = 2000

# 円形度の閾値
CIRCULARITY_THRESHOLD=0.5

# ロボットの中心位置を原点とした時のカメラの位置[mm]
CAMERA_POS_X = 100
CAMERA_POS_Y = 400
CAMERA_POS_Z = 150

# カメラ座標におけるカメラの傾き[rad]
theta_x = 30*np.pi/180
theta_y = 0
theta_z = 0

# ロボット座標におけるアームのファンで吸い込めるエリアの中心と半径[mm]
OBTAINABE_AREA_CENTER_X = 0
OBTAINABE_AREA_CENTER_Y = 550
OBTAINABE_AREA_RADIUS = 80

def calc_circularity(cnt :np.ndarray) -> float:
    '''
    円形度を求める

    Parameters
    ----------
    cnt : np.ndarray
        輪郭の(x,y)座標の配列

    Returns
    -------
    cir : float
        円形度

    '''
    # 面積
    area = cv2.contourArea(cnt)
    # 周囲長
    length = cv2.arcLength(cnt, True)
    # 円形度を求める
    try:
        cir = 4*np.pi*area/length/length
    except ZeroDivisionError:
        cir = 0
    return cir

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
    pxl = FRAME_HEIGHT
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
    external_param = np.array([[np.cos(theta_z)*np.cos(theta_y), np.cos(theta_z)*np.sin(theta_y)*np.sin(theta_x)-np.sin(theta_z)*np.cos(theta_x), np.cos(theta_z)*np.sin(theta_y)*np.cos(theta_x)+np.sin(theta_z)*np.sin(theta_x), CAMERA_POS_X],
                      [np.sin(theta_z)*np.cos(theta_y), np.sin(theta_z)*np.sin(theta_y)*np.sin(theta_x)+np.cos(theta_z)*np.cos(theta_x), np.sin(theta_z)*np.sin(theta_y)*np.cos(theta_x)-np.cos(theta_z)*np.sin(theta_x), CAMERA_POS_Y],
                      [-np.sin(theta_y), np.cos(theta_y)*np.sin(theta_x), np.cos(theta_y)*np.cos(theta_x), CAMERA_POS_Z],
                      [0, 0, 0, 1]])
    
    # Opencvの座標でいう(FRAME_WIDTH/2, FRAME_HEIGHT/2)が(0,0)になるよう平行移動
    Target = np.array([[(w-FRAME_WIDTH/2)*dis], [(-h+FRAME_HEIGHT/2)*dis], [dis]])
    
    coordinate = external_param @ internal_param_inv @ Target 
    
    # 水平方向，奥行方向，垂直方向の順に返す
    return int(coordinate[0,0]), int(coordinate[2,0]), int(coordinate[1,0])

class FrontCamera:
    def __init__(self, device_id):
        # Camera Settings
        self.cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
        # self.cap = cv2.VideoCapture(device_id)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        set_fps = self.cap.set(cv2.CAP_PROP_FPS, 30)
        print(f"device_id_{device_id} fps:{self.cap.get(cv2.CAP_PROP_FPS)}, {set_fps}")
        
        camera_parameter = [cv2.CAP_PROP_FRAME_WIDTH,
        cv2.CAP_PROP_FRAME_HEIGHT,
        cv2.CAP_PROP_FOURCC,
        cv2.CAP_PROP_BRIGHTNESS,
        cv2.CAP_PROP_CONTRAST,
        cv2.CAP_PROP_SATURATION,
        cv2.CAP_PROP_HUE,
        cv2.CAP_PROP_GAIN,
        cv2.CAP_PROP_AUTO_EXPOSURE,
        cv2.CAP_PROP_EXPOSURE,
        cv2.CAP_PROP_AUTO_WB,
        cv2.CAP_PROP_WB_TEMPERATURE,
        cv2.CAP_PROP_AUTOFOCUS,
        ]

        params = ['cv2.CAP_PROP_FRAME_WIDTH',
        'cv2.CAP_PROP_FRAME_HEIGHT',
        'cv2.CAP_PROP_FOURCC',
        'cv2.CAP_PROP_BRIGHTNESS',
        'cv2.CAP_PROP_CONTRAST',
        'cv2.CAP_PROP_SATURATION',
        'cv2.CAP_PROP_HUE',
        'cv2.CAP_PROP_GAIN',
        'cv2.CAP_PROP_AUTO_EXPOSURE',
        'cv2.CAP_PROP_EXPOSURE',
        'cv2.CAP_PROP_AUTO_WB',
        'cv2.CAP_PROP_WB_TEMPERATURE',
        'cv2.CAP_PROP_AUTOFOCUS',
        ]
        """
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, -4.0)
        self.cap.set(cv2.CAP_PROP_CONTRAST, 15)
        self.cap.set(cv2.CAP_PROP_SATURATION, 32)
        self.cap.set(cv2.CAP_PROP_HUE, 0.0)
        self.cap.set(cv2.CAP_PROP_GAIN, -1.0)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, -1.0)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, 200)
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0.0)
        self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 2500)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, -1.0)
        """

        for x in range(len(params)):
            print(f"{params[x]} = {self.cap.get(camera_parameter[x])}")
    
        # v4l2-ctlを用いたホワイトバランスの固定
        #cmd = 'v4l2-ctl -d /dev/video0 -c white_balance_automatic=0 -c white_balance_temperature=4500'
        #ret = subprocess.check_output(cmd, shell=True)
        
    def read(self):
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        self.cap.release()
        print("Closed Capturing Device")
    
    def isOpened(self):
        return self.cap.isOpened()


class MainProcess:
    def __init__(self, model_path):
        # YOLOv8 modelのロード
        # self.model = YOLO(ncnn_model_path, task='detect')
        self.model = YOLO(model_path)
        
        # キューの宣言
        self.upper_cam_q_frames = queue.Queue(maxsize=1)
        self.lower_cam_q_frames = queue.Queue(maxsize=1)
        self.rs_q_frames = queue.Queue(maxsize=1)
        self.q_results = queue.Queue(maxsize=1)
        
        # maskの値を設定する
        self.blue_lower_mask = np.array([135, 50, 50])
        self.blue_upper_mask = np.array([160, 255, 255])
        self.purple_lower_mask = np.array([165,50,50])
        self.purple_upper_mask = np.array([230,255,255])
        self.red_lower_mask_1 = np.array([0,50,50])
        self.red_upper_mask_1 = np.array([10,255,255])
        self.red_lower_mask_2 = np.array([230,50,50])
        self.red_upper_mask_2 = np.array([255,255,255])
        
    # 画像を取得してキューに入れる
    def capturing(self, q_frames, cap):
        while True:
            try:
                ret, frame = cap.read()
                if not ret:
                    # frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3))
                    continue
                q_frames.put(frame)
                print(f"read frame")
            except KeyboardInterrupt:
                break
          
    # 画像処理をしてキューに入れる
    def image_processing(self, id, q_frames, q_results):
        while True:
            try:
                frame = q_frames.get()

                # 出力画像にガウシアンフィルタを適用する。
                frame = cv2.GaussianBlur(frame, ksize=(7,7),sigmaX=0)

                # カメラ画像をHSVに変換
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV_FULL)

                # hsvの輝度成分を抽出
                # vimg = hsv[:,:,2]

                # 輝度画像に対し適応的閾値処理で二値化
                # vimg = cv2.adaptiveThreshold(vimg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,9,3)

                # モルフォロジー変換でクロージング処理
                # vimg = cv2.morphologyEx(vimg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations=3)

                # hsvの輝度成分をvimgに変更
                # hsv[:,:,2] = vimg

                # HSV画像から指定したHSVの値でマスクを作成する
                # mask = cv2.inRange(hsv, np.array([0,50,50]), np.array([255,255,255]))
                mask = cv2.inRange(hsv,self.blue_lower_mask,self.blue_upper_mask)

                items = 0
                paddy_rice_x = 0
                paddy_rice_y = 0
                paddy_rice_z = DETECTABLE_MAX_DIS
                is_obtainable = False
                contours = []
                circles = []

                # 輪郭検出
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) > 0:
                    # 小さい輪郭は誤検出として削除
                    contours = list(filter(lambda x: cv2.contourArea(x) > MIN_CONTOUR_AREA_THRESHOLD, contours))
                    
                    # 最小外接円を求める
                    circles = [cv2.minEnclosingCircle(cnt) for cnt in contours if calc_circularity(cnt)>CIRCULARITY_THRESHOLD]
                    if len(circles) > 0:

                        # デバッグ用に円を描画
                        [cv2.circle(frame,(int(c[0][0]),int(c[0][1])),int(c[1]),(0,255,0),2) for c in circles]
                        
                        items = len(circles)
                        target = circles.index(max(circles, key=lambda x:x[1]))
                        (paddy_rice_x,paddy_rice_y,paddy_rice_z) = coordinate_transformation(int(circles[target][0][0]),int(circles[target][0][1]),calc_distance(circles[target][1]))
                        is_obtainable = (paddy_rice_x-OBTAINABE_AREA_CENTER_X)**2 + (paddy_rice_y-OBTAINABE_AREA_CENTER_Y)**2 < OBTAINABE_AREA_RADIUS**2
                
                # 画像のタイプを揃える
                hsv = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR_FULL)
                mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
                show_frame = np.hstack((frame,hsv,mask))
                q_results.put((show_frame, id, items, paddy_rice_x, paddy_rice_y, paddy_rice_z, is_obtainable))
                
            except KeyboardInterrupt:
                break
    
    # 推論してキューに入れる
    def inference(self, id, q_frames, q_results):
        while True:
            try:
                frame = q_frames.get()
                results = self.model.predict(frame, imgsz=320, conf=0.5, verbose=True)
                annotated_frame = results[0].plot()
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
                        r = max(abs(x1-x2)/2, abs(y1-y2)/2)
                        z = calc_distance(r)
                        # 籾が複数ある場合は最も近いものの座標を返す
                        if z < paddy_rice_z:
                            (paddy_rice_x, paddy_rice_y, paddy_rice_z) = coordinate_transformation(int((x1+x2)/2), int((y1+y2)/2), z)
                is_obtainable = (paddy_rice_x-OBTAINABE_AREA_CENTER_X)**2 + (paddy_rice_y-OBTAINABE_AREA_CENTER_Y)**2 < OBTAINABE_AREA_RADIUS**2
            
                # 検出したボールの座標をキューに送信 (xは水平，yは奥行方向)
                q_results.put((annotated_frame, id, len(boxes), paddy_rice_x, paddy_rice_y, paddy_rice_z, is_obtainable))
                print(f"inference")
                
            except KeyboardInterrupt:
                break
    
    # カメラからの画像取得と推論をスレッドごとに分けて実行      
    def thread_start(self, upper_cam, lower_cam=None):
        thread_upper_cam_1 = threading.Thread(target=self.capturing, args=(self.upper_cam_q_frames, upper_cam), daemon=True)
        thread_upper_cam_2 = threading.Thread(target=self.image_processing, args=(0, self.upper_cam_q_frames, self.q_results), daemon=True)
        thread_upper_cam_1.start()
        thread_upper_cam_2.start()
        
        if lower_cam is not None:
            thread_lower_cam_1.start()
            thread_lower_cam_2.start()
            thread_lower_cam_1 = threading.Thread(target=self.capturing, args=(self.lower_cam_q_frames, lower_cam), daemon=True)
            thread_lower_cam_2 = threading.Thread(target=self.image_processing, args=(1,self.lower_cam_q_frames, self.q_results), daemon=True)
        
        
    
    # キューを空にする
    def finish(self):
        for q in [self.upper_cam_q_frames,self.lower_cam_q_frames,self.q_results]:
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
                
        print("queue empty")