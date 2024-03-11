import numpy as np
import cv2
from ultralytics import YOLO
import queue

FRAME_WIDTH = 320
FRAME_HEIGHT = 240
PADDY_RICE_RADIUS = 200.0
DETECTABLE_MAX_DIS = 10000.0
OBTAINABLE_MAX_DIS = 2000.0
ROBOT_POS_X = 100
ROBOT_POS_Y = 400
ROBOT_POS_Z = 150
theta_x = 60*np.pi/180
theta_y = 0
theta_z = 0
OBTAINABE_AREA_MIN_X = 100
OBTAINABE_AREA_MAX_X = 220
OBTAINABE_AREA_MIN_Y = 80
OBTAINABE_AREA_MAX_Y = 140

q_frames = queue.Queue()

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
    fy = 554
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
    
    Returns
    x:水平方向
    y:垂直方向
    z:奥行方向
    """
    C_in_inv = np.array([[  0.0018382, 0, 0], [0, 0.0018051, 0], [0, 0, 1] ,[0, 0, 0]])
    C_pos = np.array([[ROBOT_POS_X],[ROBOT_POS_Y],[ROBOT_POS_Z],[0]])
    C_rot = np.array([[np.cos(theta_z)*np.cos(theta_y), np.cos(theta_z)*np.sin(theta_y)*np.sin(theta_x)-np.sin(theta_z)*np.cos(theta_x), np.cos(theta_z)*np.sin(theta_y)*np.cos(theta_x)+np.sin(theta_z)*np.sin(theta_x), 0],
                      [np.sin(theta_z)*np.cos(theta_y), np.sin(theta_z)*np.sin(theta_y)*np.sin(theta_x)+np.cos(theta_z)*np.cos(theta_x), np.sin(theta_z)*np.sin(theta_y)*np.cos(theta_x)-np.cos(theta_z)*np.sin(theta_x), 0],
                      [-np.sin(theta_y), np.cos(theta_y)*np.sin(theta_x), np.cos(theta_y)*np.cos(theta_x), 0],
                      [0, 0, 0, 1]])
    # inverse matrix
    C_rot_inv = np.linalg.inv(C_rot)
    
    Target = np.array([[w - FRAME_WIDTH/2], [-h + FRAME_HEIGHT/2], [1]])
    
    coordinate = C_rot_inv @ ((C_in_inv @ Target)*dis) + C_pos
    
    return int(coordinate[0,0]), int(coordinate[1,0]), int(coordinate[2,0])
    
class FrontCamera:
    def __init__(self, model_path, device_id):
        # Load the YOLOv8 model
        self.model = YOLO(model_path)
        
        # Camera Settings
        self.cap = cv2.VideoCapture(device_id)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret = False
        
        # Paddy Rice Parameters
        self.paddy_rice_x = 0
        self.paddy_rice_y = 0
        self.paddy_rice_z = DETECTABLE_MAX_DIS
        
    def DetectedObjectCounter(self) -> int:
        """
        検出したオブジェクト数を返す

        Returns:
            len(self.boxes) : int
                検知数
        """
        
        try:
            self.ret, img = self.cap.read()
            results = self.model.track(img, save=False, imgsz=320, conf=0.5, persist=True, verbose=False)
            self.annotated_frame = results[0].plot()
            self.names = results[0].names
            self.classes = results[0].boxes.cls
            self.boxes = results[0].boxes
        finally:
            return len(self.boxes)
    
    def ObjectPosition(self):
        try:
            self.paddy_rice_x = 0
            self.paddy_rice_y = 0
            self.paddy_rice_z = DETECTABLE_MAX_DIS
            for box, cls in zip(self.boxes, self.classes):
                name = self.names[int(cls)]
                if(name == "blueball"):
                    x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
                    # 長方形の短辺を籾の半径とする
                    r = min(abs(x1-x2), abs(y1-y2))
                    z = calc_distance(r)
                    if z < self.paddy_rice_z:
                        (self.paddy_rice_x, self.paddy_rice_y, self.paddy_rice_z) = coordinate_transformation(int((x1+x2)/2), int((y1+y2)/2), z)
        finally:
            return self.paddy_rice_x, self.paddy_rice_y, self.paddy_rice_z
    
    def IsObtainable(self):
        return (self.paddy_rice_x > OBTAINABE_AREA_MIN_X and self.paddy_rice_x < OBTAINABE_AREA_MAX_X
            and self.paddy_rice_y > OBTAINABE_AREA_MIN_Y and self.paddy_rice_y < OBTAINABE_AREA_MAX_Y
            and self.paddy_rice_z < OBTAINABLE_MAX_DIS)
        
    def __del__(self):
        self.cap.release()
        print("Closed capturing device")