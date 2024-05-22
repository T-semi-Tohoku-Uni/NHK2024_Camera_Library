import numpy as np
import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
import threading
import queue
from enum import Enum
from queue import Queue
from .camera import RealsenseObject
from torch import Tensor
import typing
import time

# Camera Frame Width and Height[pxl]
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# depthのヒストグラムの刻み幅[mm]
DISTANCE_INTERVAL = 20

# ball in silo 判定の許容範囲[mm]
BALL_IN_SILO_RANGE = 150

# ball in silo 判定の許容閾値[pxl]
BALL_IN_SILO_THRESHOLD = 10

# realsense d435iのdepth最大/最小距離[mm]
RS_MAX_DISTANCE = 5000
RS_MIN_DISTANCE = 100

# デプス画像に対して大津の二値化を適用する時のデプスの間隔を表す配列
DEPTH_BINS=[i for i in range(RS_MIN_DISTANCE,RS_MAX_DISTANCE+RS_MIN_DISTANCE,DISTANCE_INTERVAL)]

# 籾の半径[mm]
PADDY_RICE_RADIUS = 100.0

# サイロの高さ[mm]
SILO_HEIGHT = 425.0

# 検出可能最大距離[mm]
DETECTABLE_MAX_DIS = 10000.0

# 検出した輪郭の最小面積(上部のカメラ)[pxl]
UPPER_MIN_CONTOUR_AREA_THRESHOLD = 180

# 検出した輪郭の最小面積(下部のカメラ)[pxl]
LOWER_MIN_CONTOUR_AREA_THRESHOLD = 800

# 上部カメラの円形度の閾値
UPPER_CIRCULARITY_THRESHOLD=0.2

# 下部カメラの円形度の閾値
LOWER_CIRCULARITY_THRESHOLD=0.2

# ロボット座標におけるアームのファンで吸い込めるエリアの中心と半径[mm]
OBTAINABE_AREA_CENTER_X = 0
OBTAINABE_AREA_CENTER_Y = 760
OBTAINABE_AREA_RADIUS = 50

# カメラからラインの検出点までの距離[mm]
LOWER_LINE_DETECTION_POINT_TO_CAMERA_DISTANCE = 990
UPPER_LINE_DETECTION_POINT_TO_CAMERA_DISTANCE = 990

# 上部カメラでボール検出した時のY[mm]
UPPER_STATIC_Y = 1000

# 画像端から何ピクセル分の点までを、画像端から伸びてる線分とみなすか
LINE_MARGIN = 30

# ラインの最大の太さ[pxl]
LINE_MAX_BOLD = 30

# 縦線、横線の角度の閾値[rad]
LINE_SLOPE_THRESHOLD = 0.69

# サイロの個数
NUMBER_OF_SILO = 5
    
class OUTPUT_ID(Enum):
    BALL = 0
    SILO = 1
    LINE = 2
    
class LINE_TYPE(Enum):
    FORWARD = 0
    RIGHT = 1
    LEFT = 2

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

def calc_distance(pxl_size :float, real_size :float) -> float:
    """
    球の半径から距離を計算する
    
    Parameters
    ----------
    pxl_size : float
    フレーム中のオブジェクトの大きさ(ピクセル)
    
    real_size : float
    実際のオブジェクトの大きさ(mm)
    
    Returns
    -------
    dis : float
        カメラからオブジェクトまでの距離[mm](キャリブレーションを用いた値)
    """
    pxl = FRAME_HEIGHT
    # y方向の焦点距離(inrofの時のBufferlo web camera)
    # fy = 470
    # y方向の焦点距離(Logi C615n)
    fy = 270
    # WebカメラのCMOSセンサー(1/4インチと仮定)の高さ[mm]
    camy = 2.7
    try:
        pxl_size = pxl_size * camy / pxl
        fy = fy * camy / pxl
        dis = real_size *  fy / pxl_size
    except ZeroDivisionError:
        dis = DETECTABLE_MAX_DIS
    return dis

def image_to_robot_coordinate_transformation(params,w, h, dis):
    """
    画像座標（ピクセル値）からロボット座標（mm）へ変換する:
    
    Parameters
    -----------
    param : tuple
        カメラの姿勢パラメータ
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
    (focal_length,pos_x,pos_y,pos_z,theta_x,theta_y,theta_z,_) = params
    internal_param_inv = np.array([[1/focal_length, 0, 0], [0,1/focal_length, 0], [0, 0, 1] ,[0, 0, 1/dis]])
    external_param = np.array([[np.cos(theta_z)*np.cos(theta_y), np.cos(theta_z)*np.sin(theta_y)*np.sin(theta_x)-np.sin(theta_z)*np.cos(theta_x), np.cos(theta_z)*np.sin(theta_y)*np.cos(theta_x)+np.sin(theta_z)*np.sin(theta_x), pos_x],
                        [np.sin(theta_z)*np.cos(theta_y), np.sin(theta_z)*np.sin(theta_y)*np.sin(theta_x)+np.cos(theta_z)*np.cos(theta_x), np.sin(theta_z)*np.sin(theta_y)*np.cos(theta_x)-np.cos(theta_z)*np.sin(theta_x), pos_y],
                        [-np.sin(theta_y), np.cos(theta_y)*np.sin(theta_x), np.cos(theta_y)*np.cos(theta_x), pos_z],
                        [0, 0, 0, 1]])
    # Opencvの座標でいう(FRAME_WIDTH/2, FRAME_HEIGHT/2)が(0,0)になるよう平行移動
    Target = np.array([[(w-FRAME_WIDTH/2)*dis], [(-h+FRAME_HEIGHT/2)*dis], [dis]])
    
    coordinate = external_param @ internal_param_inv @ Target 
    
    # 水平方向，奥行方向，垂直方向の順に返す
    return int(coordinate[0,0]), int(coordinate[2,0]), int(coordinate[1,0])

def threshold_otsu(hist, min_value=0, max_value=10):
    """
    大津の手法をdepthに適用

    Parameters
    --------------
        hist : _description_
        min_value (int, optional): ヒストグラムの最小値 Defaults to 0.
        max_value (int, optional): ヒストグラムの最大値 Defaults to 10.

    Returns
    --------------
        th : _description_
    """
    s_max = (0,-10)

    for th in range(min_value, max_value):
        # クラス1とクラス2の画素数を計算
        n1 = sum(hist[:th])
        n2 = sum(hist[th:])
        
        # クラス1とクラス2のヒストグラム値の平均を計算
        if n1 == 0 : mu1 = 0
        else : mu1 = sum([i * hist[i] for i in range(0,th)]) / n1   
        if n2 == 0 : mu2 = 0
        else : mu2 = sum([i * hist[i] for i in range(th, max_value)]) / n2

        # クラス間分散の分子を計算
        s = n1 * n2 * (mu1 - mu2) ** 2

        # クラス間分散の分子が最大のとき、クラス間分散の分子と閾値を記録
        if s > s_max[1]:
            s_max = (th, s)
    
    # クラス間分散が最大のときの閾値を返す
    return s_max[0]

def find_circle_contours(mask_img, min_contour_area_threshold, circularity_threshold):
    """
    マスク画像から円の輪郭を探す
    
    Parameters
    ----------
    mask_img : numpy.array
        マスク処理をした画像
    
    min_contour_area_threshold : int
        輪郭の面積の最小値
    
    Returns
    -------
    circles : list
        円の中心，半径の情報のリスト
    """
    contours = []
    circles = []
    contours, hierarchy = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        # 小さい輪郭は誤検出として削除
        contours = list(filter(lambda x: cv2.contourArea(x) > min_contour_area_threshold, contours))
        
        # 最小外接円を求める
        circles = [cv2.minEnclosingCircle(cnt) for cnt in contours if calc_circularity(cnt)>circularity_threshold]
    return circles


def bird_perspective_transform(frame, src):
    """
    俯瞰に変換
    """
    dst = np.array([[FRAME_WIDTH,FRAME_HEIGHT],[FRAME_WIDTH,0],[0,0],[0,FRAME_HEIGHT]],dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(src, dst)
    result = cv2.warpPerspective(frame,M,(FRAME_WIDTH,FRAME_HEIGHT))
    return result

def bird_to_robot_coordinate_transformation(cam_params,w,h,dis):
    """
    俯瞰からロボット座標に変換
    """
    
    (focal_length,pos_x,pos_y,pos_z,theta_x,theta_y,theta_z,point) = cam_params
    internal_param_inv = np.array([[1/focal_length, 0, 0], [0,1/focal_length, 0], [0, 0, 1] ,[0, 0, 1/dis]])
    external_param = np.array([[np.cos(theta_z)*np.cos(theta_y), np.cos(theta_z)*np.sin(theta_y)*np.sin(theta_x)-np.sin(theta_z)*np.cos(theta_x), np.cos(theta_z)*np.sin(theta_y)*np.cos(theta_x)+np.sin(theta_z)*np.sin(theta_x), pos_x],
                        [np.sin(theta_z)*np.cos(theta_y), np.sin(theta_z)*np.sin(theta_y)*np.sin(theta_x)+np.cos(theta_z)*np.cos(theta_x), np.sin(theta_z)*np.sin(theta_y)*np.cos(theta_x)-np.cos(theta_z)*np.sin(theta_x), pos_y],
                        [-np.sin(theta_y), np.cos(theta_y)*np.sin(theta_x), np.cos(theta_y)*np.cos(theta_x), pos_z],
                        [0, 0, 0, 1]])
    
    # pointの順番を変更し，かつopencv座標から左手座標へ
    x1 = point[3][0]
    x2 = point[0][0]
    x3 = point[2][0]
    x4 = point[1][0]
    y1 = FRAME_HEIGHT-point[3][1]
    y2 = FRAME_HEIGHT-point[0][1]
    y3 = FRAME_HEIGHT-point[2][1]
    y4 = FRAME_HEIGHT-point[1][1]
    
    # 係数求める
    c = x1
    f = y1
    h= ((x1-x2-x3+x4)*(y4-y2)-(y1-y2-y3+y4)*(x4-x2))/((x4-x2)*(y4-y3)-(x4-x3)*(y4-y2))
    g= (-x1+x2+x3-x4-(x4-x3)*h)/(x4-x2)
    a=(g+1)*x2-x1
    d=(g+1)*y2-y1
    b=(h+1)*x3-x1
    e=(h+1)*y3-y1
    
    # opencv座標から左手座標にし，単位正方形に変換
    target_x = w/FRAME_WIDTH
    target_y = (FRAME_HEIGHT-h)/FRAME_HEIGHT
    # 射影変換して，左手座標からopencvに変換
    _x = (a*target_x+b*target_y+c)/(g*target_x+h*target_y+1)
    _y = FRAME_HEIGHT-(d*target_x+e*target_y+f)/(g*target_x+h*target_y+1)
    
    # Opencvの座標でいう(FRAME_WIDTH/2, FRAME_HEIGHT/2)が(0,0)になるよう平行移動
    Target = np.array([[(_x-FRAME_WIDTH/2)*dis], [(-_y+FRAME_HEIGHT/2)*dis], [dis]])
    
    coordinate = external_param @ internal_param_inv @ Target 
    
    # 水平方向のみ返す
    return int(coordinate[0,0])

def detect_horizon_vertical(original_line_list, line_type, cam_param, line_detection_point_to_camera_distance, result_frame):
    """
    縦のラインもしくは横のラインを検出する
    
    Parameters
    ----------
    original_line_list : list
        元のライン情報のリスト
    line_type : Enum
        ラインの種類（縦，右，左）
    cam_param :tuple
        カメラのパラメータ
    line_detection_point_to_camera_distance:int
        camera to line dis
    result_frame :numpy
        ラインを表示する出力画像        
    Returns
    -------
    output : tuple
        縦の場合は(is_forward: bool,error_forward_x: float,error_forward_angle: float)
        右の場合は(is_right: bool)
        左の場合は(is_left: bool)
    """
    
    output = ()
    
    if line_type == LINE_TYPE.FORWARD:
        # 角度を基に縦線を求める
        filtered_line_list = [line for line in original_line_list if abs(np.pi/2-np.arccos(abs(line[0][0]-line[0][2])/np.sqrt((abs(line[0][0]-line[0][2])**2+abs(line[0][1]-line[0][3])**2))))<LINE_SLOPE_THRESHOLD]
        filtered_line_list = [line.astype(int) for line in filtered_line_list]
        # デバッグ用
        [cv2.line(result_frame,(p[0][0],p[0][1]),(p[0][2],p[0][3]),(0,255,0),3) for p in filtered_line_list]
             
        forward_list = []
        is_forward = False
        error_forward_x = 0.0
        error_forward_angle = 0.0    
        # ((縦線の下の点),(縦線の上の点))のリスト
        for p in filtered_line_list:
            if p[0][1] > p[0][3]:
                forward_list.append(((p[0][0],p[0][1]), (p[0][2],p[0][3])))
            else:
                forward_list.append(((p[0][2],p[0][3]), (p[0][0],p[0][1])))
        
        # 下点のxとFRAME_WIDTH/2の差について昇順ソート
        forward_list.sort(key=lambda x: abs(x[0][0]-FRAME_WIDTH/2))
        
        if len(forward_list)>=2:
            # もし線の太さが太すぎなければ
            if abs(forward_list[0][0][0]-forward_list[1][0][0]) < LINE_MAX_BOLD:
                # 縦線は存在True
                is_forward = True
                # 画像の中心に近い2本の線分のx座標の平均
                lower_x = (forward_list[0][0][0]+forward_list[1][0][0])/2
                upper_x = (forward_list[0][1][0]+forward_list[1][1][0])/2
                lower_y = (forward_list[0][0][1]+forward_list[1][0][1])/2
                upper_y = (forward_list[0][1][1]+forward_list[1][1][1])/2
                [cv2.drawMarker(result_frame,(int(lower_x),int(lower_y)),(255,255,0)) for p in filtered_line_list]
            
                error_forward_angle = - np.pi/2 + np.arccos((upper_x - lower_x)/np.sqrt((upper_x - lower_x)**2+(upper_y - lower_y)**2))
                error_forward_x = bird_to_robot_coordinate_transformation(cam_param,lower_x,lower_y,line_detection_point_to_camera_distance) 

        output = (is_forward, error_forward_x, error_forward_angle)
        
    elif line_type == LINE_TYPE.RIGHT:
        # 画像の右端に点がある線分のリスト
        filtered_line_list = [line for line in original_line_list if line[0][0]<LINE_MARGIN or line[0][2]<LINE_MARGIN]
        # 角度を基に横線を求める          
        filtered_line_list = [line for line in filtered_line_list if abs(np.arcsin(abs(line[0][1]-line[0][3])/np.sqrt((abs(line[0][0]-line[0][2])**2+abs(line[0][1]-line[0][3])**2))))<LINE_SLOPE_THRESHOLD]
        filtered_line_list = [line.astype(int) for line in filtered_line_list]
        # デバッグ用
        [cv2.line(result_frame,(p[0][0],p[0][1]),(p[0][2],p[0][3]),(0,255,0),3) for p in filtered_line_list]
        # 2本以上ラインがあればTrue
        is_right = True if len(filtered_line_list)>=2 else False
        output = (is_right)
        
    elif line_type == LINE_TYPE.LEFT:
        # 画像の左端に点がある線分のリスト
        filtered_line_list = [line for line in original_line_list if line[0][0]>FRAME_WIDTH-LINE_MARGIN or line[0][2]>FRAME_WIDTH-LINE_MARGIN]
        # 角度を基に横線を求める            
        filtered_line_list = [line for line in filtered_line_list if abs(np.arcsin(abs(line[0][1]-line[0][3])/np.sqrt((abs(line[0][0]-line[0][2])**2+abs(line[0][1]-line[0][3])**2))))<LINE_SLOPE_THRESHOLD]
        filtered_line_list = [line.astype(int) for line in filtered_line_list]
        # デバッグ用
        [cv2.line(result_frame,(p[0][0],p[0][1]),(p[0][2],p[0][3]),(0,255,0),3) for p in filtered_line_list]
        # 2本以上ラインがあればTrue
        is_left = True if len(filtered_line_list)>=2 else False
        output = (is_left)
        
    return output

def find_closest_ball_coordinates(
    target_ball_color,
    names,
    classes: Tensor, 
    boxes: Boxes,
    camera_params: tuple
) -> typing.Tuple[int, int, int, int, bool]:
    '''
    names: ラベルと文字がペアで入っている。names[0] = "redball"みたいな
    classes: 検出したラベルがTensor型で入る。例えば「赤, 赤, 紫」を検出した場合は「1, 1, 0」が入る
    '''
    
    # 一番近いボールの座標をpaddy_rice_x ...に入れるらしい
    paddy_rice_x = 0
    paddy_rice_y = 0
    paddy_rice_z = DETECTABLE_MAX_DIS
    is_obtainable = False
    
    # 赤のボールのboxesだけ取得する
    target_boxes: list[Boxes] = [
        box for box, cl in zip(boxes, classes) if names[int(cl)] == target_ball_color
    ]
    
    if len(target_boxes) == 0:
        return (len(target_boxes), paddy_rice_x, paddy_rice_y, paddy_rice_z, is_obtainable)
     
    # 空の場合
    if not target_boxes:
        return None
    
    for box in target_boxes:
        x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
        
        # 長方形の長辺を籾の半径とする
        r = max(abs(x1-x2)/2, abs(y1-y2)/2)
        z = calc_distance(r, PADDY_RICE_RADIUS)
        
        # 籾が複数ある場合は最も近いものの座標を返す
        if z < paddy_rice_z: # TODO: これあっているのか確認する
            (paddy_rice_x,paddy_rice_y,paddy_rice_z) = image_to_robot_coordinate_transformation(camera_params,int((x1+x2)/2),int((y1+y2)/2),z)
            
    is_obtainable = (paddy_rice_x-OBTAINABE_AREA_CENTER_X)**2 + (paddy_rice_y-OBTAINABE_AREA_CENTER_Y)**2 < OBTAINABE_AREA_RADIUS**2
    return (len(target_boxes), paddy_rice_x, paddy_rice_y, paddy_rice_z, is_obtainable)

class Silo:
    def __init__(self, x1, y1, x2, y2):
        # 自分のサイロの座標
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

        # それぞれの入っているボールの数
        self.__my_team_ball_cnt = 0
        self.__opponent_team_cnt = 0
        
        # TODO: 下から順にボールが入っているリスト
    
    def is_ball_in(self, ball_x, ball_y, is_my_team_ball):
        min_x, max_x = min(self.x1, self.x2), max(self.x1, self.x2)
        min_y, max_y = min(self.y1, self.y2), max(self.y1, self.y2)

        inside_points = []
        if min_x < ball_x and ball_x < max_x and min_y < ball_y and ball_y < max_y:
            inside_points.append((ball_x, ball_y))

            if is_my_team_ball:
                self.__my_team_ball_cnt += 1
            else:
                self.__opponent_team_cnt += 1
    
    def get_my_team_ball_cnt(self):
        return self.__my_team_ball_cnt
    
    def get_opponent_team_cnt(self):
        return self.__opponent_team_cnt
    
    def get_position(self):
        return (self.x1+self.x2)/2, (self.y1+self.y2)/2
     
    def __str__(self):
        return f"Box(({self.__my_team_ball_cnt}, {self.__opponent_team_cnt})"  
        
class DetectObj:
    def __init__(self, ball_model: YOLO, silo_model: YOLO):
        self.ball_model = ball_model
        self.silo_model = silo_model

        # preload model to GPU
        # preload_thread = threading.Thread(
        #     target = self.preload, daemon=True
        # ).start()
        self.ball_camera_out = (0,0.0,0.0,DETECTABLE_MAX_DIS,False)
        self.silo_camera_out = []
        self.line_camera_out = (False,False,False,0.0,0.0)
        
        # TODO
        # 画像表示するかどうか（q_outに画像とidを入れる
        self.show = True
        # 動画保存するかどうか
        self.save_movie = False
        
        print("Complete DetectObj Initialize")
    
    def preload(self):
        print("[Start]: preload ball model")
        self.ball_model.predict(np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3),dtype=np.uint8))
        print("[Complete]: preload ball model")
        
        print("[Start]: preload silo model")
        self.silo_model.predict(np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3),dtype=np.uint8))
        print("[Complete]: preload silo model")
    
    def detecting(self, ucam: RealsenseObject, lcam: RealsenseObject, q_out: Queue):
        print("[DetectObj.detecting]: start")
        while True:
            try:
                self.ball_detect(ucam, lcam, q_out)
                self.silo_detect(ucam, lcam, q_out)
            except Exception as e:
                print(f"{e}")
    
    def ball_detect(self, ucam: RealsenseObject, lcam: RealsenseObject, q_out: Queue):
        # 上についているカメラ（近くのボールを見つける）やつ
        lcam_frame, _ = lcam.read_image_buffer()
        lcam_results = self.ball_model.predict(lcam_frame, imgsz=320, conf=0.5, verbose=False) # TODO: rename model name 
        lcam_annotated_frame = lcam_results[0].plot()
        lcam_names = lcam_results[0].names
        lcam_classes = lcam_results[0].boxes.cls
        lcam_boxes = lcam_results[0].boxes
        lcam.save_image("red", lcam_frame, lcam_classes, lcam_boxes.xywhn)

        # 下についているカメラ（遠くのボールを見つける）やつ
        ucam_frame, _ = ucam.read_image_buffer()
        ucam_results = self.ball_model.predict(ucam_frame, imgsz=320, conf=0.5, verbose=False) # TODO: rename model name
        ucam_annotated_frame = ucam_frame
        ucam_annotated_frame = ucam_results[0].plot()
        ucam_names = ucam_results[0].names
        ucam_classes = ucam_results[0].boxes.cls
        ucam_boxes = ucam_results[0].boxes
        ucam.save_image("red", ucam_frame, ucam_classes, ucam_boxes.xywhn)
        
        boxes = lcam_results[0].boxes
        
        # (ターゲットの色のボールの数, 一番近いx, y, z, 取れるかどうか)のタプル形式
        # まず上からのカメラで認識
        ball_coordinates = find_closest_ball_coordinates(
            target_ball_color="redball",
            names=lcam_names,
            classes=lcam_classes,
            boxes=lcam_boxes,
            camera_params=lcam.params
        )
        
        # ボールを見つけられなかったら下のカメラ（遠くまで見れる方）でも確認する
        if ball_coordinates[0] == 0:
            
            ball_coordinates = find_closest_ball_coordinates(
                target_ball_color="redball",
                names=ucam_names,
                classes=ucam_classes,
                boxes=ucam_boxes,
                camera_params=ucam.params
            )
        
        # 検出したボールの座標をキューに送信 (xは水平，yは奥行方向)
        self.ball_camera_out = ball_coordinates
        # print(ball_coordinates)
        
        # デバッグで使うカメラキャプチャ画像の出力用
        show_frame = np.hstack((ucam_annotated_frame,lcam_annotated_frame))
        q_out.put((show_frame, OUTPUT_ID.BALL))

        
    def silo_detect(self, ucam: RealsenseObject, lcam: RealsenseObject, q_out: Queue, my_team_color="redball", opponent_team_color="blueball"):
        # サイロの画像を取得する
        frame, _ = ucam.read_image_buffer()
        results = self.silo_model.predict(frame, imgsz=320, conf=0.5, verbose=False)
        annotated_frame = results[0].plot()
        names = results[0].names
        classes = results[0].boxes.cls
        boxes = results[0].boxes
        
        # サイロの座標（(x1, y1), (x2, y2)）を取得する（TODO: x?座標でソート）
        silo_boxes: list[Boxes] = [
            box for box, cl in zip(boxes, classes) if names[int(cl)] == "silo"
        ]
        # サイロのインスタンスを一つずつ立てる
        silo_lists: typing.List[Silo] = []
        for silo_box in silo_boxes:
            x1, y1, x2, y2 = silo_box.xyxy[0]
            silo_lists.append(
                Silo(x1=x1, y1=y1, x2=x2, y2=y2)
            )
        
        # 自分のチームのボールの座標 ((x1, y1), (x2, y2))を取得する
        my_team_ball_box_in_silo = [
            box for box, cl in zip(boxes, classes) if names[int(cl)] == my_team_color
        ]
        # サイロの状態を更新
        for my_ball_box in my_team_ball_box_in_silo:
            for silo in silo_lists:
                x1, y1, x2, y2 = my_ball_box.xyxy[0]
                silo.is_ball_in(ball_x = (x1 + x2)/2, ball_y = (y1 + y2)/2, is_my_team_ball = True)
        
        # 相手チームのボールの座標 ((x1, y1), (x2, y2))を取得する
        opponent_team_in_silo = [
            box for box, cl in zip(boxes, classes) if names[int(cl)] == opponent_team_color
        ]
        # サイロの状態を更新
        for opponent_ball_box in opponent_team_in_silo:
            for silo in silo_lists:
                x1, y1, x2, y2 = opponent_ball_box.xyxy[0]
                silo.is_ball_in(ball_x = (x1 + x2)/2, ball_y = (y1 + y2)/2, is_my_team_ball = False)
        
        # print("#"*100)
        # for silo in silo_lists:
        #     print(silo)
        # print("#"*100)
            
        # TODO: 出力をいい感じにする
        
        q_out.put((frame, OUTPUT_ID.SILO))
        self.silo_camera_out = silo_lists

        # TODO: 消す
        # time.sleep(1)
        
            
    # 後方カメラでサイロを監視
    def silo_detect_temp(self, ucam: RealsenseObject, lcam: RealsenseObject, q_out: Queue):
        frame, _ = ucam.read_image_buffer()
        results = self.silo_model.predict(frame, imgsz=320, conf=0.5, verbose=False)
        annotated_frame = results[0].plot()
        names = results[0].names
        classes = results[0].boxes.cls
        boxes = results[0].boxes
        x1, y1, x2, y2 = [0, 0, FRAME_WIDTH, FRAME_HEIGHT]
        ucam.save_image("silo", frame, classes, boxes.xywhn)
        maximum_my_ball_in_silo_count = 0
        target_silo_x=0
        target_silo_y=0
        target_silo_z=0

        # ballのx1,y1,x2,y2を入れる
        ball_xyz = np.empty((0,4), int)
        
        for box, cls in zip(boxes, classes):
            name = names[int(cls)]
            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
            if(name == "blueball"):
                try:
                    ball_xyz = np.append(ball_xyz, [[x1,y1,x2,y2]],axis=0)
                except Exception as err:
                    print(f"This is inference_for_silo function in DetectObj class\nUnexpected {err=}, {type(err)=}")
        
        for box, cls in zip(boxes, classes):
            name = names[int(cls)]
            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
            my_ball_in_silo_counter = 0
            if(name == "silo"):
                for bxyz in ball_xyz:
                    if(x1<bxyz[0] and bxyz[2]<x2 and bxyz[3]<y2 and abs((x2-x1)-(bxyz[2]-bxyz[0]))<BALL_IN_SILO_THRESHOLD):
                        my_ball_in_silo_counter += 1
                cv2.putText(annotated_frame,f"{my_ball_in_silo_counter} in silo",(x1,y1+15),cv2.FONT_HERSHEY_PLAIN,1.0,(0,255,0),thickness=2)
            
            if maximum_my_ball_in_silo_count < my_ball_in_silo_counter:
                w = (x1+x2)/2
                h = (y1+y2)/2
                dis = calc_distance(abs(y1-y2),SILO_HEIGHT)
                (target_silo_x,target_silo_y,target_silo_z) = image_to_robot_coordinate_transformation(ucam.params,w,h,dis)
                maximum_my_ball_in_silo_count = my_ball_in_silo_counter
        
        output_data = (target_silo_x,target_silo_y,target_silo_z)
        q_out.put((annotated_frame, OUTPUT_ID.SILO))
        #if self.save_movie:
            #rcam.write(annotated_frame)
        self.silo_camera_out = output_data
                
    # def detecting_ball(self, ucam: RealsenseObject, lcam: RealsenseObject, q_out: Queue):
    #     while True:
    #         try:
    #             paddy_rice_x = 0
    #             paddy_rice_y = 0
    #             paddy_rice_z = DETECTABLE_MAX_DIS
    #             is_obtainable = False
                
    #             # lcam_frame = q_lcam.get()
    #             lcam_frame, _ = lcam.read_image_buffer()
    #             lcam_results = self.ball_model.predict(lcam_frame, imgsz=320, conf=0.5, verbose=False) # TODO: rename model name 
    #             lcam_annotated_frame = lcam_results[0].plot()
    #             names = lcam_results[0].names
    #             classes = lcam_results[0].boxes.cls
    #             boxes = lcam_results[0].boxes
                
    #             ucam_frame, _ = ucam.read_image_buffer()
    #             ucam_results = self.ball_model.predict(ucam_frame, imgsz=320, conf=0.5, verbose=False) # TODO: rename model name
    #             ucam_annotated_frame = ucam_frame
    #             ucam_annotated_frame = ucam_results[0].plot()
                
    #             # もし、下部カメラで検出できていれば
    #             if classes.dim() > 0:
    #                 for box, cls in zip(boxes, classes):
    #                     name = names[int(cls)]
    #                     if name == "blueball" :
    #                         x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
    #                         # 長方形の長辺を籾の半径とする
    #                         r = max(abs(x1-x2)/2, abs(y1-y2)/2)
    #                         z = calc_distance(r, PADDY_RICE_RADIUS)
    #                         # 籾が複数ある場合は最も近いものの座標を返す
    #                         if z < paddy_rice_z:
    #                             (paddy_rice_x,paddy_rice_y,paddy_rice_z) = image_to_robot_coordinate_transformation(lcam.params,int((x1+x2)/2),int((y1+y2)/2),z)
    #                 is_obtainable = (paddy_rice_x-OBTAINABE_AREA_CENTER_X)**2 + (paddy_rice_y-OBTAINABE_AREA_CENTER_Y)**2 < OBTAINABE_AREA_RADIUS**2
                    
    #             else:
    #                 pass
    #                 names = ucam_results[0].names
    #                 classes = ucam_results[0].boxes.cls
    #                 boxes = ucam_results[0].boxes
    #                 for box, cls in zip(boxes, classes):
    #                     name = names[int(cls)]
    #                     if(name == "blueball"):
    #                         x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
    #                         # 長方形の長辺を籾の半径とする
    #                         r = max(abs(x1-x2)/2, abs(y1-y2)/2)
    #                         z = calc_distance(r, PADDY_RICE_RADIUS)
    #                         # 籾が複数ある場合は最も近いものの座標を返す
    #                         if z < paddy_rice_z:
    #                             (paddy_rice_x,paddy_rice_y,paddy_rice_z) = image_to_robot_coordinate_transformation(ucam.params,int((x1+x2)/2),int((y1+y2)/2),z)
    #                 is_obtainable = (paddy_rice_x-OBTAINABE_AREA_CENTER_X)**2 + (paddy_rice_y-OBTAINABE_AREA_CENTER_Y)**2 < OBTAINABE_AREA_RADIUS**2
                
    #             show_frame = np.hstack((ucam_annotated_frame,lcam_annotated_frame))
    #             # 検出したボールの座標をキューに送信 (xは水平，yは奥行方向)
    #             output_data = (len(boxes), paddy_rice_x, paddy_rice_y, paddy_rice_z, is_obtainable)
    #             q_out.put((show_frame, OUTPUT_ID.BALL))
    #             self.ball_camera_out = output_data
            
    #         except KeyboardInterrupt:
    #             break
            
    # # 後方カメラでサイロを監視
    # def detecting_silo(self, ucam: RealsenseObject, lcam: RealsenseObject, q_out: Queue):
    #     while True:
    #         try:
    #             frame, _ = ucam.read_image_buffer()
    #             results = self.silo_model.predict(frame, imgsz=320, conf=0.5, verbose=False)
    #             annotated_frame = results[0].plot()
    #             names = results[0].names
    #             classes = results[0].boxes.cls
    #             boxes = results[0].boxes
    #             x1, y1, x2, y2 = [0, 0, FRAME_WIDTH, FRAME_HEIGHT]
    #             maximum_my_ball_in_silo_count = 0
    #             target_silo_x=0
    #             target_silo_y=0
    #             target_silo_z=0
        
    #             # ballのx1,y1,x2,y2を入れる
    #             ball_xyz = np.empty((0,4), int)
                
    #             for box, cls in zip(boxes, classes):
    #                 name = names[int(cls)]
    #                 x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
    #                 if(name == "blueball"):
    #                     try:
    #                         ball_xyz = np.append(ball_xyz, [[x1,y1,x2,y2]],axis=0)
    #                     except Exception as err:
    #                         print(f"This is inference_for_silo function in DetectObj class\nUnexpected {err=}, {type(err)=}")
                
    #             for box, cls in zip(boxes, classes):
    #                 name = names[int(cls)]
    #                 x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
    #                 my_ball_in_silo_counter = 0
    #                 if(name == "silo"):
    #                     for bxyz in ball_xyz:
    #                         if(x1<bxyz[0] and bxyz[2]<x2 and bxyz[3]<y2 and abs((x2-x1)-(bxyz[2]-bxyz[0]))<BALL_IN_SILO_THRESHOLD):
    #                             my_ball_in_silo_counter += 1
    #                     cv2.putText(annotated_frame,f"{my_ball_in_silo_counter} in silo",(x1,y1+15),cv2.FONT_HERSHEY_PLAIN,1.0,(0,255,0),thickness=2)
                    
    #                 if maximum_my_ball_in_silo_count < my_ball_in_silo_counter:
    #                     w = (x1+x2)/2
    #                     h = (y1+y2)/2
    #                     dis = calc_distance(abs(y1-y2),SILO_HEIGHT)
    #                     (target_silo_x,target_silo_y,target_silo_z) = image_to_robot_coordinate_transformation(ucam.params,w,h,dis)
    #                     maximum_my_ball_in_silo_count = my_ball_in_silo_counter
                
    #             output_data = (target_silo_x,target_silo_y,target_silo_z)
    #             q_out.put((annotated_frame, OUTPUT_ID.SILO))
    #             #if self.save_movie:
    #                 #rcam.write(annotated_frame)
    #             self.silo_camera_out = output_data
    #         except KeyboardInterrupt:
    #             break
