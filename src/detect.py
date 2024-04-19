import numpy as np
import cv2
from ultralytics import YOLO
import threading
import queue
from enum import Enum

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
UPPER_MIN_CONTOUR_AREA_THRESHOLD = 200

# 検出した輪郭の最小面積(下部のカメラ)[pxl]
LOWER_MIN_CONTOUR_AREA_THRESHOLD = 800


# 下部カメラの円形度の閾値
LOWER_CIRCULARITY_THRESHOLD=0.3

# 上部カメラの円形度の閾値
UPPER_CIRCULARITY_THRESHOLD=0.3

# ロボット座標におけるアームのファンで吸い込めるエリアの中心と半径[mm]
OBTAINABE_AREA_CENTER_X = 0
OBTAINABE_AREA_CENTER_Y = 550
OBTAINABE_AREA_RADIUS = 60

# カメラからラインの検出点までの距離[mm]
LOWER_LINE_DETECTION_POINT_TO_CAMERA_DISTANCE = 575
UPPER_LINE_DETECTION_POINT_TO_CAMERA_DISTANCE = 990

# 画像端から何ピクセル分の点までを、画像端から伸びてる線分とみなすか
LINE_MARGIN = 30

# 縦線、横線の角度の閾値[rad]
LINE_SLOPE_THRESHOLD = 0.52

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
        円の輪郭情報のリスト
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
    src = np.array([[FRAME_WIDTH,FRAME_HEIGHT],[FRAME_WIDTH,0],[0,0],[0,FRAME_HEIGHT]],dtype=np.float32)
    (focal_length,pos_x,pos_y,pos_z,theta_x,theta_y,theta_z,dst) = cam_params
    M_inv = cv2.getPerspectiveTransform(src, dst)
    internal_param_inv = np.array([[1/focal_length, 0, 0], [0,1/focal_length, 0], [0, 0, 1] ,[0, 0, 1/dis]])
    external_param = np.array([[np.cos(theta_z)*np.cos(theta_y), np.cos(theta_z)*np.sin(theta_y)*np.sin(theta_x)-np.sin(theta_z)*np.cos(theta_x), np.cos(theta_z)*np.sin(theta_y)*np.cos(theta_x)+np.sin(theta_z)*np.sin(theta_x), pos_x],
                        [np.sin(theta_z)*np.cos(theta_y), np.sin(theta_z)*np.sin(theta_y)*np.sin(theta_x)+np.cos(theta_z)*np.cos(theta_x), np.sin(theta_z)*np.sin(theta_y)*np.cos(theta_x)-np.cos(theta_z)*np.sin(theta_x), pos_y],
                        [-np.sin(theta_y), np.cos(theta_y)*np.sin(theta_x), np.cos(theta_y)*np.cos(theta_x), pos_z],
                        [0, 0, 0, 1]])
    # Opencvの座標でいう(FRAME_WIDTH/2, FRAME_HEIGHT/2)が(0,0)になるよう平行移動
    Target = np.array([[(w-FRAME_WIDTH/2)*dis], [(-h+FRAME_HEIGHT/2)*dis], [dis]])
    
    coordinate = external_param @ internal_param_inv @ M_inv @ Target 
    
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
        # 2本以上ラインがあればTrue
        is_forward = True if len(filtered_line_list)>=2 else False
             
        forward_list = []
        error_forward_x = 0.0
        error_forward_angle = 0.0    
        # ((縦線の下の点),(縦線の上の点))のリスト
        for p in filtered_line_list:
            if p[0][1] > p[0][3]:
                forward_list.append(((p[0][0],p[0][1]), (p[0][2],p[0][3])))
            else:
                forward_list.append(((p[0][2],p[0][3]), (p[0][0],p[0][1])))
        # 下点のxについて昇順ソート
        forward_list.sort(key=lambda x: x[0][0])    
        
        if len(forward_list)>=2:
            # 画像の中心に近い2本の線分のx座標の平均
            lower_x = (forward_list[0][0][0]+forward_list[1][0][0])/2
            upper_x = (forward_list[0][1][0]+forward_list[1][1][0])/2
            lower_y = (forward_list[0][0][1]+forward_list[1][0][1])/2
            upper_y = (forward_list[0][1][1]+forward_list[1][1][1])/2
        
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


class DetectObj:
    def __init__(self,model_path):
        # YOLOv8 modelのロード
        # self.model = YOLO(ncnn_model_path, task='detect')
        self.model = YOLO(model_path)
        
        # maskの値を設定する
        self.blue_lower_mask = np.array([139, 20, 20])
        self.blue_upper_mask = np.array([165, 255, 255])
        self.purple_lower_mask = np.array([165,40,40])
        self.purple_upper_mask = np.array([230,250,250])
        self.red_lower_mask_1 = np.array([0,40,40])
        self.red_upper_mask_1 = np.array([10,255,255])
        self.red_lower_mask_2 = np.array([230,40,40])
        self.red_upper_mask_2 = np.array([255,255,255])
        self.white_lower_mask = np.array([0,120,80])
        self.white_upper_mask = np.array([255,255,255])
        
        # fast line detector
        self.fld = cv2.ximgproc.createFastLineDetector(length_threshold=10,distance_threshold=1.41421356,canny_th1=150.0,canny_th2=180.0,canny_aperture_size=3,do_merge=True)
     
        self.ball_camera_out = (0,0.0,0.0,DETECTABLE_MAX_DIS,False)
        self.silo_camera_out = (0.0,0.0,0.0)
        self.line_camera_out = (False,False,False,0.0,0.0)
        
    # 画像を取得してキューに入れる
    def capturing(self, q_frames, cap):
        while True:
            try:
                ret, frame, _ = cap.read()
                if not ret:
                    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3),dtype=np.uint8)
                q_frames.put(frame)
            except KeyboardInterrupt:
                break
          
    # 画像(depthも)を取得してキューに入れる
    def capturing_with_depth(self, q_frames, cap):
        while True:
            try:
                ret, color, depth = cap.read()
                if not ret:
                    color = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3),dtype=np.uint8)
                    depth = np.zeros((FRAME_HEIGHT, FRAME_WIDTH),dtype=np.uint8)
                q_frames.put((color, depth))
            except KeyboardInterrupt:
                break
                
    # 前方カメラでライン検出，ボール検出（閾値によるマスキング）を行う
    def detecting_front(self,ucam_params,lcam_params,rcam_params,q_ucam,q_lcam,q_rcam,q_out):
        while True:
            try:
                ###ライン検出ここから###
                # 奥行方向のラインがあるかどうか：bool
                is_forward = False
                # 右方向のラインがあるかどうか：bool
                is_right = False
                # 左方向のラインがあるかどうか：bool
                is_left = False
                # 奥行方向のラインの、水平方向のずれを出力(ロボットの中心から前方向300mmくらい)
                error_forward_x = 0.0
                # 奥行方向のラインの角度のずれ
                error_forward_angle = 0.0
                
                # 下部カメラから画像を読み込む
                lcam_frame = q_lcam.get()
                
                # Gaussian Blur
                lcam_line_blur = cv2.GaussianBlur(lcam_frame, ksize=(3,3),sigmaX=0)
                
                _, _, _, _, _, _, _, lower_bird_point = lcam_params
                bird_frame = bird_perspective_transform(lcam_line_blur, lower_bird_point)
                
                # BGRのBを抽出
                l_blue = bird_frame[:,:,0]
                
                # ライン検出
                l_lines = self.fld.detect(l_blue)

                # image for debug
                l_all_lines = self.fld.drawSegments(lcam_frame,l_lines)
                
                # 上部カメラから画像を読み込む
                ucam_frame = q_ucam.get()
                
                # Gaussian Blur
                ucam_line_blur = cv2.GaussianBlur(ucam_frame, ksize=(3,3),sigmaX=0)
                
                _, _, _, _, _, _, _, upper_bird_point = ucam_params
                bird_frame = bird_perspective_transform(ucam_line_blur, upper_bird_point)
                
                # BGRのBを抽出
                u_blue = bird_frame[:,:,0]
                
                # ライン検出
                u_lines = self.fld.detect(u_blue)

                # image for debug
                u_all_lines = self.fld.drawSegments(ucam_frame,u_lines)
                
                l_filtered_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3),dtype=np.uint8)
                u_filtered_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3),dtype=np.uint8)
                        
                if l_lines is not None:
                    # 右線かどうかの判定
                    (is_right) = detect_horizon_vertical(l_lines, LINE_TYPE.RIGHT,lcam_params, LOWER_LINE_DETECTION_POINT_TO_CAMERA_DISTANCE, l_filtered_frame)
                    # 左線かどうかの判定
                    (is_left) = detect_horizon_vertical(l_lines,LINE_TYPE.LEFT,lcam_params, LOWER_LINE_DETECTION_POINT_TO_CAMERA_DISTANCE, l_filtered_frame)
                    # 縦線かどうかの判定
                    (is_forward,error_forward_x,error_forward_angle) = detect_horizon_vertical(l_lines, LINE_TYPE.FORWARD, lcam_params, LOWER_LINE_DETECTION_POINT_TO_CAMERA_DISTANCE, l_filtered_frame)
                    
                    # 縦線が無ければ
                    if not is_forward:
                        if u_lines is not None:
                            # 右線かどうかの判定
                            (is_right) = detect_horizon_vertical(u_lines, LINE_TYPE.RIGHT,ucam_params, UPPER_LINE_DETECTION_POINT_TO_CAMERA_DISTANCE, u_filtered_frame)
                            # 左線かどうかの判定
                            (is_left) = detect_horizon_vertical(u_lines,LINE_TYPE.LEFT,ucam_params, UPPER_LINE_DETECTION_POINT_TO_CAMERA_DISTANCE,u_filtered_frame)
                            # 縦線かどうかの判定
                            (is_forward,error_forward_x,error_forward_angle) = detect_horizon_vertical(u_lines, LINE_TYPE.FORWARD, ucam_params, UPPER_LINE_DETECTION_POINT_TO_CAMERA_DISTANCE,u_filtered_frame)
                l_blue = cv2.cvtColor(l_blue,cv2.COLOR_GRAY2BGR)
                l_line_show_frame = np.hstack((l_all_lines,l_blue,l_filtered_frame))            
                u_blue = cv2.cvtColor(u_blue,cv2.COLOR_GRAY2BGR)
                u_line_show_frame = np.hstack((u_all_lines,u_blue,u_filtered_frame))
                line_show_frame = np.vstack((u_line_show_frame,l_line_show_frame))
                # キューに結果を入れる
                output_data = (is_forward, is_right, is_left, error_forward_x, error_forward_angle)
                #q_out.put((line_show_frame, OUTPUT_ID.LINE, output_data))
                self.line_camera_out = output_data
                
                ###ライン検出ここまで###
                
                ###ボール検出ここから###
                items = 0
                paddy_rice_x = 0
                paddy_rice_y = 0
                paddy_rice_z = DETECTABLE_MAX_DIS
                is_obtainable = False

                ucam_ball_blur = cv2.GaussianBlur(ucam_frame, ksize=(7,7),sigmaX=0)
                lcam_ball_blur = cv2.GaussianBlur(lcam_frame, ksize=(7,7),sigmaX=0)
                
                # カメラ画像をHSVに変換
                ucam_hsv = cv2.cvtColor(ucam_ball_blur, cv2.COLOR_BGR2HSV_FULL)
                lcam_hsv = cv2.cvtColor(lcam_ball_blur, cv2.COLOR_BGR2HSV_FULL)

                # 閾値でmasking処理
                ucam_mask = cv2.inRange(ucam_hsv,self.blue_lower_mask,self.blue_upper_mask)
                lcam_mask = cv2.inRange(lcam_hsv,self.blue_lower_mask,self.blue_upper_mask)
                
                # モルフォロジー変換でクロージング処理
                ucam_close = cv2.morphologyEx(ucam_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations=3)
                lcam_close = cv2.morphologyEx(lcam_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations=3)
                
                # 下部カメラで円を検出する
                circles = find_circle_contours(lcam_close,LOWER_MIN_CONTOUR_AREA_THRESHOLD,LOWER_CIRCULARITY_THRESHOLD)
                # もし下部カメラで円が検出されたら
                if len(circles) > 0:
                    # デバッグ用に円を描画
                    [cv2.circle(lcam_ball_blur,(int(c[0][0]),int(c[0][1])),int(c[1]),(0,255,0),2) for c in circles]
                    # 返り値の更新
                    items = len(circles)
                    target = circles.index(max(circles, key=lambda x:x[1]))
                    (paddy_rice_x,paddy_rice_y,paddy_rice_z) = image_to_robot_coordinate_transformation(lcam_params,int(circles[target][0][0]),int(circles[target][0][1]),calc_distance(circles[target][1],PADDY_RICE_RADIUS))
                    is_obtainable = (paddy_rice_x-OBTAINABE_AREA_CENTER_X)**2 + (paddy_rice_y-OBTAINABE_AREA_CENTER_Y)**2 < OBTAINABE_AREA_RADIUS**2
                # もし下部カメラで円が検出されなければ
                else:
                    # 上部カメラで円を検出する
                    circles = find_circle_contours(ucam_close,UPPER_MIN_CONTOUR_AREA_THRESHOLD,UPPER_CIRCULARITY_THRESHOLD)
                    
                    # 上部カメラはカメラ画像の下半分だけ見る
                    circles = [[(c[0][0],c[0][1]), c[1]] for c in circles if c[0][1] > FRAME_HEIGHT/2]
                    
                    # もし上部カメラで円が検出されたら
                    if len(circles) > 0:
                        # デバッグ用に円を描画
                        [cv2.circle(ucam_ball_blur,(int(c[0][0]),int(c[0][1])),int(c[1]),(0,255,0),2) for c in circles]
                        # 返り値の更新
                        items = len(circles)
                        target = circles.index(max(circles, key=lambda x:x[1]))
                        (paddy_rice_x,paddy_rice_y,paddy_rice_z) = image_to_robot_coordinate_transformation(ucam_params,int(circles[target][0][0]),int(circles[target][0][1]),calc_distance(circles[target][1],PADDY_RICE_RADIUS))
                        is_obtainable = (paddy_rice_x-OBTAINABE_AREA_CENTER_X)**2 + (paddy_rice_y-OBTAINABE_AREA_CENTER_Y)**2 < OBTAINABE_AREA_RADIUS**2
                        
                # 画像のタイプを揃える
                ucam_close = cv2.cvtColor(ucam_close,cv2.COLOR_GRAY2BGR)
                lcam_close = cv2.cvtColor(lcam_close,cv2.COLOR_GRAY2BGR)
                ball_show_frame = np.vstack((np.hstack((ucam_ball_blur,ucam_close)),np.hstack((lcam_ball_blur,lcam_close))))
                
                # 検出したボールの座標をキューに送信 (xは水平，yは奥行方向)
                output_data = (items, paddy_rice_x, paddy_rice_y, paddy_rice_z, is_obtainable)
                #q_out.put((ball_show_frame, OUTPUT_ID.BALL, output_data))
                self.ball_camera_out = output_data
                ###ボール検出ここまで###
            except KeyboardInterrupt:
                break
            
    # 後方カメラでサイロを監視
    def detecting_rear(self,ucam_params,lcam_params,rcam_params,q_ucam,q_lcam,q_rcam,q_out):
        while True:
            try:
                frame = q_rcam.get()
                results = self.model.predict(frame, imgsz=320, conf=0.5, verbose=False)
                annotated_frame = results[0].plot()
                names = results[0].names
                classes = results[0].boxes.cls
                boxes = results[0].boxes
                x1, y1, x2, y2 = [0, 0, FRAME_WIDTH, FRAME_HEIGHT]
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
                        (target_silo_x,target_silo_y,target_silo_z) = image_to_robot_coordinate_transformation(rcam_params,w,h,dis)
                        maximum_my_ball_in_silo_count = my_ball_in_silo_counter
                
                # キューに送信
                output_data = (target_silo_x,target_silo_y,target_silo_z)
                #q_out.put((annotated_frame, OUTPUT_ID.SILO, output_data))
                self.silo_camera_out = output_data
            except KeyboardInterrupt:
                break
