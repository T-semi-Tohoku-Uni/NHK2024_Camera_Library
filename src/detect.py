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

# 輝度に対して大津の二値化を適用する時の輝度値の間隔を表す配列
LUMINANCE_BINS=[i for i in range(0,255,1)]


# 籾の半径[mm]
PADDY_RICE_RADIUS = 100.0

# サイロの高さ[mm]
SILO_HEIGHT = 425.0

# 検出可能最大距離[mm]
DETECTABLE_MAX_DIS = 10000.0

# 検出した輪郭の最小面積(手前側のカメラ)[pxl]
MIN_CONTOUR_AREA_THRESHOLD = 2000

# 円形度の閾値
CIRCULARITY_THRESHOLD=0.5

# ロボット座標におけるアームのファンで吸い込めるエリアの中心と半径[mm]
OBTAINABE_AREA_CENTER_X = 0
OBTAINABE_AREA_CENTER_Y = 550
OBTAINABE_AREA_RADIUS = 80

# カメラからラインの検出点までの距離[mm]
LINE_DETECTION_POINT_TO_CAMERA_DISTANCE = 450

# 画像端から何ピクセル分の点までを、画像端から伸びてる線分とみなすか
LINE_MARGIN = 10

# 縦線、横線の角度の閾値[rad]
LINE_SLOPE_THRESHOLD = 0.26

class AREA_STATE(Enum):
    AREA_LINE = 0
    AREA_STORAGE= 1
    
class OUTPUT_ID(Enum):
    BALL = 0
    SILO = 1
    LINE = 2

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

def coordinate_transformation(params,w, h, dis):
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
    (focal_length,pos_x,pos_y,pos_z,theta_x,theta_y,theta_z) = params
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

# 大津の手法をdepthに適用
def threshold_otsu(hist, min_value=0, max_value=10):

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

def find_circle_contours(mask_img):
    """
    マスク画像から円の輪郭を探す
    
    Parameters
    ----------
    mask_img : numpy.array
        マスク処理をした画像
    
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
        contours = list(filter(lambda x: cv2.contourArea(x) > MIN_CONTOUR_AREA_THRESHOLD, contours))
        
        # 最小外接円を求める
        circles = [cv2.minEnclosingCircle(cnt) for cnt in contours if calc_circularity(cnt)>CIRCULARITY_THRESHOLD]
    return circles

class DetectObj:
    def __init__(self,model_path):
        # YOLOv8 modelのロード
        # self.model = YOLO(ncnn_model_path, task='detect')
        self.model = YOLO(model_path)
        
        # maskの値を設定する
        self.blue_lower_mask = np.array([135, 50, 50])
        self.blue_upper_mask = np.array([160, 255, 255])
        self.purple_lower_mask = np.array([165,50,50])
        self.purple_upper_mask = np.array([230,255,255])
        self.red_lower_mask_1 = np.array([0,50,50])
        self.red_upper_mask_1 = np.array([10,255,255])
        self.red_lower_mask_2 = np.array([230,50,50])
        self.red_upper_mask_2 = np.array([255,255,255])
        
        # fast line detector
        self.fld = cv2.ximgproc.createFastLineDetector(length_threshold=10,distance_threshold=1.41421356,canny_th1=200.0,canny_th2=50.0,canny_aperture_size=3,do_merge=False)
        
        # 前方のライン検出とボール検出の切り替えのために現在いるエリアを保持
        self.current_state = AREA_STATE.AREA_LINE
     
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
    
    # ボール検出（閾値）とライン検出をself.current_stateで切り替える
    def detecting_ball_or_line(self,ucam_params,lcam_params,rcam_params,q_ucam,q_lcam,q_rcam,q_out):
        while True:
            try:
                if self.current_state == AREA_STATE.AREA_LINE:
                    # 奥行方向のラインがあるかどうか：bool
                    forward = False
                    # 右方向のラインがあるかどうか：bool
                    right = False
                    # 左方向のラインがあるかどうか：bool
                    left = False
                    # 奥行方向のラインの、水平方向のずれを出力(ロボットの中心から前方向300mmくらい)
                    diff_x = 0.0
                    
                    # カメラから画像を読み込む
                    frame = q_lcam.get()
                    
                    # BGRのBを抽出
                    binary = frame[:,:,0]
                    
                    # 出力画像にガウシアンフィルタを適用する。
                    blur = cv2.GaussianBlur(binary, ksize=(7,7),sigmaX=0)
                    
                    # ライン検出
                    lines = self.fld.detect(blur)

                    # image for debug
                    all_lines = self.fld.drawSegments(frame,lines)
                    right_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3),dtype=np.uint8)
                    left_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3),dtype=np.uint8)
                    forward_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3),dtype=np.uint8)

                    if lines is not None:
                        # 画像の右端に点がある線分のリスト
                        right_list = [line for line in lines if line[0][0]<LINE_MARGIN or line[0][2]<LINE_MARGIN]
                        # 横線かどうかの判定
                        right_list = [line for line in right_list if abs(np.arcsin(abs(line[0][1]-line[0][3])/np.sqrt((abs(line[0][0]-line[0][2])**2+abs(line[0][1]-line[0][3])**2))))<LINE_SLOPE_THRESHOLD]
                        right_list = [line.astype(int) for line in right_list]
                        [cv2.line(right_frame,(p[0][0],p[0][1]),(p[0][2],p[0][3]),(0,255,0),3) for p in right_list]
                        right = True if len(right_list)>=2 else False

                        # 画像の左端に点がある線分のリスト
                        left_list = [line for line in lines if line[0][0]>FRAME_WIDTH-LINE_MARGIN or line[0][2]>FRAME_WIDTH-LINE_MARGIN]
                        # 横線かどうかの判定
                        left_list = [line for line in left_list if abs(np.arcsin(abs(line[0][1]-line[0][3])/np.sqrt((abs(line[0][0]-line[0][2])**2+abs(line[0][1]-line[0][3])**2))))<LINE_SLOPE_THRESHOLD]
                        left_list = [line.astype(int) for line in left_list]
                        [cv2.line(left_frame,(p[0][0],p[0][1]),(p[0][2],p[0][3]),(0,255,0),3) for p in left_list]
                        left = True if len(left_list)>=2 else False

                        # 画像の下端に点がある線分のリスト
                        forward_list = [line for line in lines if line[0][1]>FRAME_HEIGHT-LINE_MARGIN or line[0][3]>FRAME_HEIGHT-LINE_MARGIN]
                        # 縦線かどうかの判定
                        forward_list = [line for line in forward_list if abs(np.pi/2-np.arccos(abs(line[0][0]-line[0][2])/np.sqrt((abs(line[0][0]-line[0][2])**2+abs(line[0][1]-line[0][3])**2))))<LINE_SLOPE_THRESHOLD]
                        forward_list = [line.astype(int) for line in forward_list]
                        [cv2.line(forward_frame,(p[0][0],p[0][1]),(p[0][2],p[0][3]),(0,255,0),3) for p in forward_list]
                        forward = True if len(forward_list)>=2 else False

                        # 縦線の下の点に対応するxの値とFRAME_WIDTH/2の差をとったリスト
                        forward_x_list = [abs(p[0][0]-FRAME_WIDTH/2) if p[0][1]>p[0][3] else abs(p[0][2]-FRAME_WIDTH/2) for p in forward_list]
                        forward_x_list.sort()
                        if len(forward_x_list)>=2:
                            # 画像の中心に近い2本の線分のx座標の平均
                            diff_x = (forward_x_list[0]+forward_x_list[1])/2

                    show_frame = np.vstack((all_lines,right_frame,left_frame,forward_frame))

                    x,y,z = coordinate_transformation(lcam_params,diff_x,FRAME_HEIGHT-LINE_MARGIN,LINE_DETECTION_POINT_TO_CAMERA_DISTANCE)
                    
                    output_data = (forward, right, left, x)
                    # キューに結果を入れる
                    q_out.put((show_frame, OUTPUT_ID.LINE, output_data))
                    
                    
                elif self.current_state == AREA_STATE.AREA_STORAGE:
                    ucam_frame = q_ucam.get()
                    lcam_frame = q_lcam.get()

                    # 出力画像にガウシアンフィルタを適用する。
                    ucam_frame = cv2.GaussianBlur(ucam_frame, ksize=(7,7),sigmaX=0)
                    lcam_frame = cv2.GaussianBlur(lcam_frame, ksize=(7,7),sigmaX=0)

                    # カメラ画像をHSVに変換
                    ucam_hsv = cv2.cvtColor(ucam_frame, cv2.COLOR_BGR2HSV_FULL)
                    lcam_hsv = cv2.cvtColor(lcam_frame, cv2.COLOR_BGR2HSV_FULL)

                    # 閾値でmasking処理
                    ucam_mask = cv2.inRange(ucam_hsv,self.blue_lower_mask,self.blue_upper_mask)
                    lcam_mask = cv2.inRange(lcam_hsv,self.blue_lower_mask,self.blue_upper_mask)
                    
                    items = 0
                    paddy_rice_x = 0
                    paddy_rice_y = 0
                    paddy_rice_z = DETECTABLE_MAX_DIS
                    is_obtainable = False
                    
                    # 下部カメラで円を検出する
                    circles = find_circle_contours(lcam_mask)
                    # もし下部カメラで円が検出されたら
                    if len(circles) > 0:
                        # デバッグ用に円を描画
                        [cv2.circle(lcam_frame,(int(c[0][0]),int(c[0][1])),int(c[1]),(0,255,0),2) for c in circles]
                        items = len(circles)
                        target = circles.index(max(circles, key=lambda x:x[1]))
                        (paddy_rice_x,paddy_rice_y,paddy_rice_z) = coordinate_transformation(lcam_params,int(circles[target][0][0]),int(circles[target][0][1]),calc_distance(circles[target][1],PADDY_RICE_RADIUS))
                        is_obtainable = (paddy_rice_x-OBTAINABE_AREA_CENTER_X)**2 + (paddy_rice_y-OBTAINABE_AREA_CENTER_Y)**2 < OBTAINABE_AREA_RADIUS**2
                    # もし下部カメラで円が検出されなければ
                    else:
                        # 上部カメラで円を検出する
                        circles = find_circle_contours(ucam_mask)
                        # もし上部カメラで円が検出されたら
                        if len(circles) > 0:
                            # デバッグ用に円を描画
                            [cv2.circle(ucam_frame,(int(c[0][0]),int(c[0][1])),int(c[1]),(0,255,0),2) for c in circles]
                            items = len(circles)
                            target = circles.index(max(circles, key=lambda x:x[1]))
                            (paddy_rice_x,paddy_rice_y,paddy_rice_z) = coordinate_transformation(ucam_params,int(circles[target][0][0]),int(circles[target][0][1]),calc_distance(circles[target][1],PADDY_RICE_RADIUS))
                            is_obtainable = (paddy_rice_x-OBTAINABE_AREA_CENTER_X)**2 + (paddy_rice_y-OBTAINABE_AREA_CENTER_Y)**2 < OBTAINABE_AREA_RADIUS**2

                    # 画像のタイプを揃える
                    ucam_mask = cv2.cvtColor(ucam_mask,cv2.COLOR_GRAY2BGR)
                    lcam_mask = cv2.cvtColor(lcam_mask,cv2.COLOR_GRAY2BGR)
                    show_frame = np.hstack((ucam_frame,ucam_mask,lcam_frame,lcam_mask))
                    
                    # 検出したボールの座標をキューに送信 (xは水平，yは奥行方向)
                    output_data = (items, paddy_rice_x, paddy_rice_y, paddy_rice_z, is_obtainable)
                    q_out.put((show_frame, OUTPUT_ID.BALL, output_data))
                
            except KeyboardInterrupt:
                break
            
    # サイロの中の自分の籾の数を推論から求めてキューに入れる
    def inference_for_silo(self,ucam_params,lcam_params,rcam_params,q_ucam,q_lcam,q_rcam,q_out):
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
                            print(f"Unexpected {err=}, {type(err)=}")
                
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
                        (target_silo_x,target_silo_y,target_silo_z) = coordinate_transformation(rcam_params,w,h,dis)
                        maximum_my_ball_in_silo_count = my_ball_in_silo_counter
                
                # キューに送信
                output_data = (target_silo_x,target_silo_y,target_silo_z)
                q_out.put((annotated_frame, OUTPUT_ID.SILO, output_data))
                
            except KeyboardInterrupt:
                break
