import numpy as np
import cv2
from ultralytics import YOLO
import threading
import queue
import time
from enum import Enum
from .camera import UpperCamera,LowerCamera,RearCamera

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

# 大津の二値化を適用するデプスの間隔を表す配列
BINS=[i for i in range(RS_MIN_DISTANCE,RS_MAX_DISTANCE+RS_MIN_DISTANCE,DISTANCE_INTERVAL)]

# 籾の半径[mm]
PADDY_RICE_RADIUS = 100.0

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

class QUEUE_ID(Enum):
    UPPER_IN = 0
    LOWER_IN = 1
    REAR_IN = 2
    FRONT_OUT = 3
    REAR_OUT = 4

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
    (pos_x,pos_y,pos_z,theta_x,theta_y,theta_z) = params
    internal_param_inv = np.array([[0.0037037, 0, 0], [0, 0.0037037, 0], [0, 0, 1] ,[0, 0, 1/dis]])
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
        
        # クラス1とクラス2のdepthヒストグラム値(depth値ではない)の平均を計算
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

class MainProcess:
    def __init__(self, model_path, ucam, lcam, rcam):
        # YOLOv8 modelのロード
        # self.model = YOLO(ncnn_model_path, task='detect')
        self.model = YOLO(model_path)
 
        # キューの辞書の宣言(上部カメラ画像のキュー，下部カメラ画像のキュー，Realsense画像のキュー，ロボット前の処理した画像のキュー，ロボット後ろの処理した画像のキュー)
        self.q_upper_in = queue.Queue(maxsize=1)
        self.q_lower_in = queue.Queue(maxsize=1)
        self.q_rear_in = queue.Queue(maxsize=1)
        self.q_front_out = queue.Queue(maxsize=1)
        self.q_rear_out = queue.Queue(maxsize=1)
        
        self.ucam = ucam
        self.lcam = lcam
        self.rcam = rcam

        # カメラ毎の処理数のdictionary
        self.counters = {QUEUE_ID.FRONT_OUT:0, QUEUE_ID.REAR_OUT:0}
        
        # 処理の開始時間
        self.start_time = 0.0
        
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
          
    # マスク処理によりボールをファンで吸い込めるかどうか判定してキューに入れる
    def masking_for_fan_obtainable_judgement(self, ucam_params, lcam_params, q_ucam, q_lcam, q_results):
        while True:
            try:
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
                    (paddy_rice_x,paddy_rice_y,paddy_rice_z) = coordinate_transformation(lcam_params,int(circles[target][0][0]),int(circles[target][0][1]),calc_distance(circles[target][1]))
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
                        (paddy_rice_x,paddy_rice_y,paddy_rice_z) = coordinate_transformation(ucam_params,int(circles[target][0][0]),int(circles[target][0][1]),calc_distance(circles[target][1]))
                        is_obtainable = (paddy_rice_x-OBTAINABE_AREA_CENTER_X)**2 + (paddy_rice_y-OBTAINABE_AREA_CENTER_Y)**2 < OBTAINABE_AREA_RADIUS**2


                # 画像のタイプを揃える
                ucam_mask = cv2.cvtColor(ucam_mask,cv2.COLOR_GRAY2BGR)
                lcam_mask = cv2.cvtColor(lcam_mask,cv2.COLOR_GRAY2BGR)
                show_frame = np.hstack((ucam_frame,ucam_mask,lcam_frame,lcam_mask))
                
                # 処理数に加算
                self.counters[QUEUE_ID.FRONT_OUT] += 1
                
                # 検出したボールの座標をキューに送信 (xは水平，yは奥行方向)
                output_data = (items, paddy_rice_x, paddy_rice_y, paddy_rice_z, is_obtainable)
                q_results.put((show_frame, QUEUE_ID.FRONT_OUT, output_data))
                
            except KeyboardInterrupt:
                break
    
    # 推論によりボールをファンで吸い込めるかどうか判定してキューに入れる
    def inference_for_fan_obtainable_judgement(self, ucam_params, lcam_params, q_ucam, q_lcam, q_results):
        while True:
            try:
                # 初期化
                paddy_rice_x = 0
                paddy_rice_y = 0
                paddy_rice_z = DETECTABLE_MAX_DIS

                # 下部カメラから推論
                lcam_frame = q_lcam.get()
                lcam_results = self.model.predict(lcam_frame, imgsz=320, conf=0.5, verbose=False)
                lcam_annotated_frame = lcam_results[0].plot()
                names = lcam_results[0].names
                classes = lcam_results[0].boxes.cls
                boxes = lcam_results[0].boxes
                # 上部カメラから推論
                ucam_frame = q_ucam.get()
                ucam_results = self.model.predict(ucam_frame, imgsz=320, conf=0.5, verbose=False)
                ucam_annotated_frame = ucam_results[0].plot()
                
                # もし、下部カメラで検出できていれば
                if boxes.dim() > 0:
                    for box, cls in zip(boxes, classes):
                        name = names[int(cls)]
                        if(name == "blueball"):
                            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
                            # 長方形の長辺を籾の半径とする
                            r = max(abs(x1-x2)/2, abs(y1-y2)/2)
                            z = calc_distance(r)
                            # 籾が複数ある場合は最も近いものの座標を返す
                            if z < paddy_rice_z:
                                (paddy_rice_x,paddy_rice_y,paddy_rice_z) = coordinate_transformation(lcam_params,int((x1+x2)/2),int((y1+y2)/2),z)
                    is_obtainable = (paddy_rice_x-OBTAINABE_AREA_CENTER_X)**2 + (paddy_rice_y-OBTAINABE_AREA_CENTER_Y)**2 < OBTAINABE_AREA_RADIUS**2
                # そうでなければ
                else:
                    names = ucam_results[0].names
                    classes = ucam_results[0].boxes.cls
                    boxes = ucam_results[0].boxes
                    for box, cls in zip(boxes, classes):
                        name = names[int(cls)]
                        if(name == "blueball"):
                            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
                            # 長方形の長辺を籾の半径とする
                            r = max(abs(x1-x2)/2, abs(y1-y2)/2)
                            z = calc_distance(r)
                            # 籾が複数ある場合は最も近いものの座標を返す
                            if z < paddy_rice_z:
                                (paddy_rice_x,paddy_rice_y,paddy_rice_z) = coordinate_transformation(ucam_params,int((x1+x2)/2),int((y1+y2)/2),z)
                    is_obtainable = (paddy_rice_x-OBTAINABE_AREA_CENTER_X)**2 + (paddy_rice_y-OBTAINABE_AREA_CENTER_Y)**2 < OBTAINABE_AREA_RADIUS**2


                # 処理数に加算
                self.counters[QUEUE_ID.FRONT_OUT] += 1

                show_frame = np.hstack((ucam_annotated_frame,lcam_annotated_frame))
                # 検出したボールの座標をキューに送信 (xは水平，yは奥行方向)
                output_data = (len(boxes), paddy_rice_x, paddy_rice_y, paddy_rice_z, is_obtainable)
                q_results.put((show_frame, QUEUE_ID.FRONT_OUT, output_data))
                
            except KeyboardInterrupt:
                break
            
    # サイロの中の自分の籾の数を推論から求めてキューに入れる
    def inference_for_silo(self, q_frames, q_results):
        while True:
            try:
                frame = q_frames.get()
                results = self.model.predict(frame, imgsz=320, conf=0.5, verbose=False)
                annotated_frame = results[0].plot()
                names = results[0].names
                classes = results[0].boxes.cls
                boxes = results[0].boxes
                x1, y1, x2, y2 = [0, 0, FRAME_WIDTH, FRAME_HEIGHT]
                my_ball_in_silo_counter = 0
        
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
                    if(name == "silo"):
                        for bxyz in ball_xyz:
                            if(x1<bxyz[0] and bxyz[2]<x2 and bxyz[3]<y2 and abs((x2-x1)-(bxyz[2]-bxyz[0]))<BALL_IN_SILO_THRESHOLD):
                                my_ball_in_silo_counter += 1
                        cv2.putText(annotated_frame,f"{my_ball_in_silo_counter} in silo",(x1,y1+15),cv2.FONT_HERSHEY_PLAIN,1.0,(0,255,0),thickness=2)
                
                # 処理数に加算
                self.counters[QUEUE_ID.REAR_OUT] += 1
            
                # 画像を送信
                output_data = ()
                q_results.put((annotated_frame, QUEUE_ID.REAR_OUT, output_data))
                
            except KeyboardInterrupt:
                break

    # サイロの中の自分の籾の数を推論とデプス情報から求めてキューに入れる
    def inference_for_silo_with_depth(self, q_frames, q_results):
        while True:
            try:
                (color, depth) = q_frames.get()
                results = self.model.predict(color, imgsz=320, conf=0.5, verbose=False)
                annotated_frame = results[0].plot()
                names = results[0].names
                classes = results[0].boxes.cls
                boxes = results[0].boxes
                x1, y1, x2, y2 = [0, 0, FRAME_WIDTH, FRAME_HEIGHT]
                my_ball_in_silo_counter = 0
        
                # ballのx1,y1,x2,y2,depthの平均を入れる
                ball_xyz = np.empty((0,5), int)
                
                for box, cls in zip(boxes, classes):
                    name = names[int(cls)]
                    x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
                    if(name == "blueball"):
                        try:
                            ball_area = depth[y1:y2,x1:x2].copy()
                            ball_xyz = np.append(ball_xyz, [[x1,y1,x2,y2,np.mean(ball_area[ball_area>RS_MIN_DISTANCE])]],axis=0)
                        except Exception as err:
                            print(f"Unexpected {err=}, {type(err)=}")
                
                for box, cls in zip(boxes, classes):
                    name = names[int(cls)]
                    x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
                    if(name == "silo"):
                        hist, _ = np.histogram(depth[y1:y2,x1:x2], BINS)
                        th = threshold_otsu(hist, 0, len(hist))
                        # 背景のdepthを0にする
                        depth[y1:y2,x1:x2][th*DISTANCE_INTERVAL<depth[y1:y2,x1:x2]]=0
                        for bxyz in ball_xyz:
                            if(x1<bxyz[0] and bxyz[2]<x2):
                                # depth_imageからボールの領域を削除(min_distance以下の値にする)
                                depth[int(bxyz[1]):int(bxyz[3]),int(bxyz[0]):int(bxyz[2])] = 0
                                ball_z = int(bxyz[4])
                                silo_z = np.mean(depth[y1:y2,x1:x2][depth[y1:y2,x1:x2]>RS_MIN_DISTANCE])
                                #cv2.putText(annotated_image,f"ball_z:{ball_z}",(int(bxyz[0]),int(bxyz[1]-10)),cv2.FONT_HERSHEY_DUPLEX,1.0,(255,255,0))
                                #cv2.putText(annotated_image,f"silo_z:{silo_z}",(int(x1),int(y1)-10),cv2.FONT_HERSHEY_DUPLEX,1.0,(255,255,0))
                                if (abs(ball_z-silo_z)<BALL_IN_SILO_RANGE):
                                    my_ball_in_silo_counter += 1
                        cv2.putText(annotated_frame,f"{my_ball_in_silo_counter} in silo",(x1,y1+15),cv2.FONT_HERSHEY_PLAIN,1.0,(0,255,0),thickness=2)
                
                # 処理数に加算
                self.counters[QUEUE_ID.REAR_OUT] += 1
            
                # 画像を送信
                output_data = ()
                q_results.put((annotated_frame, QUEUE_ID.REAR_OUT, output_data))
                
            except KeyboardInterrupt:
                break

    # カメラからの画像取得と画像処理、推論(デプス無し)をスレッドごとに分けて実行      
    def thread_start(self):
        thread_upper_capturing = threading.Thread(target=self.capturing, args=(self.q_upper_in,self.ucam), daemon=True)
        thread_lower_capturing = threading.Thread(target=self.capturing, args=(self.q_lower_in,self.lcam), daemon=True)
        thread_front_detecting = threading.Thread(target=self.masking_for_fan_obtainable_judgement, args=(self.ucam.params,self.lcam.params,self.q_upper_in,self.q_lower_in,self.q_front_out),daemon=True)
        thread_rear_capturing = threading.Thread(target=self.capturing, args=(self.q_rear_in,self.rcam),daemon=True)
        thread_rear_detecting = threading.Thread(target=self.inference_for_silo, args=(self.q_rear_in,self.q_rear_out),daemon=True)
        
        thread_upper_capturing.start()
        thread_lower_capturing.start()
        thread_front_detecting.start()
        thread_rear_capturing.start()
        thread_rear_detecting.start()
        self.start_time = time.time()

    # キューを空にする
    def finish(self):
        end_time = time.time()
        for cam in (self.ucam,self.lcam,self.rcam):
            self.cameras_dict[thread_id].release()
            print(f"{thread_id=} : {self.counters[thread_id.value] / (end_time - self.start_time)} fps")
        
        for q in self.q_frames_list:
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
                
        print("All Queue Empty")