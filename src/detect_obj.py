import numpy as np
import cv2
from ultralytics import YOLO
import threading
import queue
import time
from enum import Enum
from .camera import UpperCamera,LowerCamera,RearCamera

NUMBER_OF_CAMERAS = 3

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

BINS=[i for i in range(RS_MIN_DISTANCE,RS_MAX_DISTANCE+RS_MIN_DISTANCE,DISTANCE_INTERVAL)]

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

class THREAD_ID(Enum):
    UPPER_THREAD_ID = 0
    LOWER_THREAD_ID = 1
    REAR_THREAD_ID = 2

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

class MainProcess:
    def __init__(self, model_path, ucam,lcam,rs):
        # YOLOv8 modelのロード
        # self.model = YOLO(ncnn_model_path, task='detect')
        self.model = YOLO(model_path)
 
        # カメラ(webカメラ、Realsense)のクラスのタプル       
        self.cameras = [ucam,lcam,rs]
        
        # キューの宣言([上部カメラ画像のキュー，下部カメラ画像のキュー，Realsense画像のキュー，処理した画像のキュー])
        self.q_frames_list = []
        [self.q_frames_list.append(queue.Queue(maxsize=1)) for i in range(NUMBER_OF_CAMERAS+1)]
        
        # カメラ毎の処理数のリスト
        self.counters = [0,0,0]
        
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
                ret, frame = cap.read()
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
    def masking_for_fan_obtainable_judgement(self, id, q_frames, q_results):
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
                
                # 処理数に加算
                self.counters[id] += 1
                
                # 検出したボールの座標をキューに送信 (xは水平，yは奥行方向)
                output_data = (items, paddy_rice_x, paddy_rice_y, paddy_rice_z, is_obtainable)
                q_results.put((show_frame, id, output_data))
                
            except KeyboardInterrupt:
                break
    
    # 推論によりボールをファンで吸い込めるかどうか判定してキューに入れる
    def inference_for_fan_obtainable_judgement(self, id, q_frames, q_results):
        while True:
            try:
                frame = q_frames.get()
                results = self.model.predict(frame, imgsz=320, conf=0.5, verbose=False)
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
            
                # 処理数に加算
                self.counters[id] += 1
            
                # 検出したボールの座標をキューに送信 (xは水平，yは奥行方向)
                output_data = (len(boxes), paddy_rice_x, paddy_rice_y, paddy_rice_z, is_obtainable)
                q_results.put((annotated_frame, id, output_data))
                
            except KeyboardInterrupt:
                break
            
    # サイロの中の自分の籾の数を推論から求めてキューに入れる
    def inference_for_silo(self, id, q_frames, q_results):
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
                self.counters[id] += 1
            
                # 画像を送信
                output_data = ()
                q_results.put((annotated_frame, id, output_data))
                
            except KeyboardInterrupt:
                break

    # サイロの中の自分の籾の数を推論とデプス情報から求めてキューに入れる
    def inference_for_silo_with_depth(self, id, q_frames, q_results):
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
                            # バウンディングボックスからボールの中心と半径を求め、その円内の深度の平均をappend
                            cx = int((x1+x2)/2)
                            cy = int((y1+y2)/2)
                            r = max(abs(x1-x2), abs(y1-y2))
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
                self.counters[id] += 1
            
                # 画像を送信
                output_data = ()
                q_results.put((annotated_frame, id, output_data))
                
            except KeyboardInterrupt:
                break

    
    
    # カメラからの画像取得と画像処理、推論をスレッドごとに分けて実行      
    def thread_start(self):
        for id,cam in enumerate(self.cameras):
            if type(cam) is UpperCamera:
                thread_capturing = threading.Thread(target=self.capturing, args=(self.q_frames_list[id], cam), daemon=True)
                thread_detecting = threading.Thread(target=self.masking_for_fan_obtainable_judgement, args=(id, self.q_frames_list[id], self.q_frames_list[-1]), daemon=True)
            elif type(cam) is LowerCamera:
                thread_capturing = threading.Thread(target=self.capturing, args=(self.q_frames_list[id], cam), daemon=True)
                thread_detecting = threading.Thread(target=self.masking_for_fan_obtainable_judgement, args=(id, self.q_frames_list[id], self.q_frames_list[-1]), daemon=True)
            elif type(cam) is RearCamera:
                thread_capturing = threading.Thread(target=self.capturing_with_depth, args=(self.q_frames_list[id], cam), daemon=True)
                thread_detecting = threading.Thread(target=self.inference_for_silo_with_depth, args=(id, self.q_frames_list[id], self.q_frames_list[-1]), daemon=True)
            else:
                print("Unexpected Camera Class")
                quit()
            
            thread_capturing.start()
            thread_detecting.start()
            
        self.start_time = time.time()
        
    # カメラからの画像取得と推論をスレッドごとに分けて実行
    def all_yolo_thread_start(self):
        for id,cam in enumerate(self.cameras):
            if type(cam) is UpperCamera:
                thread_capturing = threading.Thread(target=self.capturing, args=(self.q_frames_list[id], cam), daemon=True)
                thread_detecting = threading.Thread(target=self.inference_for_fan_obtainable_judgement, args=(id, self.q_frames_list[id], self.q_frames_list[-1]), daemon=True)
            elif type(cam) is LowerCamera:
                thread_capturing = threading.Thread(target=self.capturing, args=(self.q_frames_list[id], cam), daemon=True)
                thread_detecting = threading.Thread(target=self.inference_for_fan_obtainable_judgement, args=(id, self.q_frames_list[id], self.q_frames_list[-1]), daemon=True)
            elif type(cam) is RearCamera:
                thread_capturing = threading.Thread(target=self.capturing_with_depth, args=(self.q_frames_list[id], cam), daemon=True)
                thread_detecting = threading.Thread(target=self.inference_for_silo_with_depth, args=(id, self.q_frames_list[id], self.q_frames_list[-1]), daemon=True)
            else:
                print("Unexpected Camera Class")
                quit()
            
            thread_capturing.start()
            thread_detecting.start()
            
        self.start_time = time.time()
    
    # カメラからの画像取得と画像処理、推論（デプス情報なし）をスレッドごとに分けて実行
    def no_depth_thread_start(self):
        for id,cam in enumerate(self.cameras):
            if type(cam) is UpperCamera:
                thread_capturing = threading.Thread(target=self.capturing, args=(self.q_frames_list[id], cam), daemon=True)
                thread_detecting = threading.Thread(target=self.masking_for_fan_obtainable_judgement, args=(id, self.q_frames_list[id], self.q_frames_list[-1]), daemon=True)
            elif type(cam) is LowerCamera:
                thread_capturing = threading.Thread(target=self.capturing, args=(self.q_frames_list[id], cam), daemon=True)
                thread_detecting = threading.Thread(target=self.masking_for_fan_obtainable_judgement, args=(id, self.q_frames_list[id], self.q_frames_list[-1]), daemon=True)
            elif type(cam) is RearCamera:
                thread_capturing = threading.Thread(target=self.capturing, args=(self.q_frames_list[id], cam), daemon=True)
                thread_detecting = threading.Thread(target=self.inference_for_silo, args=(id, self.q_frames_list[id], self.q_frames_list[-1]), daemon=True)
            else:
                print("Unexpected Camera Class")
                quit()
            
            thread_capturing.start()
            thread_detecting.start()
            
        self.start_time = time.time()


    # キューを空にする
    def finish(self):
        end_time = time.time()
        for id,c in enumerate(self.cameras):
            c.release()
            print(f"{id=} : {self.counters[id] / (end_time - self.start_time)} fps")
        
        for q in self.q_frames_list:
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
                
        print("All Queue Empty")