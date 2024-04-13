import cv2
import numpy as np
import queue

# ライン検出の際にロボットの中心からどれだけ奥行方向先の点を見ているのか[mm]
LINE_DETECTION_POINT = 300

class DetectLine:
    def __init__(self) -> None:
         pass
     
    def line_detector(self, lcam_params, q_lcam, q_results):
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS,30)
        fld = cv2.ximgproc.createFastLineDetector(length_threshold=10,distance_threshold=1.414213562,canny_th1=50.0,canny_th2=50.0,canny_aperture_size=3,do_merge=True)
        while True:
            try:
                # カメラから画像を読み込む
                #frame = q_lcam.get()
                ret, frame = cap.read()
                # グレースケール化
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # 出力画像にガウシアンフィルタを適用する。
                blur = cv2.GaussianBlur(gray, ksize=(7,7),sigmaX=0)
                
                # ライン検出
                lines = fld.detect(blur)
                
                if lines is not None:
                    for line in lines:
                        line = line.astype(int)
                        cv2.line(gray,(line[0][0],line[0][1]),(line[0][2],line[0][3]),(255,0,0), 5)
                        
                show = np.hstack((gray,blur))
                cv2.imshow(f'output', show)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                

                # 奥行方向のラインがあるかどうか：bool
                # 右方向のラインがあるかどうか：bool
                # 左方向のラインがあるかどうか：bool
                # 奥行方向のラインの、水平方向のずれを出力(ロボットの中心から前方向300mmくらい)
                #output_data = (forward, right, left, diff_x)
                # キューに結果を入れる
                #q_results.put((show_frame, OUTPUT_ID.LINE, output_data))
                
            except KeyboardInterrupt:
                    break
                
if __name__ == "__main__":
    detector = DetectLine()
    detector.line_detector([],queue.Queue(),queue.Queue())