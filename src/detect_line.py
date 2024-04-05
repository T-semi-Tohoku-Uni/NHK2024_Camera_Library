import cv2
import numpy as np

# ライン検出の際にロボットの中心からどれだけ奥行方向先の点を見ているのか[mm]
LINE_DETECTION_POINT = 300

def line_detector(id, q_frames, q_results):
    while True:
        try:
            # カメラから画像を読み込む
            frame = q_frames.get()

            # ライン検出
            

            # 奥行方向のラインがあるかどうか：bool
            # 右方向のラインがあるかどうか：bool
            # 左方向のラインがあるかどうか：bool
            # 奥行方向のラインの、水平方向のずれを出力(ロボットの中心から前方向300mmくらい)
            output_data = (forward, right, left, diff_x)
            # キューに結果を入れる
            q_results.put((show_frame, id, output_data))

        except KeyboardInterrupt:
                break