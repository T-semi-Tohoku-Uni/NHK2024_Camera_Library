import numpy as np
import cv2
from src import MainProcess,OUTPUT_ID

if __name__ == "__main__":
    model_path = 'models/20240109best.pt'
    
    # メインプロセスを実行するクラス
    mainprocess = MainProcess(model_path)
    
    # マルチスレッドの実行
    mainprocess.thread_start()
    
    while True:
        try:
            if mainprocess.detector.show:    
                frame, id = mainprocess.q_out.get()
                cv2.imshow(f'{id.value}', frame)
                key = cv2.waitKey(1)
                if key == ord("q"):
                    break
            
            items,x,y,z,is_obtainable = mainprocess.update_ball_camera_out()
            x,y,z = mainprocess.update_silo_camera_out()
            forward, right, left, x, theta = mainprocess.update_line_camera_out()
            print(f"\n{items=}, {x=}, {y=}, {z=}, {is_obtainable=}")
            print(f"\n{x=}, {y=}, {z=}")
            print(f"\n{forward=}, {right=}, {left=}, {x=}, {theta=}")
            
        except KeyboardInterrupt:
            break
    mainprocess.finish()
    
    cv2.destroyAllWindows()
