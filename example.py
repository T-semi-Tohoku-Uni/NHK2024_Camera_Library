import numpy as np
import cv2
from src.capture_and_detect import MainProcess

if __name__ == "__main__":
    # model_path = 'models/20240109best.pt'
    blue_model_path = 'models/NHK2024_blue_ball_model/blue_ball_model.pt'
    red_model_path = 'models/NHK2024_red_ball_model/red_ball_model.pt'
    silo_model_path = 'models/NHK2024_silo_model/silo_model.pt'
    
    # メインプロセスを実行するクラス
    print("create main instance")
    mainprocess = MainProcess(
        ball_model_path=red_model_path, 
        silo_model_path=silo_model_path,
        show=True
    )
    
    # マルチスレッドの実行
    print("start thread")
    mainprocess.thread_start()
    print("complete start thread")
    cnt = 0
    while True:
        try:
            # if mainprocess.detector.show:   
            # frame, id = mainprocess.q_out.get() 
            # # cv2.imshow(f'{id}', frame)
            # key = cv2.waitKey(1)
            # if key == ord("q"):
            #     break
            
            _, _= mainprocess.q_out.get()
            continue
            
            # items,x,y,z,is_obtainable = mainprocess.update_ball_camera_out()
            # x,y,z = mainprocess.update_silo_camera_out()
            # forward, right, left, x, theta = mainprocess.update_line_camera_out()
            # print(f"\n{items=}, {x=}, {y=}, {z=}, {is_obtainable=}")
            # print(f"\n{x=}, {y=}, {z=}")
            # print(f"\n{forward=}, {right=}, {left=}, {x=}, {theta=}")
            
        except KeyboardInterrupt:
            break
    mainprocess.finish()
    
    cv2.destroyAllWindows()
