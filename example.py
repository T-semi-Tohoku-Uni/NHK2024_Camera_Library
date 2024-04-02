import numpy as np
import cv2
from src import UpperCamera, LowerCamera, RearCamera, MainProcess

if __name__ == "__main__":
    #ncnn_model_path = 'models/20240109best_ncnn_model'
    model_path = 'models/20240109best.pt'
    
    # カメラのクラス
    cam0 = UpperCamera(0)
    cam1 = LowerCamera(2)
    rs = RearCamera()
    
    # メインプロセスを実行するクラス
    mainprocess = MainProcess(model_path,cam0,cam1,rs)
    
    # マルチスレッドの実行
    mainprocess.thread_start()
    
    while True:
        try:
            frame, id, output_data = mainprocess.q_frames_list[-1].get()
            
            cv2.imshow(f'frame{id}', frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if id==0:   #UpperCamera
                items,x,y,z,is_obtainable = output_data
                print(f"\n{id=}, {items=}, {x=}, {y=}, {z=}, {is_obtainable=}")
            elif id==1:   #LowerCamera
                items,x,y,z,is_obtainable = output_data
                print(f"\n{id=}, {items=}, {x=}, {y=}, {z=}, {is_obtainable=}")
            elif id==2:   #Realsense
                print(f"\n{id=}")
            
            
        except KeyboardInterrupt:
            break
    mainprocess.finish()
    
    cv2.destroyAllWindows()
