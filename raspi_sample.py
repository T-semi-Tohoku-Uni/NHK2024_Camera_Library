import cv2
from src import UpperCamera,LowerCamera,RearCamera,MainProcess
import time

if __name__ == "__main__":
    model_path = 'models/20240109best.pt'
    
    # カメラのクラス
    cam0 = UpperCamera(0)
    cam1 = LowerCamera(2)
    rs = RearCamera()
    
    # メインプロセスを実行するクラス
    mainprocess = MainProcess(model_path, cam0, cam1, rs)
    
    # マルチスレッドの実行
    mainprocess.all_yolo_thread_start()
    
    start_time = time.time()
    while True:
        try:
            _, id, output_data = mainprocess.q_frames_list[-1].get()
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
