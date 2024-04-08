import numpy as np
import cv2
from src import UpperCamera, LowerCamera, RearCamera, MainProcess, THREAD_ID

if __name__ == "__main__":
    #ncnn_model_path = 'models/20240109best_ncnn_model'
    model_path = 'models/20240109best.pt'
    
    # カメラのクラス
    ucam = UpperCamera()
    lcam = LowerCamera()
    rcam = RearCamera()
    
    # メインプロセスを実行するクラス
    mainprocess = MainProcess(model_path,ucam,lcam,rcam)
    
    # マルチスレッドの実行
    mainprocess.thread_start()
    
    while True:
        try:
            frame, id, output_data = mainprocess.q_frames_list[-1].get()
            
            cv2.imshow(f'frame{id}', frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if id==THREAD_ID.UPPER:   #UpperCamera
                items,x,y,z,is_obtainable = output_data
                print(f"\n{id=}, {items=}, {x=}, {y=}, {z=}, {is_obtainable=}")
            elif id==THREAD_ID.LOWER:   #LowerCamera
                items,x,y,z,is_obtainable = output_data
                print(f"\n{id=}, {items=}, {x=}, {y=}, {z=}, {is_obtainable=}")
            elif id==THREAD_ID.REAR:   #RearCamera
                print(f"\n{id=}")
            
            
        except KeyboardInterrupt:
            break
    mainprocess.finish()
    
    cv2.destroyAllWindows()
