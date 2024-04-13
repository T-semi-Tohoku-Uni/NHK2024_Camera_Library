import numpy as np
import cv2
from src import MainProcess,OUTPUT_ID

if __name__ == "__main__":
    #ncnn_model_path = 'models/20240109best_ncnn_model'
    model_path = 'models/20240109best.pt'
    
    # メインプロセスを実行するクラス
    mainprocess = MainProcess(model_path)
    
    # マルチスレッドの実行
    mainprocess.thread_start()
    
    while True:
        try:
            frame, id, output_data = mainprocess.q_out.get()
            cv2.imshow(f'{id}', frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            
            if id == OUTPUT_ID.BALL:
                items,x,y,z,is_obtainable = output_data
                print(f"\n{id=}, {items=}, {x=}, {y=}, {z=}, {is_obtainable=}")
            elif id == OUTPUT_ID.SILO:
                x,y,z = output_data
                print(f"\n{id=}, {x=}, {y=}, {z=}")
            
        except KeyboardInterrupt:
            break
    mainprocess.finish()
    
    cv2.destroyAllWindows()
