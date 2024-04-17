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
            _, id, output_data = mainprocess.q_out.get()

            
            if id == OUTPUT_ID.BALL:
                items,x,y,z,is_obtainable = output_data
                print(f"\n{id=}, {items=}, {x=}, {y=}, {z=}, {is_obtainable=}")
            elif id == OUTPUT_ID.SILO:
                x,y,z = output_data
                print(f"\n{id=}, {x=}, {y=}, {z=}")
            elif id == OUTPUT_ID.LINE:
                forward, right, left, x = output_data
                print(f"\n{id=}, {forward=}, {right=}, {left=}, {x=}")
            
        except KeyboardInterrupt:
            break
    mainprocess.finish()
    
    cv2.destroyAllWindows()
