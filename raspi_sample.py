import cv2
from src import UpperCamera, MainProcess
import time

if __name__ == "__main__":
    model_path = 'models/20240109best.pt'
    
    # カメラのクラス
    cam = UpperCamera(0)
    
    
    # メインプロセスを実行するクラス
    mainprocess = MainProcess(model_path, cam, cam, cam)
    
    # マルチスレッドの実行
    mainprocess.thread_start()
    
    start_time = time.time()
    while True:
        try:
            _, id, output_data = mainprocess.q_frames_list[-1].get()
            items,x,y,z,is_obtainable = output_data
            #_, id, items, x, y, z, is_obtainable = (1,1,1,1,True)
            print(f"\n{id=}, {items=}, {x=}, {y=}, {z=}, {is_obtainable=}")
            """
            cv2.drawMarker(frame, (160,128), (0,0,255))
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            """
            
        except KeyboardInterrupt:
            break
    
    mainprocess.finish()
    
    cv2.destroyAllWindows()
