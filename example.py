import cv2
from src import FrontCamera, MainProcess
import time

if __name__ == "__main__":
    #ncnn_model_path = 'models/20240109best_ncnn_model'
    model_path = 'models/20240109best.pt'
    
    # カメラのクラス
    cam = FrontCamera(0)
    
    # メインプロセスを実行するクラス
    mainprocess = MainProcess(model_path)

    # 処理数
    count = 0
    
    # マルチスレッドの実行
    mainprocess.thread_start(cam)
    
    start_time = time.time()
    while True:
        try:
            items, x, y, z, is_obtainable = mainprocess.q_results.get()
            #items, x, y, z, is_obtainable = (1,1,1,1,True)
            print(f"\nitems:{items}, x:{x}, y:{y}, z:{z}, is_obtainable:{is_obtainable}")
            count += 1
            """
            cv2.drawMarker(annotated_frame, (160,128), (0,0,255))
            cv2.imshow('image', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            """
        except KeyboardInterrupt:
            break
    end_time = time.time()
    print(f"count / time : {count / (end_time - start_time)}")
    cv2.destroyAllWindows()