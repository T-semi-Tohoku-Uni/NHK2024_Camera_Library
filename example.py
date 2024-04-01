import cv2
from src import FrontCamera, MainProcess
import time

if __name__ == "__main__":
    #ncnn_model_path = 'models/20240109best_ncnn_model'
    model_path = 'models/20240109best.pt'
    
    # カメラのクラス
    cam1 = FrontCamera(0)
    cam2 = FrontCamera(2)
    
    # メインプロセスを実行するクラス
    mainprocess = MainProcess(model_path)

    # 処理数
    count = [0,0]
    
    # マルチスレッドの実行
    mainprocess.thread_start(cam1)
    
    start_time = time.time()
    while True:
        try:
            frame, id, items, x, y, z, is_obtainable = mainprocess.q_results.get()
            #_, id, items, x, y, z, is_obtainable = (1,1,1,1,True)
            print(f"\nid:{id}, items:{items}, x:{x}, y:{y}, z:{z}, is_obtainable:{is_obtainable}")
            count[id] += 1
            
            cv2.drawMarker(frame, (160,128), (0,0,255))
            cv2.imshow(f'frame_{id}', frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            
        except KeyboardInterrupt:
            break
    cam1.release()
    cam2.release()
    mainprocess.finish()
    end_time = time.time()
    print(f"id0 : {count[0] / (end_time - start_time)} fps")
    print(f"id1 : {count[1] / (end_time - start_time)} fps")
    
    
    cv2.destroyAllWindows()
