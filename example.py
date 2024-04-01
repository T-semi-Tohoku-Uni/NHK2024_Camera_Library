import cv2
from src import FrontCamera, MainProcess

if __name__ == "__main__":
    #ncnn_model_path = 'models/20240109best_ncnn_model'
    model_path = 'models/20240109best.pt'
    
    # カメラのクラス
    cam1 = FrontCamera(0)
    cam2 = FrontCamera(2)
    
    # メインプロセスを実行するクラス
    mainprocess = MainProcess(model_path,cam1,cam2)
    
    # マルチスレッドの実行
    mainprocess.thread_start()
    
    while True:
        try:
            frame, id, items, x, y, z, is_obtainable = mainprocess.q_frames_list[-1].get()
            #_, id, items, x, y, z, is_obtainable = (1,1,1,1,True)
            print(f"\nid:{id}, items:{items}, x:{x}, y:{y}, z:{z}, is_obtainable:{is_obtainable}")
            
            cv2.drawMarker(frame, (160,128), (0,0,255))
            cv2.imshow(f'frame_{id}', frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            
        except KeyboardInterrupt:
            break
    mainprocess.finish()
    
    cv2.destroyAllWindows()
