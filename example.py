import cv2
from src import FrontCamera, coordinate_transformation

if __name__ == "__main__":
    model_path = 'models/20240109best.pt'
    cam = FrontCamera(model_path, 0)

    while True:
        try:
            items, x,y,z, is_obtainable, annotated_frame = cam.queue.get()
            print(f"\nitems:{items}, x:{x}, y:{y}, z:{z}, is_obtainable:{is_obtainable}")

            cv2.drawMarker(annotated_frame, (160,128), (0,0,255))
            cv2.imshow('image', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        except KeyboardInterrupt:
            break    
    cv2.destroyAllWindows()