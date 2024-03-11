import cv2
from src import FrontCamera, coordinate_transformation

if __name__ == "__main__":
    model_path = 'models/20240109best.pt'
    cam = FrontCamera(model_path, 0)

    while cam.cap.isOpened():
        items = cam.DetectedObjectCounter()
        x,y,z = cam.ObjectPosition()
        is_obtainable = cam.IsObtainable()
        print(f"\nitems:{items}, x:{x}, y:{y}, z:{z}, is_obtainable:{is_obtainable}")

        if not cam.ret:
            continue
        cv2.drawMarker(cam.annotated_frame, (160,120), (0,0,255))
        cv2.imshow('image', cam.annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
    cv2.destroyAllWindows()