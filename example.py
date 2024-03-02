import cv2
from src import FrontCamera

if __name__ == "__main__":
    cam = FrontCamera(0)
    while cam.cap.isOpened():
        items = cam.DetectedObjectCounter()
        print(f"items:{items}")
        x,y,z = cam.ObjectPosition()
        print(f"x:{x}, y:{y}, z:{z}")
        is_obtainable = cam.IsObtainable()
        print(is_obtainable)

        if not cam.ret:
            continue
        cv2.drawMarker(cam.annotated_frame, (x,y), (0,0,255))
        cv2.imshow('image', cam.annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
    cv2.destroyAllWindows()