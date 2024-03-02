import cv2
from src import FrontCamera

if __name__ == "__main__":
    cam = FrontCamera('models/20240109best.pt', 0)
    while cam.cap.isOpened():
        items = cam.DetectedObjectCounter()
        x,y,z = cam.ObjectPosition()
        is_obtainable = cam.IsObtainable()
        if is_obtainable:
            print(f"\nitems:{items}, x:{x}, y:{y}, z:{z}, is_obtainable:{is_obtainable}")

        if not cam.ret:
            continue
        cv2.drawMarker(cam.annotated_frame, (x,y), (0,0,255))
        cv2.imshow('image', cam.annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
    cv2.destroyAllWindows()