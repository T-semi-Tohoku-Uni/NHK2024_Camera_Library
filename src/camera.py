import numpy as np
import cv2
import pyrealsense2 as rs
import subprocess
from enum import Enum

# カメラからの画像の幅と高さ[pxl]
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# FPS
FPS = 30

class PORT_ID(Enum):
    USB3_UPPER = 1
    USB3_LOWER = 2
    USB2_UPPER = 3
    USB2_LOWER = 4

def usb_video_device(port : int):
    try:
        cmd = 'ls -la /dev/v4l/by-path'
        res = subprocess.check_output(cmd.split())
        by_path = res.decode()
        for line in by_path.split('\n'):
            # wsl2だとusb-0:{port},raspiだとusb-0:1.{port}
            if(f'usb-0:1.{port}' in line):
                tmp = line.split('index')[1][0]
                if int(tmp) % 2 == 0:
                    video_index = line.split('../../video')[1][0]
        
        cmd = 'ls -la /dev/v4l/by-id'
        res = subprocess.check_output(cmd.split())
        by_id = res.decode()
        for line in by_id.split('\n'):
            if(f'video{video_index}' in line):
                name = line.split('usb-')[1].split('-video-index')[0]
        
        print(f'{port=}\n{video_index=}\n{name=}')
        
        return int(video_index)
                
    except:
        return

class UpperCamera:
    def __init__(self):
        device_id = usb_video_device(PORT_ID.USB3_UPPER.value)
        # Camera Settings
        self.cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
        # self.cap = cv2.VideoCapture(device_id)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        set_fps = self.cap.set(cv2.CAP_PROP_FPS, FPS)
        print(f"device_id_{device_id} fps:{self.cap.get(cv2.CAP_PROP_FPS)}, {set_fps}")
        
        camera_parameter = [cv2.CAP_PROP_FRAME_WIDTH,
        cv2.CAP_PROP_FRAME_HEIGHT,
        cv2.CAP_PROP_FOURCC,
        cv2.CAP_PROP_BRIGHTNESS,
        cv2.CAP_PROP_CONTRAST,
        cv2.CAP_PROP_SATURATION,
        cv2.CAP_PROP_HUE,
        cv2.CAP_PROP_GAIN,
        cv2.CAP_PROP_AUTO_EXPOSURE,
        cv2.CAP_PROP_EXPOSURE,
        cv2.CAP_PROP_AUTO_WB,
        cv2.CAP_PROP_WB_TEMPERATURE,
        cv2.CAP_PROP_AUTOFOCUS,
        ]

        params = ['cv2.CAP_PROP_FRAME_WIDTH',
        'cv2.CAP_PROP_FRAME_HEIGHT',
        'cv2.CAP_PROP_FOURCC',
        'cv2.CAP_PROP_BRIGHTNESS',
        'cv2.CAP_PROP_CONTRAST',
        'cv2.CAP_PROP_SATURATION',
        'cv2.CAP_PROP_HUE',
        'cv2.CAP_PROP_GAIN',
        'cv2.CAP_PROP_AUTO_EXPOSURE',
        'cv2.CAP_PROP_EXPOSURE',
        'cv2.CAP_PROP_AUTO_WB',
        'cv2.CAP_PROP_WB_TEMPERATURE',
        'cv2.CAP_PROP_AUTOFOCUS',
        ]
        """
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, -4.0)
        self.cap.set(cv2.CAP_PROP_CONTRAST, 15)
        self.cap.set(cv2.CAP_PROP_SATURATION, 32)
        self.cap.set(cv2.CAP_PROP_HUE, 0.0)
        self.cap.set(cv2.CAP_PROP_GAIN, -1.0)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, -1.0)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, 200)
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0.0)
        self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 2500)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, -1.0)
        """

        for x in range(len(params)):
            print(f"{params[x]} = {self.cap.get(camera_parameter[x])}")
    
        # v4l2-ctlを用いたホワイトバランスの固定
        #cmd = 'v4l2-ctl -d /dev/video0 -c white_balance_automatic=0 -c white_balance_temperature=4500'
        #ret = subprocess.check_output(cmd, shell=True)
            
    def read(self):
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        self.cap.release()
        print("Closed Capturing Device")
    
    def isOpened(self):
        return self.cap.isOpened()

class LowerCamera:
    def __init__(self):
        if len(rs.context().query_devices()) == 0:
            print(f"rs not connected")
        else:
            # Configure depth and color streams
            self.pipeline = rs.pipeline()
            config = rs.config()

            # Get device product line for setting a supporting resolution
            pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
            pipeline_profile = config.resolve(pipeline_wrapper)
            device = pipeline_profile.get_device()
            
            found_rgb = False
            for s in device.sensors:
                if s.get_info(rs.camera_info.name) == 'RGB Camera':
                    found_rgb = True
                    break
            if not found_rgb:
                print("The demo requires Depth camera with Color sensor")
                exit(0)

            config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, FPS)
            config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, FPS)
            
            # Start streaming
            self.pipeline.start(config)
            
            # Create an align object
            # rs.align allows us to perform alignment of depth frames to others frames
            # The "align_to" is the stream type to which we plan to align depth frames.
            align_to = rs.stream.color
            self.align = rs.align(align_to)
            print(f"{device} fps:{FPS}")
            
        
    def read(self):
        if len(rs.context().query_devices()) == 0:
            return False, None, None
        else:
            # Wait for a coherent pair of frames: depth and color
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                return False, None, None
            
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            return True, color_image, depth_image

    def release(self):
        self.pipeline.stop()
        print("Closed Realsense Device")
    
    def isOpened(self):
        return False if len(rs.context().query_devices())==0 else True

class RearCamera:
    def __init__(self):
        if len(rs.context().query_devices()) == 0:
            print(f"rs not connected")
        else:
            # Configure depth and color streams
            self.pipeline = rs.pipeline()
            config = rs.config()

            # Get device product line for setting a supporting resolution
            pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
            pipeline_profile = config.resolve(pipeline_wrapper)
            device = pipeline_profile.get_device()
            
            found_rgb = False
            for s in device.sensors:
                if s.get_info(rs.camera_info.name) == 'RGB Camera':
                    found_rgb = True
                    break
            if not found_rgb:
                print("The demo requires Depth camera with Color sensor")
                exit(0)

            config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, FPS)
            config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, FPS)
            
            # Start streaming
            self.pipeline.start(config)
            
            # Create an align object
            # rs.align allows us to perform alignment of depth frames to others frames
            # The "align_to" is the stream type to which we plan to align depth frames.
            align_to = rs.stream.color
            self.align = rs.align(align_to)
            print(f"{device} fps:{FPS}")
            
        
    def read(self):
        if len(rs.context().query_devices()) == 0:
            return False, None, None
        else:
            # Wait for a coherent pair of frames: depth and color
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                return False, None, None
            
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            return True, color_image, depth_image

    def release(self):
        self.pipeline.stop()
        print("Closed Realsense Device")
    
    def isOpened(self):
        return False if len(rs.context().query_devices())==0 else True