import numpy as np
import cv2
import pyrealsense2 as rs
import subprocess
from enum import Enum
import time

# カメラからの画像の幅と高さ[pxl]
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# FPS
FPS = 30

# WHITE BALANCE
WB = 4500

# Front Upper Realsense serial number
FRONT_UPPER_REALSENSE_SERIAL_NUMBER = "242622071603"

# Rear Realsense serial number
REAR_REALSENSE_SERIAL_NUMBER = "944122072123"

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
            if(f'usb-0:1.{port}' in line):
                tmp = line.split('index')[1][0]
                if int(tmp) % 2 == 0:
                    video_index = line.split('../../video')[1]
        
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
        try:
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
                    rgb_camera_sensor = s
                    break
            if not found_rgb:
                print("The demo requires Depth camera with Color sensor")
                exit(0)
            # RealSenseのシリアル番号で指定
            config.enable_device(FRONT_UPPER_REALSENSE_SERIAL_NUMBER)
            config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, FPS)
            config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, FPS)
            
            # Start streaming
            self.pipeline.start(config)
            
            # Create an align object
            # rs.align allows us to perform alignment of depth frames to others frames
            # The "align_to" is the stream type to which we plan to align depth frames.
            align_to = rs.stream.color
            self.align = rs.align(align_to)
            rgb_camera_sensor.set_option(rs.option.enable_auto_white_balance, False)
            rgb_camera_sensor.set_option(rs.option.white_balance, WB)
            print(f"realsense{device.get_info(rs.camera_info.serial_number)}, fps:{FPS}, WB:{rgb_camera_sensor.get_option(rs.option.white_balance)}")
        except RuntimeError:
            print(f"realsense{FRONT_UPPER_REALSENSE_SERIAL_NUMBER} not connected")
        
        # 焦点距離
        focal_length = 270
        # ロボットの中心位置を原点とした時のカメラの位置[mm]
        pos_x = -100
        pos_y = 400
        pos_z = 150
        # カメラ座標におけるカメラの傾き[rad]
        theta_x = 15*np.pi/180
        theta_y = 0
        theta_z = 0
        
        self.params = (focal_length,pos_x,pos_y,pos_z,theta_x,theta_y,theta_z)
        
        # counter for calculate fps
        self.counter = 0
        self.start_time = time.time()

    def read(self):
        devices = rs.context().query_devices()
        serial_number_list = [d.get_info(rs.camera_info.serial_number) for d in devices]
        if FRONT_UPPER_REALSENSE_SERIAL_NUMBER not in serial_number_list:
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
            
            self.counter += 1
            
            return True, color_image, depth_image

    def release(self):
        self.pipeline.stop()
        print(f"UpperCamera : {self.counter/(time.time()-self.start_time)}fps")
        print("Closed Realsense Device")
    
    def isOpened(self):
        devices = rs.context().query_devices()
        serial_number_list = [d.get_info(rs.camera_info.serial_number) for d in devices]
        return False if FRONT_UPPER_REALSENSE_SERIAL_NUMBER not in serial_number_list else True

class LowerCamera:
    def __init__(self, id=None):
        if id is not None:
            device_id = id
        else:
            device_id = usb_video_device(PORT_ID.USB2_LOWER.value)
        # Camera Settings
        #self.cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
        self.cap = cv2.VideoCapture(device_id)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        set_fps = self.cap.set(cv2.CAP_PROP_FPS, FPS)
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0.0)
        set_wb = self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, WB)
        print(f"device_id_{device_id} fps:{self.cap.get(cv2.CAP_PROP_FPS)}->{set_fps} wb:{self.cap.get(cv2.CAP_PROP_WB_TEMPERATURE)}->{set_wb}")
        
        # v4l2-ctlを用いたホワイトバランスの固定
        #cmd = 'v4l2-ctl -d /dev/video0 -c white_balance_automatic=0 -c white_balance_temperature=4500'
        #ret = subprocess.check_output(cmd, shell=True)

        # 焦点距離
        focal_length = 270
        # ロボットの中心位置を原点とした時のカメラの位置[mm]
        pos_x = 100
        pos_y = 400
        pos_z = 150
        # カメラ座標におけるカメラの傾き[rad]
        theta_x = 30*np.pi/180
        theta_y = 0
        theta_z = 0
        
        self.params = (focal_length,pos_x,pos_y,pos_z,theta_x,theta_y,theta_z)
    
        # counter for calculate fps
        self.counter = 0
        self.start_time = time.time()
        
    def read(self):
        ret, frame = self.cap.read()
        self.counter += 1
        # Noneはダミー（デプスがある時と同じ引数の数にするため）
        return ret, frame, None

    def release(self):
        self.cap.release()
        print(f"LowerCamera : {self.counter/(time.time()-self.start_time)}fps")
        print("Closed Capturing Device")
    
    def isOpened(self):
        return self.cap.isOpened()
    

class RearCamera:
    def __init__(self):
        try:
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
                    rgb_camera_sensor = s
                    break
            if not found_rgb:
                print("The demo requires Depth camera with Color sensor")
                exit(0)
            # RealSenseのシリアル番号で指定
            config.enable_device(REAR_REALSENSE_SERIAL_NUMBER)
            config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, FPS)
            config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, FPS)
            
            # Start streaming
            self.pipeline.start(config)
            
            # Create an align object
            # rs.align allows us to perform alignment of depth frames to others frames
            # The "align_to" is the stream type to which we plan to align depth frames.
            align_to = rs.stream.color
            self.align = rs.align(align_to)
            rgb_camera_sensor.set_option(rs.option.enable_auto_white_balance, False)
            rgb_camera_sensor.set_option(rs.option.white_balance, WB)
            print(f"realsense{device.get_info(rs.camera_info.serial_number)}, fps:{FPS}, WB:{rgb_camera_sensor.get_option(rs.option.white_balance)}")
        except RuntimeError:
            print(f"realsense{REAR_REALSENSE_SERIAL_NUMBER} not connected")
         

        # 焦点距離
        focal_length = 270
        # ロボットの中心位置を原点とした時のカメラの位置[mm]
        pos_x = 0
        pos_y = 300
        pos_z = -150
        # カメラ座標におけるカメラの傾き[rad]
        theta_x = 0
        theta_y = 0
        theta_z = 0
        
        self.params = (focal_length,pos_x,pos_y,pos_z,theta_x,theta_y,theta_z)

        # counter for calculate fps
        self.counter = 0
        self.start_time = time.time()
        
    def read(self):
        devices = rs.context().query_devices()
        serial_number_list = [d.get_info(rs.camera_info.serial_number) for d in devices]
        if REAR_REALSENSE_SERIAL_NUMBER not in serial_number_list:
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
            
            self.counter += 1
                
            return True, color_image, depth_image

    def release(self):
        self.pipeline.stop()
        print(f"RearCamera : {self.counter/(time.time()-self.start_time)}fps")
        print("Closed Realsense Device")
    
    def isOpened(self):
        devices = rs.context().query_devices()
        serial_number_list = [d.get_info(rs.camera_info.serial_number) for d in devices]
        return False if REAR_REALSENSE_SERIAL_NUMBER not in serial_number_list else True
