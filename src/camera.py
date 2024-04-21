import numpy as np
import cv2
import pyrealsense2 as rs
import subprocess
from enum import Enum
import time
import datetime

# カメラからの画像の幅と高さ[pxl]
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# FPS
FPS = 30

# WHITE BALANCE
WB = 5500
RS_WB = 5000

# Gain
GAIN = 50
RS_GAIN = 50

# Contrast
CONTRAST = 50
RS_CONTRAST = 50

# Hue
HUE = -1
RS_HUE = 0

# Saturation
SATURATION = 32
RS_SATURATION = 64

# Brightness
BRIGHTNESS = 170
RS_BRIGHTNESS = 0

# Front Upper Realsense serial number
FRONT_UPPER_REALSENSE_SERIAL_NUMBER = '242622071603'

# Rear Realsense serial number
REAR_REALSENSE_SERIAL_NUMBER = '944122072123'

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
    def __init__(self, timestamp):
        try:
            # Configure depth and color streams
            self.pipeline = rs.pipeline()
            config = rs.config()
            
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
            
            device = self.pipeline.get_active_profile().get_device()
            
            rgb_camera_sensor = [s for s in device.sensors if s.get_info(rs.camera_info.name) == 'RGB Camera'][0]
            rgb_camera_sensor.set_option(rs.option.enable_auto_white_balance, False)
            rgb_camera_sensor.set_option(rs.option.white_balance, RS_WB)
            rgb_camera_sensor.set_option(rs.option.gain, RS_GAIN)
            rgb_camera_sensor.set_option(rs.option.contrast, RS_CONTRAST)
            rgb_camera_sensor.set_option(rs.option.hue, RS_HUE)
            rgb_camera_sensor.set_option(rs.option.saturation, RS_SATURATION)
            rgb_camera_sensor.set_option(rs.option.brightness, RS_BRIGHTNESS)
            
            print(f"{rgb_camera_sensor.get_option(rs.option.gain)=}")
            print(f"{rgb_camera_sensor.get_option(rs.option.contrast)=}")
            print(f"{rgb_camera_sensor.get_option(rs.option.hue)=}")
            print(f"{rgb_camera_sensor.get_option(rs.option.saturation)=}")
            print(f"{rgb_camera_sensor.get_option(rs.option.brightness)=}")
            print(f"{rgb_camera_sensor.get_option(rs.option.enable_auto_white_balance)=}")
            print(f"realsense{device.get_info(rs.camera_info.serial_number)}, fps:{FPS}, WB:{rgb_camera_sensor.get_option(rs.option.white_balance)}")
        
            #output_filename = f"{timestamp}_UpperCamera.mp4"
            #fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            #self.output_file = cv2.VideoWriter(output_filename,fourcc,FPS,(FRAME_WIDTH*2,FRAME_HEIGHT*2))
        except Exception as e:
            print(e)
            print(f"realsense{FRONT_UPPER_REALSENSE_SERIAL_NUMBER} not connected")
        
        # 複数のRealsenseのパイプラインを開く時間に間隔を設けることでRuntimeErrorの解消を図る
        time.sleep(1)
        
        # 焦点距離
        focal_length = 270
        # ロボットの中心位置を原点とした時のカメラの位置[mm]
        pos_x = -155
        pos_y = 430
        pos_z = 100
        # カメラ座標におけるカメラの傾き[rad]
        theta_x = 10*np.pi/180
        theta_y = 0
        theta_z = 0
        
        # 俯瞰画像にする領域のマージン
        UPPER_MERGIN = 110
        upper_bird_point = np.array([[FRAME_WIDTH,FRAME_HEIGHT],[FRAME_WIDTH-UPPER_MERGIN,FRAME_HEIGHT/2],[UPPER_MERGIN,FRAME_HEIGHT/2],[0,FRAME_HEIGHT]], dtype=np.float32)
        
        self.params = (focal_length,pos_x,pos_y,pos_z,theta_x,theta_y,theta_z,upper_bird_point)
        
        # counter for calculate fps
        self.counter = 0
        self.start_time = time.time()

    def read(self):
        try:
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
        except:
            return False, None, None

    #def write(self, frame):
        # self.output_file.write(frame)
        
    def release(self):
        self.pipeline.stop()
        # self.output_file.release()
        print(f"UpperCamera : {self.counter/(time.time()-self.start_time)}fps")
        print("Closed Realsense Device")
    
    def isOpened(self):
        connected_devices = rs.context().query_devices()
        serial_number_list = [d.get_info(rs.camera_info.serial_number) for d in connected_devices]
        return True if FRONT_UPPER_REALSENSE_SERIAL_NUMBER in serial_number_list else False
        
class LowerCamera:
    def __init__(self, timestamp, id=None):
        if id is not None:
            device_id = id
        else:
            device_id = usb_video_device(PORT_ID.USB2_LOWER.value)
        # Camera Settings
        self.cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        set_fps = self.cap.set(cv2.CAP_PROP_FPS, FPS)
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0.0)
        set_wb = self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, WB)
        self.cap.set(cv2.CAP_PROP_GAIN, GAIN)
        self.cap.set(cv2.CAP_PROP_CONTRAST, CONTRAST)
        self.cap.set(cv2.CAP_PROP_SATURATION, SATURATION)
        self.cap.set(cv2.CAP_PROP_HUE, HUE)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, BRIGHTNESS)
        
        print(f"{self.cap.get(cv2.CAP_PROP_GAIN)=}")
        print(f"{self.cap.get(cv2.CAP_PROP_CONTRAST)=}")
        print(f"{self.cap.get(cv2.CAP_PROP_SATURATION)=}")
        print(f"{self.cap.get(cv2.CAP_PROP_HUE)=}")
        print(f"{self.cap.get(cv2.CAP_PROP_BRIGHTNESS)=}")
        print(f"{self.cap.get(cv2.CAP_PROP_AUTO_WB)=}")
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
        
        LOWER_MERGIN = 100
        lower_bird_point = np.array([[FRAME_WIDTH,FRAME_HEIGHT],[FRAME_WIDTH-LOWER_MERGIN,0],[LOWER_MERGIN,0],[0,FRAME_HEIGHT]], dtype=np.float32)
        
        self.params = (focal_length,pos_x,pos_y,pos_z,theta_x,theta_y,theta_z,lower_bird_point)
    
        # counter for calculate fps
        self.counter = 0
        self.start_time = time.time()
        
        #output_filename = f"{timestamp}_LowerCamera.mp4"
        #fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        #self.output_file = cv2.VideoWriter(output_filename,fourcc,FPS,(FRAME_WIDTH*2,FRAME_HEIGHT*2))
        
    def read(self):
        ret, frame = self.cap.read()
        self.counter += 1
        # Noneはダミー（デプスがある時と同じ引数の数にするため）
        return ret, frame, None
    
    #def write(self, frame):
        # self.output_file.write(frame)

    def release(self):
        self.cap.release()
        # self.output_file.release()
        print(f"LowerCamera : {self.counter/(time.time()-self.start_time)}fps")
        print("Closed Capturing Device")
    
    def isOpened(self):
        return self.cap.isOpened()
    

class RearCamera:
    def __init__(self, timestamp):
        try:
            # Configure depth and color streams
            self.pipeline = rs.pipeline()
            config = rs.config()
            
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
            
            device = self.pipeline.get_active_profile().get_device()
            
            rgb_camera_sensor = [s for s in device.sensors if s.get_info(rs.camera_info.name) == 'RGB Camera'][0]
            rgb_camera_sensor.set_option(rs.option.enable_auto_white_balance, False)
            rgb_camera_sensor.set_option(rs.option.white_balance, RS_WB)
            rgb_camera_sensor.set_option(rs.option.gain, RS_GAIN)
            rgb_camera_sensor.set_option(rs.option.contrast, RS_CONTRAST)
            rgb_camera_sensor.set_option(rs.option.hue, RS_HUE)
            rgb_camera_sensor.set_option(rs.option.saturation, RS_SATURATION)
            rgb_camera_sensor.set_option(rs.option.brightness, RS_BRIGHTNESS)
            
            print(f"{rgb_camera_sensor.get_option(rs.option.contrast)=}")
            print(f"{rgb_camera_sensor.get_option(rs.option.gain)=}")
            print(f"{rgb_camera_sensor.get_option(rs.option.hue)=}")
            print(f"{rgb_camera_sensor.get_option(rs.option.saturation)=}")
            print(f"{rgb_camera_sensor.get_option(rs.option.brightness)=}")
            print(f"{rgb_camera_sensor.get_option(rs.option.enable_auto_white_balance)=}")
            print(f"realsense{device.get_info(rs.camera_info.serial_number)}, fps:{FPS}, WB:{rgb_camera_sensor.get_option(rs.option.white_balance)}")
        
            #output_filename = f"{timestamp}_RearCamera.mp4"
            #fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            #self.output_file = cv2.VideoWriter(output_filename,fourcc,FPS,(FRAME_WIDTH,FRAME_HEIGHT))
        except Exception as e:
            print(e)
            print(f"realsense{REAR_REALSENSE_SERIAL_NUMBER} not connected")
        
        # 複数のRealsenseのパイプラインを開く時間に間隔を設けることでRuntimeErrorの解消を図る
        time.sleep(1)
        
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
        
        REAR_MERGIN = 110
        rear_bird_point = np.array([[FRAME_WIDTH,FRAME_HEIGHT],[FRAME_WIDTH-REAR_MERGIN,FRAME_HEIGHT/2],[REAR_MERGIN,FRAME_HEIGHT/2],[0,FRAME_HEIGHT]], dtype=np.float32)
        
        self.params = (focal_length,pos_x,pos_y,pos_z,theta_x,theta_y,theta_z,rear_bird_point)

        # counter for calculate fps
        self.counter = 0
        self.start_time = time.time()
        
    def read(self):
        try:
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
        except:
            return False, None, None

    #def write(self, frame):
        # self.output_file.write(frame)
        
    def release(self):
        self.pipeline.stop()
        # self.output_file.release()
        print(f"RearCamera : {self.counter/(time.time()-self.start_time)}fps")
        print("Closed Realsense Device")
    
    def isOpened(self):
        connected_devices = rs.context().query_devices()
        serial_number_list = [d.get_info(rs.camera_info.serial_number) for d in connected_devices]
        return True if REAR_REALSENSE_SERIAL_NUMBER in serial_number_list else False
