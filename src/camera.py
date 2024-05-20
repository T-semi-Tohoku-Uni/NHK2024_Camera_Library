import numpy as np
import cv2
import pyrealsense2 as rs
import subprocess
from enum import Enum
import time
import datetime
from multiprocessing import RawArray, Lock, Process
import ctypes
from typing import Tuple
import sys
from threading import Thread
import os

# カメラからの画像の幅と高さ[pxl]
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# FPS
FPS = 30

# WHITE BALANCE
WB = 5400
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

class RealsenseObject:
    def __init__(
            self, 
            timestamp,
            serial_number: str,
            focal_length,
            pos_x,
            pos_y,
            pos_z,
            theta_x,
            theta_y,
            theta_z,
            saved_image_dir=None,
        ):
        try:
            self.__serial_number = serial_number
            
            # Configure depth and color streams
            self.pipeline = rs.pipeline()
            config = rs.config()
            
            # RealSenseのシリアル番号で指定
            config.enable_device(serial_number)
            config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, FPS)
            config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, FPS)
            
            # Start streaming
            self.pipeline.start(config)
            
            align_to = rs.stream.color
            self.align = rs.align(align_to)
            
            device = self.pipeline.get_active_profile().get_device()
            
            rgb_camera_sensor = [s for s in device.sensors if s.get_info(rs.camera_info.name) == 'RGB Camera'][0]
            rgb_camera_sensor.set_option(rs.option.enable_auto_white_balance, True)
            # rgb_camera_sensor.set_option(rs.option.white_balance, RS_WB)
            # rgb_camera_sensor.set_option(rs.option.gain, RS_GAIN)
            # rgb_camera_sensor.set_option(rs.option.contrast, RS_CONTRAST)
            # rgb_camera_sensor.set_option(rs.option.hue, RS_HUE)
            # rgb_camera_sensor.set_option(rs.option.saturation, RS_SATURATION)
            # rgb_camera_sensor.set_option(rs.option.brightness, RS_BRIGHTNESS)
            
            print(f"{rgb_camera_sensor.get_option(rs.option.gain)=}")
            print(f"{rgb_camera_sensor.get_option(rs.option.contrast)=}")
            print(f"{rgb_camera_sensor.get_option(rs.option.hue)=}")
            print(f"{rgb_camera_sensor.get_option(rs.option.saturation)=}")
            print(f"{rgb_camera_sensor.get_option(rs.option.brightness)=}")
            print(f"{rgb_camera_sensor.get_option(rs.option.enable_auto_white_balance)=}")
            print(f"realsense{device.get_info(rs.camera_info.serial_number)}, fps:{FPS}, WB:{rgb_camera_sensor.get_option(rs.option.white_balance)}")
            
        except Exception as e:
            print(e)
            print(f"realsense{serial_number} not connected")
            
        # 複数のRealsenseのパイプラインを開く時間に間隔を設けることでRuntimeErrorの解消を図る
        time.sleep(1)
        
        UPPER_MERGIN = 110
        upper_bird_point = np.array([[FRAME_WIDTH,FRAME_HEIGHT],[FRAME_WIDTH-UPPER_MERGIN,FRAME_HEIGHT/2],[UPPER_MERGIN,FRAME_HEIGHT/2],[0,FRAME_HEIGHT]], dtype=np.float32)
        
        self.params = (focal_length,pos_x,pos_y,pos_z,theta_x,theta_y,theta_z,upper_bird_point)
        
        # counter for calculate fps
        self.counter = 0
        self.start_time = time.time()
        
        # 最新の画像を載せる共有メモリの変数
        self.__image_buffer = ImageSharedMemory((FRAME_HEIGHT, FRAME_WIDTH, 3))

        # self.capture_thread = Thread(target=self.capture, daemon=True)
        self.saved_dir = saved_image_dir
        
    # カメラのキャプチャー, 別のプロセスで動かす
    def capture(self):
        try:
            while True:
                try:
                    # Wait for a coherent pair of frames: depth and color
                    frames = self.pipeline.wait_for_frames()
                    aligned_frames = self.align.process(frames)
                    depth_frame = aligned_frames.get_depth_frame()
                    color_frame = aligned_frames.get_color_frame()
                    if not depth_frame or not color_frame:
                        self.__image_buffer.write_black_frame()
                    
                    # Convert images to numpy arrays
                    depth_image = np.asanyarray(depth_frame.get_data())
                    color_image = np.asanyarray(color_frame.get_data())
                    
                    self.counter += 1
                    
                    self.__image_buffer.write_rgb(color_image)
                    self.__image_buffer.write_depth(depth_image)
                except:
                    self.__image_buffer.write_black_frame()
        except Exception as e:
            print(f"{e}")
    
    def read_image_buffer(self) -> Tuple[np.ndarray, np.ndarray]:
        color_image = self.__image_buffer.read_rgb()
        depth_image = self.__image_buffer.read_depth()
        return (color_image, depth_image)
    
    def save_image(
        self, 
        dir: str, 
        frame: np.ndarray,
        classes=None,
        xywhn=None
    ):
        if self.saved_dir is None:
            raise ValueError("You must set saved_image_dir when creating instance")
        
        base_dir = os.path.join(self.saved_dir, self.__serial_number, dir)
        saved_image_dir = os.path.join(base_dir, "image")
        timestamp = time.time()
        if not os.path.exists(saved_image_dir):
            os.makedirs(saved_image_dir)
        
        cv2.imwrite(os.path.join(saved_image_dir, f"{str(timestamp)}.jpg"), frame)
        
        if classes is None:
            return
        
        if xywhn is None:
            return
    
        saved_bounding_box_dir = os.path.join(base_dir, "box")
        if not os.path.exists(saved_bounding_box_dir):
            os.makedirs(saved_bounding_box_dir)
        
        with open(os.path.join(saved_bounding_box_dir, f"{str(timestamp)}.txt"), mode="w") as f:
            for index, cls in enumerate(classes):
                f.write(f"{int(cls)} {(xywhn[index][0])} {xywhn[index][1]} {xywhn[index][2]} {xywhn[index][3]}\n")
        
    
    def release(self):
        self.pipeline.stop()
        # self.output_file.release()
        print(f"{self.__serial_number} : {self.counter/(time.time()-self.start_time)}fps")
        print("Closed Realsense Device")
    
    def isOpened(self):
        connected_devices = rs.context().query_devices()
        serial_number_list = [d.get_info(rs.camera_info.serial_number) for d in connected_devices]
        return True if self.__serial_number in serial_number_list else False
    
# class UpperCamera:
#     def __init__(self, timestamp):
#         try:
#             # Configure depth and color streams
#             self.pipeline = rs.pipeline()
#             config = rs.config()
            
#             # RealSenseのシリアル番号で指定
#             config.enable_device(FRONT_UPPER_REALSENSE_SERIAL_NUMBER)
#             config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, FPS)
#             config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, FPS)
            
#             # Start streaming
#             self.pipeline.start(config)
            
#             # Create an align object
#             # rs.align allows us to perform alignment of depth frames to others frames
#             # The "align_to" is the stream type to which we plan to align depth frames.
#             align_to = rs.stream.color
#             self.align = rs.align(align_to)
            
#             device = self.pipeline.get_active_profile().get_device()
            
#             rgb_camera_sensor = [s for s in device.sensors if s.get_info(rs.camera_info.name) == 'RGB Camera'][0]
#             rgb_camera_sensor.set_option(rs.option.enable_auto_white_balance, True)
#             # rgb_camera_sensor.set_option(rs.option.white_balance, RS_WB)
#             # rgb_camera_sensor.set_option(rs.option.gain, RS_GAIN)
#             # rgb_camera_sensor.set_option(rs.option.contrast, RS_CONTRAST)
#             # rgb_camera_sensor.set_option(rs.option.hue, RS_HUE)
#             # rgb_camera_sensor.set_option(rs.option.saturation, RS_SATURATION)
#             # rgb_camera_sensor.set_option(rs.option.brightness, RS_BRIGHTNESS)
            
#             print(f"{rgb_camera_sensor.get_option(rs.option.gain)=}")
#             print(f"{rgb_camera_sensor.get_option(rs.option.contrast)=}")
#             print(f"{rgb_camera_sensor.get_option(rs.option.hue)=}")
#             print(f"{rgb_camera_sensor.get_option(rs.option.saturation)=}")
#             print(f"{rgb_camera_sensor.get_option(rs.option.brightness)=}")
#             print(f"{rgb_camera_sensor.get_option(rs.option.enable_auto_white_balance)=}")
#             print(f"realsense{device.get_info(rs.camera_info.serial_number)}, fps:{FPS}, WB:{rgb_camera_sensor.get_option(rs.option.white_balance)}")

#             # 最新の画像を載せる共有メモリの変数
#             # original_image = np.zeros((480, 640, 3), dtype=np.uint8)
#             # rgb_shard_array = RawArray(ctypes.c_uint8, original_image.size)
#             # depth_image_buf = RawArray(ctypes.c_uint8, original_image.size)
#             # self.rgb_image_buf = np.frombuffer(rgb_shard_array, dtype=np.uint8).reshape(original_image.shape)
#             # self.depth_image_buf = np.frombuffer(depth_image_buf, dtype=np.uint8).reshape(original_image.shape)
            
#             #output_filename = f"{timestamp}_UpperCamera.mp4"
#             #fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#             #self.output_file = cv2.VideoWriter(output_filename,fourcc,FPS,(FRAME_WIDTH*2,FRAME_HEIGHT*2))
#         except Exception as e:
#             print(e)
#             print(f"realsense{FRONT_UPPER_REALSENSE_SERIAL_NUMBER} not connected", file=sys.__stderr__)
        
#         # 複数のRealsenseのパイプラインを開く時間に間隔を設けることでRuntimeErrorの解消を図る
#         time.sleep(1)
        
#         # 焦点距離
#         focal_length = 270
#         # ロボットの中心位置を原点とした時のカメラの位置[mm]
#         pos_x = -155
#         pos_y = 430
#         pos_z = 100
#         # カメラ座標におけるカメラの傾き[rad]
#         theta_x = 10*np.pi/180
#         theta_y = 0
#         theta_z = 0
        
#         # 俯瞰画像にする領域のマージン
#         UPPER_MERGIN = 110
#         upper_bird_point = np.array([[FRAME_WIDTH,FRAME_HEIGHT],[FRAME_WIDTH-UPPER_MERGIN,FRAME_HEIGHT/2],[UPPER_MERGIN,FRAME_HEIGHT/2],[0,FRAME_HEIGHT]], dtype=np.float32)
        
#         self.params = (focal_length,pos_x,pos_y,pos_z,theta_x,theta_y,theta_z,upper_bird_point)
        
#         # counter for calculate fps
#         self.counter = 0
#         self.start_time = time.time()
        
#         # 最新の画像を載せる共有メモリの変数
#         self.image_buffer = ImageSharedMemory((FRAME_HEIGHT, FRAME_WIDTH, 3))

#     def read(self):
#         try:
#             # Wait for a coherent pair of frames: depth and color
#             frames = self.pipeline.wait_for_frames()
#             aligned_frames = self.align.process(frames)
#             depth_frame = aligned_frames.get_depth_frame()
#             color_frame = aligned_frames.get_color_frame()
#             if not depth_frame or not color_frame:
#                 return False, None, None
            
#             # Convert images to numpy arrays
#             depth_image = np.asanyarray(depth_frame.get_data())
#             color_image = np.asanyarray(color_frame.get_data())
            
#             self.counter += 1
            
#             return True, color_image, depth_image
#         except:
#             return False, None, None

#     #def write(self, frame):
#         # self.output_file.write(frame)
        
#     def release(self):
#         self.pipeline.stop()
#         # self.output_file.release()
#         print(f"UpperCamera : {self.counter/(time.time()-self.start_time)}fps")
#         print("Closed Realsense Device")
    
#     def isOpened(self):
#         connected_devices = rs.context().query_devices()
#         serial_number_list = [d.get_info(rs.camera_info.serial_number) for d in connected_devices]
#         return True if FRONT_UPPER_REALSENSE_SERIAL_NUMBER in serial_number_list else False

# class LowerCamera:
#     def __init__(self, timestamp):
#         try:
#             # Configure depth and color streams
#             self.pipeline = rs.pipeline()
#             config = rs.config()
            
#             # RealSenseのシリアル番号で指定
#             config.enable_device(REAR_REALSENSE_SERIAL_NUMBER)
#             config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, FPS)
#             config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, FPS)
            
#             # Start streaming
#             self.pipeline.start(config)
            
#             # Create an align object
#             # rs.align allows us to perform alignment of depth frames to others frames
#             # The "align_to" is the stream type to which we plan to align depth frames.
#             align_to = rs.stream.color
#             self.align = rs.align(align_to)
            
#             device = self.pipeline.get_active_profile().get_device()
            
#             rgb_camera_sensor = [s for s in device.sensors if s.get_info(rs.camera_info.name) == 'RGB Camera'][0]
#             rgb_camera_sensor.set_option(rs.option.enable_auto_white_balance, True)
#             # rgb_camera_sensor.set_option(rs.option.white_balance, RS_WB)
#             # rgb_camera_sensor.set_option(rs.option.gain, RS_GAIN)
#             # rgb_camera_sensor.set_option(rs.option.contrast, RS_CONTRAST)
#             # rgb_camera_sensor.set_option(rs.option.hue, RS_HUE)
#             # rgb_camera_sensor.set_option(rs.option.saturation, RS_SATURATION)
#             # rgb_camera_sensor.set_option(rs.option.brightness, RS_BRIGHTNESS)
            
#             print(f"{rgb_camera_sensor.get_option(rs.option.contrast)=}")
#             print(f"{rgb_camera_sensor.get_option(rs.option.gain)=}")
#             print(f"{rgb_camera_sensor.get_option(rs.option.hue)=}")
#             print(f"{rgb_camera_sensor.get_option(rs.option.saturation)=}")
#             print(f"{rgb_camera_sensor.get_option(rs.option.brightness)=}")
#             print(f"{rgb_camera_sensor.get_option(rs.option.enable_auto_white_balance)=}")
#             print(f"realsense{device.get_info(rs.camera_info.serial_number)}, fps:{FPS}, WB:{rgb_camera_sensor.get_option(rs.option.white_balance)}")
            
#             #output_filename = f"{timestamp}_RearCamera.mp4"
#             #fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#             #self.output_file = cv2.VideoWriter(output_filename,fourcc,FPS,(FRAME_WIDTH,FRAME_HEIGHT))
#         except Exception as e:
#             print(e)
#             print(f"realsense{FRONT_UPPER_REALSENSE_SERIAL_NUMBER} not connected")
        
#         # 複数のRealsenseのパイプラインを開く時間に間隔を設けることでRuntimeErrorの解消を図る
#         time.sleep(1)
        
#         # 焦点距離
#         focal_length = 270
#         # ロボットの中心位置を原点とした時のカメラの位置[mm]
#         pos_x = 0
#         pos_y = 300
#         pos_z = -150
#         # カメラ座標におけるカメラの傾き[rad]
#         theta_x = 0
#         theta_y = 0
#         theta_z = 0
        
#         REAR_MERGIN = 110
#         rear_bird_point = np.array([[FRAME_WIDTH,FRAME_HEIGHT],[FRAME_WIDTH-REAR_MERGIN,FRAME_HEIGHT/2],[REAR_MERGIN,FRAME_HEIGHT/2],[0,FRAME_HEIGHT]], dtype=np.float32)
        
#         self.params = (focal_length,pos_x,pos_y,pos_z,theta_x,theta_y,theta_z,rear_bird_point)

#         # counter for calculate fps
#         self.counter = 0
#         self.start_time = time.time()
        
#         # 最新の画像を載せる共有メモリの変数
#         self.image_buffer = ImageSharedMemory((FRAME_HEIGHT, FRAME_WIDTH, 3))
        
#     def read(self):
#         try:
#             # Wait for a coherent pair of frames: depth and color
#             frames = self.pipeline.wait_for_frames()
#             aligned_frames = self.align.process(frames)
#             depth_frame = aligned_frames.get_depth_frame()
#             color_frame = aligned_frames.get_color_frame()
#             if not depth_frame or not color_frame:
#                 return False, None, None
            
#             # Convert images to numpy arrays
#             depth_image = np.asanyarray(depth_frame.get_data())
#             color_image = np.asanyarray(color_frame.get_data())
            
#             self.counter += 1
            
#             return True, color_image, depth_image
#         except:
#             return False, None, None

#     #def write(self, frame):
#         # self.output_file.write(frame)
        
#     def release(self):
#         self.pipeline.stop()
#         # self.output_file.release()
#         print(f"RearCamera : {self.counter/(time.time()-self.start_time)}fps")
#         print("Closed Realsense Device")
    
#     def isOpened(self):
#         connected_devices = rs.context().query_devices()
#         serial_number_list = [d.get_info(rs.camera_info.serial_number) for d in connected_devices]
#         return True if REAR_REALSENSE_SERIAL_NUMBER in serial_number_list else False

class ImageSharedMemory:
    def __init__(self, image_size: Tuple[int, int, int]):
        # 使用する共有メモリの宣言
        original_image = np.zeros(image_size, dtype=np.uint8)
        rgb_shard_array = RawArray(ctypes.c_uint8, original_image.size)
        depth_image_buf = RawArray(ctypes.c_uint8, original_image.size)
        self.__rgb_image_buf: np.ndarray = np.frombuffer(rgb_shard_array, dtype=np.uint8).reshape(original_image.shape)
        self.__depth_image_buf: np.ndarray = np.frombuffer(depth_image_buf, dtype=np.uint8).reshape(original_image.shape)
        
        # print(original_image.shape)
        # print(self.rgb_image_buf.shape)
        # print(self.depth_image_buf.shape)
        
        # 排他ロック用の変数
        self.__rgb_lock = Lock()
        self.__depth_lock = Lock()
        
        # 例外が生じた時用の真っ黒な画像
        self.__black_image = np.zeros(image_size,dtype=np.uint8)
        
    def read_rgb(self) -> np.ndarray:
        with self.__rgb_lock:
            rgb_image = self.__rgb_image_buf
        return rgb_image
    
    def read_depth(self) -> np.ndarray:
        with self.__depth_lock:
            depth_image = self.__depth_image_buf
        return depth_image
    
    def write_rgb(self, rgb_image: np.ndarray):
        with self.__rgb_lock:
            self.__rgb_image_buf = rgb_image
        return
    
    def write_depth(self, depth_image: np.ndarray):
        with self.__depth_lock:
            self.__depth_image_buf = depth_image
        return

    def write_black_frame(self):
        self.write_rgb(self.__black_image)
        self.write_depth(self.__black_image)