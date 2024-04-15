# NHK2024_Camera_Library
R2のラズパイで用いるカメラ関係のライブラリ

# src/camera.py
カメラ関係のスクリプト

# src/detect.py
検出関係のスクリプト

# src/capture_and_detect.py
カメラのキャプチャと検出をカメラ毎にスレッドを立てて実行

# 使い方
インポート
`from NHK2024_Camera_Library import MainProcess, OUTPUT_ID, AREA_STATE`

ループの前に実行すべきもの
`# 物体検出モデルのパス`
`model_path = 'models/20240109best.pt'`
`# メインプロセスを実行するクラス`
`mainprocess = MainProcess(model_path)`
`# マルチスレッドの実行`
`mainprocess.thread_start()`

ループの中で実行すべきもの
- 出力の受け取り
frame:画像（デバッグ用なのでラズパイで動かすときはアンダースコアで受け取らない）
id:OUTPUT_ID(列挙型)　どの出力形式なのかを表す
output_data:タプル　出力データ　idによって中身が異なる
`# 出力を受け取る`
`frame, id, output_data = mainprocess.q_out.get()`

- idによる出力の受け取り方
ボールの検出
items:検出したオブジェクト数  int
x:目標のボールまでのx座標[mm] float
y:目標のボールまでのy座標[mm] float
z:目標のボールまでのz座標[mm] float
`if id == OUTPUT_ID.BALL:`
`    items,x,y,z,is_obtainable = output_data`

サイロの検出
x:目標のサイロまでのx座標[mm] float
y:目標のサイロまでのy座標[mm] float
z:目標のサイロまでのz座標[mm] float
`if id == OUTPUT_ID.SILO:`
`    x,y,z = output_data`

ラインの検出
forward:奥行方向のラインが伸びているかどうか bool
right:右方向のラインが伸びているかどうか bool
left:左方向のラインが伸びているかどうか bool
x:ラインがロボットの中心からどれだけずれているか[mm] float
`if id == OUTPUT_ID.LINE:`
`    forward, right, left, x = output_data`







# ncnnモデルを使用するためのaptパッケージ
https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-linux
```
sudo apt install build-essential git cmake libprotobuf-dev protobuf-compiler libomp-dev libvulkan-dev vulkan-tools libopencv-dev
```

## ラズパイでRealsenseを使えるようにしたかった
- まず、以下を参考にやってみた\
https://github.com/IntelRealSense/librealsense/blob/development/doc/installation.md#prerequisites
https://raspida.com/rpi-buster-error

```
sudo apt-get update --allow-releaseinfo-change
sudo apt full-upgrade
```

- 前準備
```
sudo apt-get install libssl-dev libusb-1.0-0-dev libudev-dev pkg-config libgtk-3-dev
```
```
sudo apt-get install git wget cmake build-essential
```
```
sudo apt-get install libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev at
```
- librealsenceインストール
```
git clone https://github.com/IntelRealSense/librealsense.git
```
```
./scripts/setup_udev_rules.sh
```
ここでPermission deniedがたくさん出てエラー

- 次に以下を参考にやってみた\
https://github.com/IntelRealSense/librealsense/blob/master/doc/installation_raspbian.md
https://openvino.jp/intel-realsense-camera-d435i-2/

swap追加
```
sudo nano /etc/dphys-swapfile
```
CONF_SWAPSIZE=2048に変更

```
sudo /etc/init.d/dphys-swapfile restart swapon -s
```

E: Unable to locate packageが出るものは飛ばした結果、以下をインストール
```
sudo apt-get install -y libdrm-amdgpu1 libdrm-dev libdrm-exynos1 libdrm-freedreno1 libdrm-nouveau2 libdrm-omap1 libdrm-radeon1 libdrm-tegra0 libdrm2

sudo apt-get install libglu1-mesa libglu1-mesa-dev glusterfs-common libglu1-mesa libglu1-mesa-dev

sudo apt-get install libglu1-mesa libglu1-mesa-dev mesa-utils mesa-utils-extra xorg-dev libgtk-3-dev libusb-1.0-0-dev
```

librealsenseのクローン
```
git clone https://github.com/IntelRealSense/librealsense.git

cd librealsense

sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/ 

sudo udevadm control --reload-rules && udevadm trigger 
```

ここでPermission deniedがたくさん出てエラー

- 以下を参考にやってみた\
https://github.com/datasith/Ai_Demos_RPi/wiki/Raspberry-Pi-4-and-Intel-RealSense-D435

```
sudo su
udevadm control --reload-rules && udevadm trigger
exit
```

pathの追加
~/.bashrcにexport LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATHを追加
```
source ~/.bashrc 
```

- まだインストールしてなかったパッケージのインストール
```
sudo apt-get install automake libtool
```

システム領域の拡張
```
sudo raspi-config
```
Advanced Options -> Expand Filesystemsを選択し、再起動

~~protobufのインストール~~

~~`cd ~`~~
~~git clone --depth=1 -b v3.10.0 https://github.com/google/protobuf.git~~
~~cd protobuf~~
~~./autogen.sh~~
~~./configure~~
~~make -j1~~
~~sudo make install~~
~~cd python~~
~~export LD_LIBRARY_PATH=../src/.libs~~
~~python3 setup.py build --cpp_implementation~~

~~error: invalid use of incomplete type ‘PyFrameObject’ {aka ‘struct _frame’}がでる。~~
~~->python 3.11以降'PyFrameObject'が使えないことによるエラーらしい~~

~~sudo ldconfig をして sudo make uninstall をして cd .. && rm -rf protobuf/~~\
protobuf,libtbb-devはインストールされていたので飛ばす

librealsenseのmake
```
cd ~/librealsense
mkdir  build  && cd build
cmake .. -DBUILD_EXAMPLES=true -DCMAKE_BUILD_TYPE=Release -DFORCE_LIBUVC=true
make -j1
sudo make install
```
realsense-viewerが起動できる

~~pyrealsenseのmake
(このときenvをactivateにするとwhich python3がenvの方を指してくれる)~~\

~~`cd ~/librealsense/build`
cmake .. -DBUILD_PYTHON_BINDINGS=bool:true -DPYTHON_EXECUTABLE=$(which python3)
make -j1
sudo make install~~\
~~`~/.bashrcにexport PYTHONPATH=$PYTHONPATH:/home/pi/NHK2024/NHK2024_R2_Raspi/env/lib/を追加`~~
~~`source ~/.bashrc`~~

~~openglのインストール(envをactivateにすること)~~
~~pip install pyopengl
pip install pyopengl_accelerate~~
raspi-configでのGL Driverの設定は無かったので飛ばした

~~NHK2024_Camera_Libraryのrs_sample.pyを実行すると
no module named pyrealsense2のエラーが出る~~

librealsenseのみmakeした状態で
~/.bashrcにexport PYTHONPATH=$PYTHONPATH:/usr/local/OFFを追加して
source ~/.bashrcを実行するとpyrealsense2が使えるようになった
