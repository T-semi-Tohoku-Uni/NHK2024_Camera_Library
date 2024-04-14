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
# ラズパイでRealsenseを使えるようにしたかった
- https://github.com/IntelRealSense/librealsense/blob/development/doc/installation.md#prerequisitesを参考にしたやつ

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
これだけ実行した
```
rm -rf librealsense/
```

- https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.mdを参考にしたやつ
```
sudo mkdir -p /etc/apt/keyrings
curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null
```
```
echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo `lsb_release -cs` main" | \
sudo tee /etc/apt/sources.list.d/librealsense.list
sudo apt-get update
```
ここでエラー
```
deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo bookworm main
Hit:1 http://deb.debian.org/debian bookworm InRelease
Get:2 http://deb.debian.org/debian-security bookworm-security InRelease [48.0 kB]
Hit:3 http://deb.debian.org/debian bookworm-updates InRelease                
Get:4 http://archive.raspberrypi.com/debian bookworm InRelease [23.6 kB]         
Err:5 https://librealsense.intel.com/Debian/apt-repo bookworm InRelease
  403  Forbidden [IP: 13.227.62.101 443]
Reading package lists... Done                            
E: Repository 'http://deb.debian.org/debian-security bookworm-security InRelease' changed its 'Label' value from 'Debian' to 'Debian-Security'
N: Repository 'http://deb.debian.org/debian-security bookworm-security InRelease' changed its 'Version' value from '12-updates' to '12'
N: Repository 'http://deb.debian.org/debian-security bookworm-security InRelease' changed its 'Suite' value from 'stable-updates' to 'stable-security'
E: Repository 'http://deb.debian.org/debian-security bookworm-security InRelease' changed its 'Codename' value from 'bookworm-updates' to 'bookworm-security'
N: This must be accepted explicitly before updates for this repository can be applied. See apt-secure(8) manpage for details.
E: Failed to fetch https://librealsense.intel.com/Debian/apt-repo/dists/bookworm/InRelease  403  Forbidden [IP: 13.227.62.101 443]
E: The repository 'https://librealsense.intel.com/Debian/apt-repo bookworm InRelease' is not signed.
N: Updating from such a repository can't be done securely, and is therefore disabled by default.
N: See apt-secure(8) manpage for repository creation and user configuration details.
E: Repository 'http://archive.raspberrypi.com/debian bookworm InRelease' changed its 'Origin' value from 'Debian' to 'Raspberry Pi Foundation'
E: Repository 'http://archive.raspberrypi.com/debian bookworm InRelease' changed its 'Label' value from 'Debian' to 'Raspberry Pi Foundation'
N: Repository 'http://archive.raspberrypi.com/debian bookworm InRelease' changed its 'Version' value from '12-updates' to ''
N: Repository 'http://archive.raspberrypi.com/debian bookworm InRelease' changed its 'Suite' value from 'stable-updates' to 'stable'
E: Repository 'http://archive.raspberrypi.com/debian bookworm InRelease' changed its 'Codename' value from 'bookworm-updates' to 'bookworm'
N: This must be accepted explicitly before updates for this repository can be applied. See apt-secure(8) manpage for details.
(env) pi@tsemiR2:~/NHK2024 $ 
```


