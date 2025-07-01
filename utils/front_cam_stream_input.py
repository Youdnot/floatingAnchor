# Originally from: hl2ss/viewer/client_stream_pv.py
# Modified for use as utility for getting streaming data source from HoloLens 2

#------------------------------------------------------------------------------
# This script receives video from the HoloLens front RGB camera and plays it.
# The camera supports various resolutions and framerates. See
# https://github.com/jdibenes/hl2ss/blob/main/etc/pv_configurations.txt
# for a list of supported formats. The default configuration is 1080p 30 FPS. 
# The stream supports three operating modes: 0) video, 1) video + camera pose, 
# 2) query calibration (single transfer).
# Press esc to stop.
#------------------------------------------------------------------------------

from pynput import keyboard

import cv2
from src.hl2ss.viewer import hl2ss, hl2ss_imshow, hl2ss_lnm, hl2ss_utilities

# Settings --------------------------------------------------------------------

# HoloLens address
# host = "192.168.1.7"
# host = "192.168.137.230"
host = "169.254.10.1"

# Operating mode
# 0: video
# 1: video + camera pose
# 2: query calibration (single transfer)
mode = hl2ss.StreamMode.MODE_1

# Enable Mixed Reality Capture (Holograms)
enable_mrc = False

# Enable Shared Capture
# If another program is already using the PV camera, you can still stream it by
# enabling shared mode, however you cannot change the resolution and framerate
# shared = False
shared = True

# Camera parameters
# Ignored in shared mode
width     = 1920
height    = 1080
framerate = 5    # 30

# Video encoding profile and bitrate (None = default)
profile = hl2ss.VideoProfile.H265_MAIN
bitrate = None

# Decoded format
# Options include:
# 'bgr24'
# 'rgb24'
# 'bgra'
# 'rgba'
# 'gray8'
decoded_format = 'bgr24'

#------------------------------------------------------------------------------

class FrontCamStreamInput:
    """
    用于连接HoloLens2前置摄像头并获取视频流的类。
    提供open、read、close方法。
    """
    def __init__(self,
                 host="169.254.10.1",
                 mode=None,
                 enable_mrc=False,
                 shared=True,
                 width=1920,
                 height=1080,
                 framerate=5,
                 profile=None,
                 bitrate=None,
                 decoded_format='bgr24'):
        from src.hl2ss.viewer import hl2ss
        self.host = host
        self.mode = mode if mode is not None else hl2ss.StreamMode.MODE_1
        self.enable_mrc = enable_mrc
        self.shared = shared
        self.width = width
        self.height = height
        self.framerate = framerate
        self.profile = profile if profile is not None else hl2ss.VideoProfile.H265_MAIN
        self.bitrate = bitrate
        self.decoded_format = decoded_format
        self._client = None
        self._listener = None
        self._opened = False
        self._hl2ss = hl2ss
        from src.hl2ss.viewer import hl2ss_lnm, hl2ss_utilities
        self._hl2ss_lnm = hl2ss_lnm
        self._hl2ss_utilities = hl2ss_utilities

    def open(self):
        # 启动子系统
        self._hl2ss_lnm.start_subsystem_pv(
            self.host,
            self._hl2ss.StreamPort.PERSONAL_VIDEO,
            enable_mrc=self.enable_mrc,
            shared=self.shared
        )
        # 只支持mode 0/1视频流
        if self.mode == self._hl2ss.StreamMode.MODE_2:
            raise NotImplementedError("Mode 2 (calibration) 暂不支持流式读取")\
            
        self._listener = self._hl2ss_utilities.key_listener(keyboard.Key.esc)
        self._listener.open()

        self._client = self._hl2ss_lnm.rx_pv(
            self.host,
            self._hl2ss.StreamPort.PERSONAL_VIDEO,
            mode=self.mode,
            width=self.width,
            height=self.height,
            framerate=self.framerate,
            profile=self.profile,
            bitrate=self.bitrate,
            decoded_format=self.decoded_format
        )
        self._client.open()
        self._opened = True

    def read(self):
        """获取下一帧图像，返回numpy数组（BGR）或None。"""
        if not self._opened:
            raise RuntimeError("请先调用open()方法")
        data = self._client.get_next_packet()
        # 返回BGR格式图像
        return data.payload.image

    def close(self):
        if self._client is not None:
            self._client.close()
        if self._listener is not None:
            self._listener.close()
        self._hl2ss_lnm.stop_subsystem_pv(self.host, self._hl2ss.StreamPort.PERSONAL_VIDEO)
        self._opened = False

# 保留原有main示例（可选）
if __name__ == "__main__":
    cam = FrontCamStreamInput()
    cam.open()
    try:
        while True:
            frame = cam.read()
            if frame is None:
                break
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC退出
                break
    finally:
        cam.close()
        cv2.destroyAllWindows()

hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)


