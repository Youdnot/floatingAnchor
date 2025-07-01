import copy
import os

import cv2
import numpy as np
# import supervision as sv
import torch
from PIL import Image
from src.grounding_sam2.sam2.build_sam import build_sam2, build_sam2_video_predictor
from src.grounding_sam2.sam2.sam2_image_predictor import SAM2ImagePredictor
# from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
# from src.grounding_sam2.utils.common_utils import CommonUtils
from src.grounding_sam2.utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
from src.grounding_sam2.utils.track_utils import sample_points_from_masks
from src.grounding_sam2.utils.video_utils import create_video_from_images

# Setup environment
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from utils.camera_tracking import GroundingDinoPredictor, SAM2ImageSegmentor, IncrementalObjectTracker
from utils.front_cam_stream_input import FrontCamStreamInput

import os

import cv2
import torch
# from src.grounding_sam2.utils.common_utils import CommonUtils

from pynput import keyboard

import cv2
from src.hl2ss.viewer import hl2ss, hl2ss_imshow, hl2ss_lnm, hl2ss_utilities

def main():
    # Detection parameter settings
    output_dir = "./outputs"
    prompt_text = "person."
    detection_interval = 20
    max_frames = 300  # Maximum number of frames to process (prevents infinite loop)

    os.makedirs(output_dir, exist_ok=True)

    # Initialize the object tracker
    tracker = IncrementalObjectTracker(
        grounding_model_id="IDEA-Research/grounding-dino-tiny",
        sam2_model_cfg= "configs/sam2.1/sam2.1_hiera_l.yaml",   # here seems to be the issue of installing the sam package as dependency?
        # sam2_ckpt_path= "./checkpoints/sam2.1_hiera_large.pt",
        sam2_ckpt_path= "./src/grounding_sam2/checkpoints/sam2.1_hiera_large.pt",
        device="cuda",
        prompt_text=prompt_text,
        detection_interval=detection_interval,
    )
    tracker.set_prompt("person. obstacle. viehicle. car. bus. truck. desk. table. ground. road. sidewalk. wall.")

    # 优先尝试用HoloLens2前置摄像头流
    cam = None
    use_hololens = True
    try:
        cam = FrontCamStreamInput()
        cam.open()
        print("[Info] Using HoloLens2 front camera stream.")
    except Exception as e:
        print(f"[Warning] Failed to open HoloLens2 stream: {e}\nFallback to local camera.")
        use_hololens = False
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            print("[Error] Cannot open local camera.")
            return
        print("[Info] Using local camera.")

    print("[Info] Camera opened. Press 'q' to quit.")
    frame_idx = 0

    try:
        while True:
            if use_hololens:
                frame = cam.read()
                if frame is None:
                    print(f"[Warning] Failed to get frame from HoloLens2 stream.")
                    break
            else:
                ret, frame = cam.read()
                if not ret:
                    print("[Warning] Failed to capture frame.")
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            print(f"[Frame {frame_idx}] Processing live frame...")
            process_image = tracker.add_image(frame)

            if process_image is None or not isinstance(process_image, np.ndarray):
                print(f"[Warning] Skipped frame {frame_idx} due to empty result.")
                frame_idx += 1
                continue

            # process_image_bgr = cv2.cvtColor(process_image, cv2.COLOR_RGB2BGR)
            # cv2.imshow("Live Inference", process_image_bgr)

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     print("[Info] Quit signal received.")
            #     break

            tracker.save_current_state(output_dir=output_dir, raw_image=frame)
            frame_idx += 1

            if frame_idx >= max_frames:
                print(f"[Info] Reached max_frames {max_frames}. Stopping.")
                break
    except KeyboardInterrupt:
        print("[Info] Interrupted by user (Ctrl+C).")
    finally:
        if use_hololens and cam is not None:
            cam.close()
        elif cam is not None:
            cam.release()
        cv2.destroyAllWindows()
        print("[Done] Live inference complete.")

if __name__ == "__main__":
    main()