# uv add kinect-next[all]

import cv2
import numpy as np

from kinect_next import (
    KinectSensor,
    StreamType,
)

def kinect():
    streams = StreamType.COLOR | StreamType.DEPTH

    with KinectSensor(streams=streams) as kinect:
        print("Kinect started. Press q to quit.")

        for frames in kinect.poll_frames():
            if frames.color is None or frames.depth is None:
                continue

            # 1. RGB camera feed (OpenCV uses BGR)
            color = frames.color.as_bgr()
            cv2.imshow("Kinect RGB", color)

            # 2. Grayscale depth map
            depth = frames.depth.to_normalized_uint8()
            cv2.imshow("Kinect Depth", depth)

            # 3. Colorized depth map
            depth_color = cv2.applyColorMap(
                depth, cv2.COLORMAP_TURBO
            )
            cv2.imshow("Kinect Depth Color", depth_color)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # exit if 'q' pressed
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    kinect()
