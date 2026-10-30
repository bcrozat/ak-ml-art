import cv2
import subprocess
import streamlink
from ultralytics import YOLO

model = YOLO("models/yolo11n.pt") # n, s, m, l, xl

# TWITCH_URL = "https://www.twitch.tv/username"
# YOUTUBE_URL = "https://www.youtube.com/watch?v=VIDEO"
YOUTUBE_URL = "https://www.youtube.com/watch?v=vYRfQo6JMxc"
STREAM_URL = YOUTUBE_URL

import cv2
from ultralytics import YOLO

model = YOLO("models/yolo11n.pt") # n, s, m, l, xl

def get_res(stream_url, verbose=True):
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if verbose:
        print(f"Webcam stream resolution: {width}x{height}")
    return (width, height)

def get_res_from_frame(stream_url, verbose=True):
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return None
    ret, frame = cap.read()
    height, width= frame.shape[:2]  # get height and width from the frame shape
    cap.release()
    if verbose:
        print(f"Webcam stream resolution (from frame): {width}x{height}")
    return (width, height)

def get_stream_url(url):
    streams = streamlink.streams(url)
    print(f"Available streams: {streams.keys()}")
    if "best" in streams:
        return streams["best"].url
    else:
        Exception("Could not retrieve stream.")

def yolo_stream(stream_url, res=(1920, 1080), model=YOLO("models/yolo11n.pt"), record=False, output="output/yolo_stream           .avi"):
    stream_url = get_stream_url(STREAM_URL)  
    if record: # recording setup
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # select codec (also 'MJPG')
        out = cv2.VideoWriter(output, fourcc, 20.0, res)
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
                print("Failed to open stream.")
                return
    print("Streaming and running inference...")
    print("(Press 'q' to quit.)")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or frame could not be read.")
            break
        results = model(frame, conf=0.5) # run inference
        annotated_frame = results[0].plot() # annotate frame
        cv2.imshow("YOLO11+Sream", annotated_frame) # show stream
        # annotated_frame = cv2.resize(annotated_frame, res) # resize annotated frame to specified resolution
        print(results[0].boxes)
        if record:
            out.write(annotated_frame)  # save this frame to the file
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    if record:
        out.release()
    cv2.destroyAllWindows()

# stream_url = get_stream_url(STREAM_URL) 

stream_url = get_stream_url(STREAM_URL)

get_res(stream_url)

get_res_from_frame(stream_url)

yolo_stream(stream_url=STREAM_URL,
            res=(1920, 1080),
            model=model,
            record=True,
            output="output/yolo_stream.avi")