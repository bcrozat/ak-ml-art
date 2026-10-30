import cv2
from ultralytics import YOLO

model = YOLO("models/yolo11n.pt") # n, s, m, l, xl

def get_res(stream_url, verbose=True):
    cap = cv2.VideoCapture(stream_url)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if verbose:
        print(f"Webcam stream resolution: {width}x{height}")
    return (width, height)

def get_res_from_frame(stream_url, verbose=True):
    cap = cv2.VideoCapture(stream_url)
    ret, frame = cap.read()
    height, width= frame.shape[:2]  # get height and width from the frame shape
    cap.release()
    if verbose:
        print(f"Webcam stream resolution (from frame): {width}x{height}")
    return (width, height)

def yolo_webcam(stream_url, res=(500, 500), model=YOLO("models/yolo11n.pt"), record=False, output="output/yolo_webcam.avi"):
    if record: # recording setup
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # select codec (also 'MJPG')
        out = cv2.VideoWriter(output, fourcc, 20.0, res)
    cap = cv2.VideoCapture(stream_url)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam stream.")
            break
        results = model(frame, conf=0.5)
        annotated_frame = results[0].plot()
        # annotated_frame = cv2.resize(annotated_frame, res) # resize annotated frame to specified resolution
        print(results[0].boxes)
        if record:
            out.write(annotated_frame)  # Save this frame to the file
        cv2.imshow("YOLO11+Webcam", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    if record:
        out.release()
    cv2.destroyAllWindows()

get_res(stream_url = "http://80.66.36.54/cgi-bin/faststream.jpg?stream=half&fps=15&rand=COUNTER")

get_res_from_frame(stream_url = "http://80.66.36.54/cgi-bin/faststream.jpg?stream=half&fps=15&rand=COUNTER")

yolo_webcam(stream_url = "http://80.66.36.54/cgi-bin/faststream.jpg?stream=half&fps=15&rand=COUNTER",
            res=(1280, 960),
            model=model,
            record=True,
            output="output/yolo_webcam.avi")