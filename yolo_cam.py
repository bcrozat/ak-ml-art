import cv2
from ultralytics import YOLO

model = YOLO("models/yolo11n.pt") # n, s, m, l, xl

def yolo_cam(cam_n=0, res=(640, 840), model=YOLO("models/yolo11n.pt"), record=False, output="output/yolo_webcam.avi"):
    if record: # recording setup
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # select codec (also 'MJPG')
        out = cv2.VideoWriter(output, fourcc, 20.0, res)
    cap = cv2.VideoCapture(cam_n)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from cam.")
            break
        results = model(frame, conf=0.5)
        annotated_frame = results[0].plot()
        annotated_frame = cv2.resize(annotated_frame, res) # resize annotated frame to specified resolution
        if record:
            out.write(annotated_frame)  # Save this frame to the file
        cv2.imshow("YOLO11+Cam", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    if record:
        out.release()
    cv2.destroyAllWindows()

yolo_cam(cam_n=1,
            # res=(640, 480), # cam_n=0
            res=(1280, 720), # logitech c505 cam_n=0
            model=model,
            record=True,
            output="output/yolo_cam.avi")