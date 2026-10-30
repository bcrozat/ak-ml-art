import cv2
from numpy import record

# CAM

def cam(cam_n=0, record=False, output="cam.avi"):
    if record:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # select codec (also 'MJPG')
        out = cv2.VideoWriter(output, fourcc, 20.0, (640, 480))
    cap = cv2.VideoCapture(cam_n)
    while True:
        ret, frame = cap.read() # ret: A boolean that is True if the frame was read successfully.
        if not ret:
            print("Error: Failed to read frame from cam.")
            break
        # frame = cv2.resize(frame, (100, 100)) 
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # opencv loads colors in BGR order instead of RGB
        if record:
            out.write(frame)  # Save this frame to the file
        cv2.imshow('Cam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): # exit if 'q' pressed
            break

cam(cam_n=0,
    record=True,
    output="output/cam.avi")