import cv2

# WEBCAM STREAM
# find direct video stream (ex: right click "copy image link")
# add suffix after ip address for direct stream
# ! match the resolution of stream with resolution of recording !

def webcam(stream_url, res=(320, 240), record=False, output="webcam_output.avi"):
    if record: # recording setup
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # select codec (also 'MJPG')
        out = cv2.VideoWriter(output, fourcc, 20.0, res)
    cap = cv2.VideoCapture(stream_url)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam stream.")
            break
        if record:
            out.write(frame)  # Save this frame to the file
        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    if record:
        out.release()
    cv2.destroyAllWindows()

webcam(stream_url="http://202.142.10.11/nphMotionJpeg?Resolution=320x240&Quality=Standard",
       record=True,
       output="output/webcam_output.avi")