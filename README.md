# Machine learning for artists
AK, W2026

Baptiste Crozat
Art & Science

## Setup

```
uv init ak-ml-art
uv sync
uv add opencv-python
uv add ultralytics
uv add ffmpeg-python streamlink # for twich & youtube
```

## Notes
- clone full repo + pull each time a new file is added (cloning only a specific file or folder is convoluted)
- use YOLOv8
- adapt for bubble recognition & count
- run in real-time
- four character code (fourcc): an identifier for a video codec
- using yolo set parameters model(
    conf=0.1 # confidence of prediction
    iou=0.2 # intersection over union overlap
    device="cuda" # "cuda" for nvidia gpus or "mps" for apple silicon chips (>=m1)
    max_det=1 # limit number of detection(s)
    classes=[67] # set class to detect (ex: 67 for cell phone)
    )
- choose among different yolo models
    -YOLO("yolo11n.pt"): object detection model
    -seg YOLO("yolo11n-seg.pt"): segmentation model
    -pose YOLO("yolo11n-pose.pt"): joints detection model
    -cls YOLO("yolo11n-cls.pt"): class detection model

## References
- https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml
- https://painlessprogramming.com/work-with-camera-feed-in-python/
- Ren, Y., Wang, Y., Yang, B., Zheng, S., & Chai, M. (2026). A review of deep learning-based bubble recognition methods. Flow Measurement and Instrumentation, 103204. https://www.sciencedirect.com/science/article/abs/pii/S095559862600018X

#### Ren et al., 2026

> Representative efforts include the development of lightweight detection models such as YOLOv8 combined with tracking strategies for real-time bubble analysis [19], the enhancement of model performance in ultra dense bubbly flows [20], and the mitigation of data scarcity through domain adaptation and generative modeling techniques such as CycleGAN which enables image translation without paired training samples [21]. (Ren et al., 2026)
