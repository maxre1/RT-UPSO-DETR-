import warnings, os

warnings.filterwarnings('ignore')
from ultralytics import RTDETR



if __name__ == '__main__':
    model = RTDETR("E:/data/RTDETR-main/new/RTDETR-main/runs/train/weights/last.pt")

    model.train(data="data.yaml",
                cache=False,
                imgsz=640,
                epochs=40,
                batch=4, 
                workers=4, 
                project='runs',
                name='train',
                )