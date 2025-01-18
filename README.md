# Citrus Leaves Disease Detection Using YOLOv5
This repository contains all the resources for the project "Citrus Leaves Disease Detection Using YOLOv5." It includes the dataset, the Jupyter Notebook, and documentation of the methodology and results.

## Study Overview
This study uses YOLOv5 to detect and classify diseases in Calamansi and Dalandan leaves in the researchers' household. It involves creating a custom dataset with Roboflow, training a YOLOv5 model in Google Colab, and evaluating the results.

The study focused on detecting and classifying five categories of citrus leaf conditions:
- Healthy
- Greening
- Others
- Pest
- White spots

## Features

- Object detection using YOLOv5 for citrus leaf diseases.
- Custom dataset containing images of citrus leaves (dalandan and calamansi).
- Classes: Healthy, Greening, Others, Pests, and White Spots.
- Fine-tuned YOLOv5 model.

---

## Dataset

- **Source**: Locally captured images of citrus leaves (Calamansi and Dalandan). It has three sets for training, validation, and testing.
- **Classes**: 5 (as listed above)
- **Augmentation**: Performed using [Roboflow](https://universe.roboflow.com/aubsmin/citrus-leaves) (Version 10 under filename: final-new).
- Available for [Direct Download](https://universe.roboflow.com/ds/QyK7SUjhBR?key=UniBRbSYWh) or copy-paste this code for YOLOv5:

```
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="kJ98LX97AwuwrMipefBy")
project = rf.workspace("aubsmin").project("citrus-leaves")
version = project.version(10)
dataset = version.download("yolov5")
```

---

## Training the Model

1. Ensure the dataset is in the correct folder.
2. Run the script for the training and validation sets:

   ```bash
   %cd /content/yolov5/
    !python train.py \
      --img 640 \
      --batch 16 \
      --epochs 50 \
      --data /content/Citrus-Leaves-10/data.yaml \
      --weights yolov5s.pt \
      --name citrus_yolov5 \
      --cache \
   ```

---

## Results of the Study

### Training and Validation Sets:

Results:
  ```bash
  !find /content/yolov5/runs/train/ -name "results.png"
  
  from IPython.display import Image, display
  from utils.plots import plot_results  # plot results.txt as results.png
  
  # Display the results image
  display(Image(filename='/content/yolov5/runs/train/citrus_yolov5/results.png', width=1000))
  ```

![image](https://github.com/user-attachments/assets/7215c5d3-9048-42d5-b8b5-92e48d77bc77)

Ground Truth (Training):
```bash
  print("GROUND TRUTH AUGMENTED TRAINING DATA:")
  Image(filename='/content/yolov5/runs/train/citrus_yolov5/train_batch0.jpg', width=900)
  ```
![image](https://github.com/user-attachments/assets/d7ff298b-ad6f-4ea2-8b6e-285b1aa7f1a3)

Validation Batch Label:
  ```bash
  Image(filename='/content/yolov5/runs/train/citrus_yolov5/val_batch0_labels.jpg', width=900)
  ```
![image](https://github.com/user-attachments/assets/2d73570a-4fec-4bde-a2a3-8443132229e9)

### Test Set

Use model on Test Set:
```bash
  !python detect.py --weights /content/yolov5/runs/train/citrus_yolov5/weights/best.pt --img 640 --conf 0.25 --source /content/Citrus-Leaves-10/test/images
  ```

Result:
```bash
!python val.py --weights /content/yolov5/runs/train/citrus_yolov5/weights/best.pt --img 640 --data /content/Citrus-Leaves-10/data.yaml --task test
  ```
![image](https://github.com/user-attachments/assets/dc6be8ce-12f3-4df7-bdd6-0d0c64c9de17)
![image](https://github.com/user-attachments/assets/6c4e32cd-b8ed-49f2-b03a-89897054a5c1)
![image](https://github.com/user-attachments/assets/de72f6e3-5d73-45da-a694-6db1c145bbac)

Confusion Matrix:
![image](https://github.com/user-attachments/assets/9d5900b3-68d9-4aea-9399-5076368d8b83)

---

## Documentation

Refer to the `Citrus-Leaves-Disease-Detection.ipynb` file for step-by-step details on:

- Data preprocessing and augmentation.
- YOLOv5 model setup and training.
- Results analysis.

---

## Contributing

Contributions are welcome!

---

## Acknowledgments

- YOLOv5 by Ultralytics: [YOLOv5 GitHub](https://github.com/ultralytics/yolov5)
- Roboflow Universe for dataset hosting and augmentation tools.

---

## Contributors:

This is a Final Project requirement for the Professional Elective 5 course.

Aubrey Min M. Lasala
Britney G. Beligan
4th Year BS Computer Science students from New Era University, Quezon City
