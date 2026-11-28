# Semantic Road Segmentation on IDD-Lite Dataset Using SegFormer

## Project Overview
This project focuses on semantic road segmentation using the IDD-Lite dataset and the SegFormer-B0 deep learning architecture.

The model performs pixel-level classification of road scenes into seven categories, helping autonomous driving systems understand complex road environments.

## Dataset
The project uses the IDD-Lite dataset, which contains road-scene images and corresponding segmentation masks.

### Classes
- Drivable Area
- Non-drivable Area
- Living Things
- Vehicles
- Roadside Objects
- Far Objects
- Sky

## Technologies Used
- Python
- PyTorch
- Hugging Face Transformers
- OpenCV
- Google Colab
- SegFormer-B0

## Methodology
1. Dataset extraction and preprocessing.
2. Pairing images with their corresponding segmentation masks.
3. Resizing images and masks to 256 × 256.
4. Loading the pretrained SegFormer-B0 model.
5. Fine-tuning the model for seven segmentation classes.
6. Using weighted Cross-Entropy Loss.
7. Evaluating performance using Intersection over Union (IoU).

## Results

| Class | IoU |
|---|---:|
| Drivable Area | 92.32% |
| Sky | 93.49% |
| Vehicles | 69.38% |
| Non-drivable Area | 40.33% |
| Mean IoU | 64.73% |

## Applications
- Autonomous driving
- Road scene understanding
- Driver assistance systems
- Intelligent transportation systems

## Future Improvements
- Improve segmentation of underrepresented classes.
- Experiment with larger SegFormer architectures.
- Increase training data and image resolution.
- Improve overall segmentation accuracy.

## Project Report
The formal project report is available in the `report` folder.

## Author
Add your name and college details here.
