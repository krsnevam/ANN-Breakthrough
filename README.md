# Smart Mobility & Road Safety Using AI and Neural Networks

## 1. Project Overview

This project uses Artificial Intelligence and Deep Learning to understand Indian road scenes through **Semantic Segmentation**.

The system identifies different parts of a road image, such as drivable areas, non-drivable areas, vehicles, living things, roadside objects, far objects, and sky.

The main objective is to explore how AI-based road-scene understanding can support future smart mobility and road-safety applications.

## 2. Objectives

- Understand road scenes using deep learning.
- Identify different road elements at the pixel level.
- Train a semantic segmentation model using Indian road images.
- Evaluate model performance using Intersection over Union (IoU).
- Improve the identification of difficult classes through weighted loss and fine-tuning.

## 3. Dataset

**Dataset:** IDD-Lite (India Driving Dataset – Lite)

The dataset contains Indian road images and their corresponding segmentation masks.

A segmentation mask assigns a class label to each pixel in an image. It acts as the ground-truth reference during model training and evaluation.

### Dataset Details

| Dataset split | Image-mask pairs |
|---|---:|
| Training | 1,403 |
| Validation | 204 |

### Classes

The model identifies seven classes:

| Class ID | Class |
|---:|---|
| 0 | Drivable |
| 1 | Non-drivable |
| 2 | Living things |
| 3 | Vehicles |
| 4 | Roadside objects |
| 5 | Far objects |
| 6 | Sky |
| 255 | Ignore label |

Pixels marked with 255 are excluded from the training loss.

## 4. Model Used

**SegFormer-B0**

SegFormer is a deep-learning architecture designed for semantic segmentation.

This project uses a pretrained SegFormer-B0 model and adapts its classification head to identify seven road-scene classes.

### Transfer Learning

Transfer learning means using a model that has already learned useful image features and adapting it to a new task.

Instead of training the entire model from scratch, the pretrained model is fine-tuned for the IDD-Lite road-scene classes.

## 5. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming |
| Google Colab | Development environment |
| PyTorch | Model training |
| Hugging Face Transformers | SegFormer model |
| IDD-Lite | Dataset |
| Tesla T4 GPU | Accelerated training |

## 6. Project Workflow

The project follows these steps:

1. Load the IDD-Lite dataset.
2. Read the road images and their corresponding segmentation masks.
3. Resize images and masks to 256 × 256 pixels.
4. Load the pretrained SegFormer-B0 model.
5. Adapt the model to predict seven classes.
6. Train the model using the training dataset.
7. Evaluate predictions using IoU.
8. Identify classes with lower performance.
9. Apply weighted loss and additional fine-tuning.
10. Evaluate the final model again.

## 7. Training Configuration

| Parameter | Value |
|---|---|
| Model | SegFormer-B0 |
| Input image size | 256 × 256 |
| Number of classes | 7 |
| Batch size | 4 |
| Initial training epochs | 10 |
| Initial learning rate | 0.00005 |
| Initial loss function | Cross Entropy Loss |
| Optimizer | AdamW |
| Fine-tuning loss | Weighted Cross Entropy Loss |
| Fine-tuning learning rate | 0.00001, followed by 0.000003 |

The ignore label was set to 255 so that ignored pixels would not contribute to the loss.

## 8. Model Improvement

After the initial training, the model achieved a Mean IoU of 63.96%.

During evaluation, the non-drivable class had a relatively low IoU of 35.86%.

To improve this class, weighted loss was introduced. This gives greater importance to mistakes involving difficult classes during training.

The model was then fine-tuned further using a smaller learning rate.

This helped improve the non-drivable class and the overall Mean IoU.

## 9. Evaluation Metric: IoU

IoU stands for **Intersection over Union**.

It measures the overlap between the model's predicted region and the actual ground-truth region.

The formula is:

IoU = Intersection / Union

Mean IoU (mIoU) is calculated by averaging the IoU scores across the seven classes.

A higher IoU generally indicates better segmentation performance.

## 10. Results

### Overall Performance

| Metric | Initial Model | Fine-tuned Model |
|---|---:|---:|
| Mean IoU | 63.96% | **64.73%** |

### Class-wise Performance

| Class | Initial IoU | Final IoU |
|---|---:|---:|
| Drivable | 92.23% | 92.32% |
| Non-drivable | 35.86% | **40.33%** |
| Living things | 37.78% | 37.87% |
| Vehicles | 69.70% | 69.38% |
| Roadside objects | 44.78% | 46.10% |
| Far objects | 73.89% | 73.60% |
| Sky | 93.48% | 93.49% |

### Key Improvements

- Mean IoU increased from 63.96% to 64.73%.
- Non-drivable IoU increased from 35.86% to 40.33%.
- Roadside object IoU increased from 44.78% to 46.10%.

The overall Mean IoU improved by 0.77 percentage points.

The non-drivable class improved by 4.47 percentage points.

Some classes showed small decreases, so further training and experimentation may be useful.

## 11. Current Project Capabilities

The current implementation focuses on image-based semantic segmentation.

It can:

- Process road images.
- Predict pixel-level segmentation classes.
- Identify seven road-scene categories.
- Evaluate segmentation performance using IoU.

## 12. Future Scope

The project can be extended in several ways:

1. **Real-time road monitoring:** Process live dashcam or camera footage.
2. **Road hazard identification:** Identify obstacles and non-drivable areas.
3. **Driver safety alerts:** Provide warnings about potential road hazards.
4. **Smart vehicle applications:** Support road understanding for intelligent vehicles.
5. **Model improvement:** Use additional data and further fine-tuning to improve accuracy.

These are proposed future extensions and are not part of the current implementation.

## 13. Conclusion

This project demonstrates the use of AI and deep learning for understanding Indian road scenes.

Using the IDD-Lite dataset and a pretrained SegFormer-B0 model, seven road-scene classes were identified through semantic segmentation.

Weighted loss and additional fine-tuning helped improve the Mean IoU from 63.96% to 64.73%. The non-drivable class improved from 35.86% to 40.33%.

This work provides a foundation for future smart mobility and road-safety applications.

## 14. Project Status

**Completed:**
- IDD-Lite dataset preparation
- SegFormer-B0 model setup
- Initial model training
- Validation and IoU evaluation
- Weighted loss fine-tuning
- Performance comparison

**Future work
