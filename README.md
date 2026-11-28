# AgriTech Tomato Disease Classifier

This project trains a 10-class tomato leaf disease classifier with transfer learning using torchvision's pretrained MobileNetV2. The target is at least 95% accuracy on the held-out PlantVillage test split; the script reports the measured score rather than assuming it will be met.

## Data

`src/prepare_data.py` downloads the public [PlantVillage dataset mirror](https://huggingface.co/datasets/BrandonFors/Plant-Diseases-PlantVillage-Dataset) from Hugging Face (about 850 MB downloaded). It keeps the 10 tomato classes, makes a stratified validation split from the published training split, and preserves the published test split for final evaluation. The downloaded source is parquet; the prepared local dataset is converted to RGB JPEG files in `data/processed/plantvillage/{train,validation,test}/<class-name>/`, compatible with torchvision `ImageFolder`. A `manifest.json` records class counts and the split seed.

The dataset mirror identifies its license as CC0-1.0 and cites the original PlantVillage work. Review the dataset card and original source terms before redistributing the downloaded images.

## Run on Windows

Use a Python version supported by PyTorch and install the project dependencies into a virtual environment:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\prepare_data.py
.\.venv\Scripts\python.exe src\train.py --batch-size 16
```

The training script uses CUDA when available, trains the new classifier head first, then fine-tunes the final MobileNetV2 feature blocks. The default batch size is conservative for GPUs with 6 GB of memory. Use `--head-epochs`, `--finetune-epochs`, and `--workers` to adjust the run.

## Google Colab

This workstation's Windows Application Control currently blocks native PyTorch/Matplotlib DLLs. To train before that policy is changed, upload the project source to a Colab notebook, select a GPU runtime, and run:

```python
!pip install datasets Pillow scikit-learn matplotlib tqdm
!python src/prepare_data.py
!python src/train.py --batch-size 16
```

Colab downloads the dataset and ImageNet weights when the preparation and training scripts run. The prepared data in this local workspace is not required in Colab.

## Outputs

- `models/mobilenetv2_tomato.pt`: best-validation model checkpoint and class names
- `outputs/test_metrics.json`: held-out accuracy, macro F1, loss, and target status
- `outputs/training_log.csv`: flushed to disk after every epoch, with train/validation loss, accuracy, and error
- `outputs/graphs/training_progress.png`: refreshed after every epoch with loss, accuracy, and error curves
- `outputs/graphs/epoch_XX.png`: graph snapshot saved after each completed epoch
- `outputs/classification_report.txt`: per-class precision, recall, and F1
- `outputs/confusion_matrix.png`: held-out confusion matrix
- `outputs/history.json`: training and validation metrics for each epoch

PlantVillage contains clean, lab-style leaf images. A high score on this dataset does not establish accuracy on real farm photos; PlantDoc or locally collected field images would be a separate generalization test.

## Loss and Error

The model is optimized with mean categorical cross-entropy, where $z_i$ is the vector of class logits for example $i$, and $y_i$ is its correct class:

$$L = -\frac{1}{N}\sum_{i=1}^{N}\log\left(\operatorname{softmax}(z_i)_{y_i}\right)$$

Classification error is the fraction of incorrect predictions: $E = 1 - \text{accuracy}$. The live epoch line and `training_log.csv` report these for both training and validation data; test error is reported once after training.