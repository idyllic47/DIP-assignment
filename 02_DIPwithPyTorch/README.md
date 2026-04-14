# Assignment 2 - DIP with PyTorch

## This repository is HeJiaxuan's implementation of Assignment_02 of DIP.

---

### 1. Implement Poisson Image Editing with PyTorch.
## Running

To run Poisson Image Editing, run:

```basic
python run_blending_gradio.py
```

## Results 
<img src="pic/process.png" alt="alt text" width="800">

<img src="pic/output.webp" alt="alt text" width="800">

---

### 2. Pix2Pix implementation.
## Running
Run:
```bash
bash download_cityscapes_dataset.sh
python train.py
```

The code will train the model on the [Cityscapes Dataset](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/cityscapes.tar.gz).

## Results 
The results are in train_results and val_results. Here are the visualization of comparison.
epoch = 0
<img src="Pix2Pix/visualization/epoch_0_comparison.png" alt="alt text" width="800">

epoch = 150
<img src="Pix2Pix/visualization/epoch_150_comparison.png" alt="alt text" width="800">

epoch = 295
<img src="Pix2Pix/visualization/epoch_295_comparison.png" alt="alt text" width="800">

The gif
<img src="Pix2Pix/visualization/training_progress.gif" alt="alt text" width="800">

---

### Requirements:
To install requirements:

```setup
python -m pip install -r requirements.txt
```

