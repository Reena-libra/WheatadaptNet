# WheatadaptNet
New segmentation framework
## Running WheatAdaptNet Experiments

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare the dataset
Download the dataset from:
https://doi.org/10.25416/edgehill.30228148

Configure the dataset paths as required by the implementation.

### 3. Train the model
```bash
python train_deform_off_k_new.py
```

### 4. Test the trained model
```bash
python test_wheat_deformoff.py
```

Ensure that the dataset paths, model checkpoint and experimental parameters are configured correctly before running the scripts.
