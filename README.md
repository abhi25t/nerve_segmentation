# Brachial Plexus Nerve Segmentation

Code for the accompanying paper, "*Automated Real Time Delineation of Supraclavicular Brachial Plexus in Neck Ultrasonography Videos: A Deep Learning Approach*" 

![high](./blob/main/other/high_gain.gif)

### Dataset
- Available in the `data` folder
- If you wish to change the location of data folder, edit the `DATA_DIR` path in `configurations.ini`

### Installation
Implemented for Python 3, with following dependencies:
- PyTorch
- Torchvision
- OpenCV
- PIL
- sklearn
- Numpy
- Pandas
- tqdm

### Usage
See `segmentation.ipynb` jupyter notebook
- More configurations and hyperparameters can be modified in `configurations.ini` 
