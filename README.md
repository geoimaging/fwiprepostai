# _FWIPrePostAI_ - Shear Wave Velocity Prediction with data-driven Full Waveform Inversion (FWI) models.

> Sanish Bhochhibhoya and Joseph P. Vantassel [jpvantassel.com](https://www.jpvantassel.com/)

## Table of Contents

-   [About _FWIPrePostAI_](#about-fwiprepostai)
-   [Getting Started](#getting-started)
  - [Installation Guide for _FWIPrePostAI_](#installation-guide-for-fwiprepostai)
  - [Using _FWIPrePostAI_](#using-fwiprepostai)
  - [Provided Datasets in `data` directory.](#provided-datasets-in-data-directory)


## About _FWIPrePostAI_

`FWIPrePostAI` is a Python-based toolkit designed to predict shear wave velocity profiles from waveform datasets using pre-trained artificial intelligence (AI) models. The repository includes tools for working with both simulated and real-field data, and features a pre-trained model developed as part of the study by Vantassel and Bhochhibhoya (2025).

This work facilitates rapid and data-driven full waveform inversion (FWI) for geotechnical applications, lowering the technical barrier for non-expert users while maintaining robust predictive performance.

If you use these tools in your research or consulting, we ask you please cite the following:

> Vantassel, J. P. and Bhochhibhoya, S. (2025). “Toward More-Robust, AI-enabled Subsurface Seismic Imaging for Geotechnical Applications.” Computers and Geotechnics. [In-Review]

## Getting Started

### Installation Guide for _FWIPrePostAI_

1. Ensure Python is Installed.\
   Verify that Python version 3.10 or later is installed on your system. If not, please follow the detailed installation instructions provided [here](https://jpvantassel.github.io/python3-course/#/intro/installing_python).

2. Download the Repository.\
Obtain a local copy of the FWIPrePostAI repository, either by downloading the repository as a .zip file and extracting its contents, or cloning the repository via Git.

3. Install Required Python Packages.\
Navigate to the repository directory and install the necessary Python dependencies using the following command:\
python -m pip install -r requirements.txt\
For users unfamiliar with pip, a comprehensive tutorial is available [here](https://jpvantassel.github.io/python3-course/#/intro/pip).

4. Download the `data` Folder.\
Download the `data` directory from the provided [Google Drive Link](https://drive.google.com/drive/folders/1vjlRO7hp6bSu-4H1NwrhKH8qGjo5GSHz?usp=sharing).
Ensure that the downloaded `data`, `fwiprepostai`, and `.pynb files` are all placed in the same parent directory. More details on the contents of `data` folder in the later section.

### Using _FWIPrePostAI_

1.  Launch the Notebooks.\
Open either `Data_Driven_FWI_Prediction_field_data.ipynb` or `Data_Driven_FWI_Prediction_simulated.ipynb` to get started with the prediction models. `Data_Driven_FWI_Prediction_field_data.ipynb` - Notebook to perform a prediction on field data using Vantassel and Bhochhibhoya (2025) model. \
`Data_Driven_FWI_Prediction_simulated.ipynb` - Notebook to perform a prediction on the simulated (testing) dataset using Vantassel and Bhochhibhoya (2025) model.

2. Explore the Notebooks.\
Experiment with the notebooks with different provided dataset. Once comfortable, feel free to apply Vantassel and Bhochhibhoya (2025) model to any field/simulated dataset of your own. 

### Provided Datasets in `data` directory.
The `data` directory includes the following files and subdirectories essential for running the notebooks and reproducing results:
1. `model_weights/VantasselAndBhochhibhoya2025.weights.h5` contains the pretrained model weights corresponding to the Vantassel and Bhochhibhoya (2025) architecture. This file is compatible with the model class defined in `fwiprepostai/data_driven_fwi_models.py`.
2. `waveforms\generated_data\samples_500_116.h5` contains a subset (500 samples) out of 10000 testing dataset that was used during the testing of the Vantassel and Bhochhibhoya (2025) model.\ __Note:__ Samples with indices 269, 87, 252 were used in Figure 4 of the Vantassel and Bhochhibhoya (2025) paper.
3. `waveforms\real_field_data_numpy` contains two NumPy arrays that represents raw (drillfield_June28_raw.npy), and field-corrected (drillfield_June28_lisousi.npy) version of the real-field waveform data. \ Applying the correction procedure proposed by Forbriger et al. (2014) to raw version reproduces lisousi version of the real-field waveformd data.
__Recommendation:__ When using real-field data, apply the correction procedure described in Forbriger et al. (2014) 
4. `waveforms\real_field_data_geode` contains ten raw real-field waveform dataset obtained during field experiments at the Drillfield, Blacksburg, VA. Stacking these ten datasets reproduces drillfield_June28_raw.npy, as mentioned above. 

