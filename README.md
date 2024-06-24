# RATS: Robust Automated Tracking and Segmentation of Similar Instances
This is the official repository for "RATS: Robust Automated Tracking and Segmentation of Similar Instances" presented at 30th International Conference on Artificial Neural Networks (ICANN).

*Note: The repository includes updates to the tracking approach proposed in [Gelencsér-Horváth et al.](https://github.com/g-h-anna/sepaRats) Additionally, it has been extended to process full-length videos and detect behaviors.*

## Setup

### Requirements
- **GPU**: Modern GPU with at least 4 GB of VRAM and CUDA 11 (or 10) capability.
  - **Training Mask R-CNN**: GPU with at least 8 GB of VRAM.
- **RAM**: Minimum 8 GB (16 GB on Windows).
- **CPU**: At least 8 cores.

The software has been tested on Ubuntu and Windows 10.

### Before You Start
Ensure you have the following installed:

- [git](https://git-scm.com/downloads)
- [Cuda 11.1.1](https://developer.nvidia.com/cuda-11.1.1-download-archive) (Note: Cuda 10 is also supported)
- [Anaconda](https://docs.anaconda.com/anaconda/install/)

For Windows 10, additionally install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with the "Desktop development with C++" packages.

### Installation
Open Anaconda Prompt and navigate to the desired folder to clone the repository. Execute the following commands:

```bash
git clone https://github.com/lkopi/rat_tracking.git
cd rat_tracking
```

Create and set up the conda environment:

```bash
conda create -n rats python=3.7 anaconda -y
conda activate rats

conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge -y
pip install cython opencv-python mmcv einops imageio luigi moviepy
```

Clone and install external dependencies:

- the Mask R-CNN implementation of [Detectron2](https://github.com/facebookresearch/detectron2),
- the [GMA](https://github.com/zacjiang/GMA.git) optical flow estimation method,
- the [wbAug](https://github.com/mahmoudnafifi/WB_color_augmenter.git) augmentation method,
- and the [sepaRats](https://github.com/g-h-anna/sepaRats) segmentation method (optional).

```bash
mkdir -p external
cd external

git clone https://github.com/zacjiang/GMA.git
git clone https://github.com/facebookresearch/detectron2.git
git clone https://github.com/mahmoudnafifi/WB_color_augmenter.git
# git clone https://github.com/g-h-anna/sepaRats.git

pip install -e detectron2
cd ..
```

### Optional (Windows)
To monitor GPU usage, add `nvidia-smi` to the system path. Typically located at: `"C:\Program Files\NVIDIA Corporation\NVSMI"`

To add it to the path:

1. Right-click on Computer → Properties → Advanced System Properties → Environment Variables.
2. Click on Path → Edit → New, then insert the folder path containing `nvidia-smi`.

### Troubleshooting
Common errors and solutions during installation on Windows:

- **Error**: Microsoft Visual C++ 14.0 or greater is required.
  - **Solution**: Install/update "Microsoft C++ Build Tools" as described in the [Before You Start](#Before-you-start) section.
  
- **Error**: CondaValueError: Malformed version string '~': invalid character(s).
  - **Solution**: Update conda: `conda upgrade -n base conda`
  
- **Error**: ImportError: DLL load failed.
  - **Solution**: Ensure pip and conda environments have matching packages:
  ```bash
  conda remove --force numpy scipy
  pip install numpy scipy
  ```

## Dataset
Download the [dataset](https://drive.google.com/drive/folders/1Z4wGbUKAiPZwgE6GMhnCT23OKYxOezTj?usp=sharing) into the `dataset` folder.

Structure:
- `dataset/rats.avi`: Raw video used for training and testing.
- `dataset/automatic_annotation`: Contains 100 hand-annotated samples for testing the annotation method with body part and keypoint labels.
- `dataset/tracking`: Contains 18x200 hand-annotated frames for testing the tracking method.

## Models
Download the [models](https://drive.google.com/drive/folders/1cuFypvxaPn65Zgbo8mrmVPjC525XcA18?usp=sharing) into the `models` folder.

Pre-trained models:
- `models/instace_finetuned`: Instance segmentation model, pre-trained on synthetic samples and fine-tuned on 18x200 hand-annotated samples.
- `models/instace`: Instance segmentation model, pre-trained on synthetic samples.
- `models/bodypart`: Body part segmentation model, pre-trained on synthetic samples.
- `models/keypoint`: Keypoint detection and instance segmentation model, pre-trained on synthetic samples.
- `models/separats/dexined`: Instance segmentation model proposed in [Gelecsér-Horváth et al.](https://github.com/g-h-anna/sepaRats), pre-trained on synthetic samples.

## Usage
The software consists of several pipelines for various functionalities. For more information on each pipeline, run:

```bash
python -m luigi --module pipeline <PipelineName> --help
```

Main scripts can be run individually. For required parameters, use:

```bash
python <script>.py --help
```

To run the overall pipeline:

```bash
python -m luigi --module pipeline MatchSequences \
  --PrepareVideo-video "path/to/video" \
  --PrepareVideo-crop '[x1,y1,x2,y2]' \
  --PrepareVideo-n-sequences 16 \
  --Predict-model-dir "path/to/model" \
  --Predict-batch-size 8 \
  --OpticalFlow-n-processes 4 \
  --PropagatePredictions-chunksize 100 \
  --BehaviorEstimation-config-file config.arch --local-schedule
```

### Tips
Not all parts of the pipeline require intensive GPU or CPU calculations. To increase speed of processing large datasets, run parts of the software in parallel.

*Example: While PropagatePrediction and BehaviorDetection are running on the current video, start running Predict and OpticalFlow on the next video in a new terminal.*

### Prediction and Propagation
To run the tracking method on a given video:

```bash
python -m luigi --module pipeline PropagatePredictions \
    --PrepareVideo-video "path/to/video" \
    --PrepareVideo-crop '[x1,y1,x2,y2]' \
    --Predict-model-dir "path/to/model" \
    --local-schedule
```

Evaluate the tracking method:

```bash
python -m luigi --module pipeline Evaluate \
    --PrepareVideo-video "path/to/test_set" \
    --PrepareVideo-crop '[h1,w1,h2,w2]' \
    --Predict-model-dir "path/to/model" \
    --PropagatePredictions-overlap-thrs 0.1 \
    --gt-dir "path/to/gt_masks" \
    --local-schedule
```

### Pre-training and Fine-tuning
To pre-train the Mask R-CNN network:

```bash
python -m luigi --module pipeline PreTrain \
    --ForegroundSegmentation-video "path/to/video" \
    --ForegroundSegmentation-end-frame 15020 \
    --ForegroundSegmentation-crop '[h1,w1,h2,w2]' \
    --n-iter 50000 \
    --local-schedule
```

Fine-tune the model:

```bash
python -m luigi --module pipeline TuneModel \
    --annot-dir "path/to/annotations" \
    --img-dir "path/to/images" \
    --model-dir "path/to/model" \
    --local-schedule
```

### Behavior Detection
To run behavior estimation:

```bash
python -m luigi --module pipeline BehaviorEstimation \
    --PrepareVideo-video "path/to/video" \
    --PrepareVideo-crop '[h1,w1,h2,w2]' \
    --Predict-model-dir "path/to/model" \
    --config-file "template.arch" \
    --local-schedule
```

For behavior evaluation:

```bash
python -m luigi --module pipeline BehaviorEvaluation \
    --PrepareVideo-video "path/to/test_set" \
    --Predict-model-dir "path/to/model" \
    --MatchPredictions-guide-dir "path/to/gt_instance_segmentation" \
    --config-file "template.arch" \
    --gt-dir "path/to/gt_behavior_annotations" \
    --local-schedule
```

### Training
Using the prepared dataset, a Mask R-CNN can be trained. This repository uses [Detectron2](https://github.com/facebookresearch/detectron2).

## Citation
If you use this codebase, please cite the following papers:
```
@inbook{Kopacsi2021,
  title = {RATS: Robust Automated Tracking and Segmentation of Similar Instances},
  ISBN = {9783030863654},
  ISSN = {1611-3349},
  url = {http://dx.doi.org/10.1007/978-3-030-86365-4_41},
  DOI = {10.1007/978-3-030-86365-4_41},
  booktitle = {Artificial Neural Networks and Machine Learning – ICANN 2021},
  publisher = {Springer International Publishing},
  author = {Kopácsi,  László and Dobolyi,  Árpád and Fóthi,  Áron and Keller,  Dávid and Varga,  Viktor and Lőrincz,  András},
  year = {2021},
  pages = {507–518}
}

@misc{Kopacsi2024,
  doi = {10.48550/ARXIV.2405.04650},
  url = {https://arxiv.org/abs/2405.04650},
  author = {Kopácsi,  László and Fóthi,  Áron and Lőrincz,  András},
  keywords = {Computer Vision and Pattern Recognition (cs.CV),  Artificial Intelligence (cs.AI),  FOS: Computer and information sciences,  FOS: Computer and information sciences},
  title = {A Self-Supervised Method for Body Part Segmentation and Keypoint Detection of Rat Images},
  publisher = {arXiv},
  year = {2024},
  copyright = {Creative Commons Attribution 4.0 International}
}

@article{GelencserHorvath2022,
  title = {Tracking Highly Similar Rat Instances under Heavy Occlusions: An Unsupervised Deep Generative Pipeline},
  volume = {8},
  ISSN = {2313-433X},
  url = {http://dx.doi.org/10.3390/jimaging8040109},
  DOI = {10.3390/jimaging8040109},
  number = {4},
  journal = {Journal of Imaging},
  publisher = {MDPI AG},
  author = {Gelencsér-Horváth,  Anna and Kopácsi,  László and Varga,  Viktor and Keller,  Dávid and Dobolyi,  Árpád and Karacs,  Kristóf and Lőrincz,  András},
  year = {2022},
  month = apr,
  pages = {109}
}
```