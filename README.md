# Fingerprint Image Quality

TODO
- add research paper link
- add link and code for implementation

## MiDeCon: Minutia Detection Confidence for Unsupervised and Accurate Minutia and Fingerprint Quality Assessment


IEEE International Joint Conference on Biometrics (IJCB) 2021

* [Research Paper](link to follow)
* [Implementation on FineNet](face_image_quality.py)


## Table of Contents 

- [Abstract](#abstract)
- [Key Points](#key-points)
- [Results](#results)
- [Installation](#installation)
- [Citing](#citing)
- [Acknowledgement](#acknowledgement)
- [License](#license)

## Abstract

<img src="CVPR_2020_teaser_1200x1200.gif" width="400" height="400" align="right">

The most determinant factor to achieve high accuracies in fingerprint recognition systems is the quality of its samples. Previous works mainly proposed supervised solutions based on image properties that neglects the minutiae extraction process, despite that most fingerprint recognition techniques are based on this extracted information. Consequently, a fingerprint image might be assigned as high quality even if the utilized minutia extractor produces unreliable information for recognition. In this work, we propose a novel concept of assessing minutia and fingerprint quality based on minutia detection confidence (MiDeCon). MiDeCon can be applied to an arbitrary deep learning based minutia extractor and does not require quality labels for learning. Moreover, the training stage of MiDeCon can be completely avoided if a pre-trained minutiae extraction neural network is available. We propose using the detection reliability of the extracted minutia as its quality indicator. By combining the highest minutia qualities, MeDiCon accurately determines the quality of a full fingerprint. Experiments are done on the publicly available databases of the FVC 2006 and compared against NIST’s widely-used fingerprint image quality software NFIQ1 and NFIQ2. The results demonstrate a significantly stronger quality assessment performance of the proposed MiDeCon-qualities as related works on both, minutia- and fingerprint-level. 

## Key Points
In contrast to previous works, the proposed approach:

- **Does not require quality labels for training** - Previous works that often rely on error-prone labelling mechanisms without a clear definition of quality. Our approach avoids the use of inaccurate quality labels by using the minutia detection confidence as a quality estimate. Moreover, the training state can be completely avoided if pre-trained minutiae extraction neural network trained with dropout is available.

- **Considers difficulties in the minutiae extraction** - Previous works estimates the quality of a fingerprint based on the properties of the image neglecting the minutiae extraction process. However, the extraction process might faces difficulties that is not considered in the image properties and thus, produce unreliable minutia information. Our solution defines quality through the prediction confidence of the extractor and thus, considers this problem.

- **Produces continuous quality values** - While previous works often categorizes the quality outputs in discrete categories (e.g. {good, bad, ugly}; {1,2,3,4,5}), our approach produces continuous quality values that allow more fine-grained and flexible enrolment and matching processes.

- **Includes quality assessment of single minutiae** - Unlike previous works, our solution assesses the quality of full fingerprints as well as the quality of single minutiae. This is specifically useful in forensic scenarios where forensic examiner aim to find reliable minutiae suitable for identification.

For more details, please take a look at the paper.

## Results

Face image quality assessment results are shown below on LFW (left) and Adience (right). SER-FIQ (same model) is based on ArcFace and shown in red. The plots show the FNMR at ![\Large 10^{-3}](https://latex.codecogs.com/gif.latex?\inline&space;10^{-3}) FMR as recommended by the [best practice guidelines](https://op.europa.eu/en/publication-detail/-/publication/e81d082d-20a8-11e6-86d0-01aa75ed71a1) of the European Border Guard Agency Frontex. For more details and results, please take a look at the paper.

<img src="FQA-Results/001FMR_lfw_arcface.png" width="430" >  <img src="FQA-Results/001FMR_adience_arcface.png" width="430" >

## Installation

We recommend Anaconda to install the required packages.
This can be done by creating an virtual environment via

```shell
conda env create -f environment.yml
```

or by manually installing the following packages.


```shell
conda create -n serfiq python=3.6.9
conda install cudatoolkit
conda install cudnn
conda install tensorflow=1.14.0
conda install mxnet
conda install mxnet-gpu
conda install tqdm
conda install -c conda-forge opencv
conda install -c anaconda scikit-learn
conda install -c conda-forge scikit-image
conda install keras=2.2.4
```

After the required packages have been installed, also download the [Insightface codebase at the needed git point in the repository history](https://github.com/deepinsight/insightface/tree/60bb5829b1d76bfcec7930ce61c41dde26413279) to a location of your choice and extract the archive if necessary.

We will refer to this location as _$Insightface_ in the following. 

The path to the Insightface repository must be passed to the [InsightFace class in face_image_quality.py](https://github.com/pterhoer/FaceImageQuality/blob/b59b2ec3c58429ee867dee25a4d8165b9c65d304/face_image_quality.py#L25). To avoid any problems, absolute paths can be used. Our InsightFace class automatically imports the required dependencies from the Insightface repository.
```
insightface = InsightFace(insightface_path = $Insightface) # Repository-path as parameter
```
[Please be aware to change the location in our example code according to your setup](https://github.com/pterhoer/FaceImageQuality/blob/b59b2ec3c58429ee867dee25a4d8165b9c65d304/serfiq_example.py#L9).

A pre-trained Arcface model is also required. We recommend using the "_LResNet100E-IR,ArcFace@ms1m-refine-v2_" model. [This can be downloaded from the Insightface Model-Zoo](https://github.com/deepinsight/insightface/wiki/Model-Zoo#31-lresnet100e-irarcfacems1m-refine-v2).

Extract the downloaded _model-0000.params_ and _model-symbol.json_ to the following location on your computer:
```
$Insightface/models/
```

After following these steps you can activate your environment (default: _conda activate serfiq_) and run the [example code](serfiq_example.py).

The implementation for SER-FIQ based on ArcFace can be found here: [Implementation](face_image_quality.py). <br/>
In the [Paper](https://arxiv.org/abs/2003.09373), this is refered to _SER-FIQ (same model) based on ArcFace_. <br/>






## Citing

If you use this code, please cite the following paper.


```
@inproceedings{DBLP:conf/icb/TerhorstKDKK20,
  author    = {Philipp Terh{\"{o}}rst and
               Jan Niklas Kolf and
               Naser Damer and
               Florian Kirchbuchner and
               Arjan Kuijper},
  title     = {Face Quality Estimation and Its Correlation to Demographic and Non-Demographic
               Bias in Face Recognition},
  booktitle = {2020 {IEEE} International Joint Conference on Biometrics, {IJCB} 2020,
               Houston, TX, USA, September 28 - October 1, 2020},
  pages     = {1--11},
  publisher = {{IEEE}},
  year      = {2020},
  url       = {https://doi.org/10.1109/IJCB48548.2020.9304865},
  doi       = {10.1109/IJCB48548.2020.9304865},
  timestamp = {Thu, 14 Jan 2021 15:14:18 +0100},
  biburl    = {https://dblp.org/rec/conf/icb/TerhorstKDKK20.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}


```

If you make use of our SER-FIQ implementation based on ArcFace, please additionally cite the original ![ArcFace module](https://github.com/deepinsight/insightface).

## Acknowledgement

This research work has been funded by the German Federal Ministry of Education and Research and the Hessen State Ministry for Higher Education, Research and the Arts within their joint support of the National Research Center for Applied Cybersecurity ATHENE. 

## License 

This project is licensed under the terms of the Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license.
Copyright (c) 2021 Fraunhofer Institute for Computer Graphics Research IGD Darmstadt
