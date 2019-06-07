NOISE SURPRESSION
=====
v1.0.0

Python package for noise supression in audio based on DNN

## Acknowledgements
Based on the Yong Xu and Qiuqiang Kong SEDDN package https://github.com/yongxuUSTC/sednn

## Model
A trained model (https://github.com/nsu-ai-team/noise_supression/blob/master/model/md_10000iters.h5) is distributed with the python package.
#### Model information:
1) Trained on a subset of the LibriSpeech audio corpus (14.6 hours)
2) PESQ=1.82 +- 0.24
3) Trained using the _sednn/mixture2clean_dnn_ subrepo
4) Audio with english speech are used, but can show relatively good quality on other languages
5) Will show worse enhancement results on data from other corpuses (or real-life data), keep that in mind

## Installation
```
pip install git+https://git@github.com/nsu-ai-team/noise_supression.git
```

## Python usage
Loading the module
```
>>> from noise_supression.enhance_audio import AudioEnhancer
>>> ae = AudioEnhancer()
```
Enhancing an audio file
```
>>> ae.enhance_audio('example/61-70968-0000.n.wav','example/61-70968-0000.enh.wav')
```

## Command line usage (if cloned)
Loading the module
```
$ python enhance_audio.py example/61-70968-0000.n.wav example/61-70968-0000.enh.wav
```
