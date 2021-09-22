Reimplementation of Tao Lei's Rationalizing Neural Prediction [paper](https://arxiv.org/abs/1606.04155), adapted from Adam's [code](https://github.com/yala/text_nn). This reimplementation departs from the original paper in the following ways:
1. Uses bi-lstm as our choice of recurrent unit instead of RCNN. 
2. Uses gumble softmax instead of REINFORCE to approximate the binary mask selection. 

This reimplementation is still WIP as I'm ironing out some kinks.

## Requirements
1. Download the (modified) ERASER's e-snli dataset [here](https://drive.google.com/file/d/1G70bdWTaGz1gpBuajVWS5tM8TPGPyX13/view?usp=sharing).
> We labelled each evidence on a character-level instead of token-level in order to accommodate different tokenizers.

2. Install dependencies in `requirements.txt`

## Usage
To train the vanilla model, run

```
cd pipeline
bash main.sh
```

Gradient flows are saved in the specified `out_dir`. View training metrics recorded via tensorboard by running
```
tensorboard --logdir ../out
```