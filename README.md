# MMD (Multi-Modal Dialogue) 

## Preprocess

1. `$ pip install -r requirements.txt`
2. [Install Git LFS](https://github.com/git-lfs/git-lfs/wiki/Installation) and run `$ git lfs pull` to download large files 
3. `$ python preprocess.py`

## Usage

### Train options
```
Options:

    --task                response / action
    --text_model          lstm / transformer / bert
    --image_model         vgg / resnet / efficientnet
    --joint_method        concat / sum / product / late
```

### Eval options
```
Options:

    --path    saved model path
    --log     output log
```

### Text modality

1. `$ cd modality/text`
2. `$ python train.py --task response --text_model lstm`
3. `$ python eval.py --path <path> --log`

### Text and Image Modality

1. `$ cd modality/text_image`
2. `$ python train.py --task action --text_model transformer --image_model vgg --joint_method late`
3. `$ python eval.py --path <path> --log`

### Text, Image and Gaze Modality

1. `$ cd modality/text_image_gaze`
2. `$ python train.py --task action --text_model transformer --image_model vgg --joint_method late`
3. `$ python eval.py --path <path> --log`
