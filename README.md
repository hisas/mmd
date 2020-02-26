# MMD (Multi-Modal Dialogue) 

## Preprocess

1. `$ pip install -r requirements.txt`
2. [Install Git LFS](https://github.com/git-lfs/git-lfs/wiki/Installation) and run `$ git lfs pull` to download large files 
3. `$ python preprocess.py`

## Usage

### Train options
```
Options:

    --task                response or action
    --text_model          lstm or transformer or bert
    --image_model         vgg or resnet or efficientnet
    --synthesis_method    matmul or concat
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
2. `$ python train.py --task action --text_model transformer --image_model vgg --synthesis_method matmul`
3. `$ python eval.py --path <path> --log`

### Text, Image and Gaze Modality

1. `$ cd modality/text_image_gaze`
2. `$ python train.py --task action --text_model transformer --image_model vgg --synthesis_method matmul`
3. `$ python eval.py --path <path> --log`
