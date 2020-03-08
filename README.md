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
    --image_model         vgg / resnet
    --fusion_method        concat / sum / product / mcb / mlb / mutan / block
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

### Image modality

1. `$ cd modality/image`
2. `$ python train.py --task action --text_model lstm --image_model vgg`
3. `$ python eval.py --path <path> --log`

### Text and Image modality

1. `$ cd modality/text_image`
2. `$ python train.py --task response --text_model bert --image_model resnet --fusion_method block`
3. `$ python eval.py --path <path> --log`

### Text, Image and Gaze modality

1. `$ cd modality/text_image_gaze`
2. `$ python train.py --task action --text_model bert --image_model resnet --fusion_method block`
3. `$ python eval.py --path <path> --log`

## Result

### Response task

#### Text modality
| Model / Metrics | 1 in 10 R@1 | 1 in 10 R@2 | 1 in 10 R@5 | 1 in 2 R@1 |
| --------------- |:------------:|:----------:|:-----------:|:----------:|
| LSTM            | 0.502        | 0.694      | 0.915       | 0.772      |
| Transformer     | 0.530        | 0.707      | 0.916       | 0.776      |
| Bert            | 0.502        | 0.678      | 0.896       | 0.760      |

#### Image modality
| Model / Metrics | 1 in 10 R@1 | 1 in 10 R@2 | 1 in 10 R@5 | 1 in 2 R@1 |
| --------------- |:-----------:|:-----------:|:-----------:|:----------:|
| VGG             | 0.499       | 0.689       | 0.913       | 0.704      |
| ResNet          | 0.376       | 0.560       | 0.842       | 0.703      |

#### Text and Image modality
| Model / Metrics         | 1 in 10 R@1 | 1 in 10 R@2 | 1 in 10 R@5 | 1 in 2 R@1 |
| ----------------------- |:-----------:|:-----------:|:-----------:|:----------:|
| LSTM w VGG, late        | 0.499       | 0.700       | 0.913       | 0.774      |
| Transformer w VGG, late | 0.504       | 0.704       | 0.919       | 0.777      | 

#### Text, Image and Gaze modality
| Model / Metrics         | 1 in 10 R@1 | 1 in 10 R@2 | 1 in 10 R@5 | 1 in 2 R@1 |
| ----------------------- |:-----------:|:-----------:|:-----------:|:----------:|
| LSTM w VGG, concat      | 0.529       | 0.721       | 0.928       | 0.790      |
| LSTM w VGG, sum         | 0.534       | 0.722       | 0.932       | 0.791      |
| LSTM w VGG, late        | 0.514       | 0.709       | 0.927       | 0.786      |
| Transformer w VGG, late | 0.406       | 0.584       | 0.855       | 0.713      |


### Action Task

#### Text modality
| Model / Metrics | 1 in 10 R@1 | 1 in 10 R@2 | 1 in 10 R@5 | 1 in 2 R@1 |
| --------------- |:-----------:|:-----------:|:-----------:|:----------:|
| LSTM            | 0.357       | 0.579       | 0.870       | 0.728      |
| Transformer     | 0.384       | 0.577       | 0.856       | 0.722      |
| Bert            | 0.405       | 0.595       | 0.864       | 0.726      |

#### Image modality
| Model / Metrics | 1 in 10 R@1 | 1 in 10 R@2 | 1 in 10 R@5 | 1 in 2 R@1 |
| --------------- |:-----------:|:-----------:|:------------:|:---------:|
| VGG             | 0.353       | 0.514       | 0.820        | 0.687     |
| ResNet          | 0.326       | 0.499       | 0.808        | 0.686     |

#### Text and Image modality
| Model / Metrics         | 1 in 10 R@1 | 1 in 10 R@2 | 1 in 10 R@5 | 1 in 2 R@1 |
| ----------------------- |:-----------:|:-----------:|:-----------:|:----------:|
| LSTM w VGG, late        | 0.360       | 0.558       | 0.861       | 0.729      |
| Transformer w VGG, late | 0.348       | 0.517       | 0.826       | 0.686      |
