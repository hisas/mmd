# MMD (Multi-Modal Dialogue) 

## Preprocess

1. `$ mkdir data`
2. `$ wget -O data/data.zip http://gazefollow.csail.mit.edu/downloads/data.zip`
3. `$ unzip data/data.zip -d data`
4. `$ rm data/data.zip`
5. Download [`vector.zip`](https://drive.google.com/open?id=0ByFQ96A4DgSPNFdleG1GaHcxQzA), unzip `vector.zip` and put `model.vec` under `data` folder
6. Put `jparvsu-response.tsv` under `data` folder
7. `$ python preprocess.py`

## Usage

### Train options
```
Options:

    --task                response or action
    --text_model          lstm or transformer
    --image_model         vgg
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

