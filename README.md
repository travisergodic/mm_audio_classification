# MM Audio Classification
An audio classification model library containing the following SOTA models: 
+ **Audio Spectrogram Transformer** (AST)
+ **Self-Supervised Audio Spectrogram Transformer** (SSAST)
+ **Hierarchical Token-Semantic Audio Transformer** (HTSAT)
+ **EfficientAT**
+ **Masked Modeling Duo** (M2D)
+ **CNN14**

## Install
```
python -m venv audio_cls_env
source audio_cls_env/bin/activate
cd audio_cls_env/bin/activate/mm_audio_classification
pip install -r requirements.txt
```

## Train
```
python tools/train.py --config_file <config_file>
```

## Evaluate
```
python tools/test.py --task "evaluate" --config_file <config_file> --data_dir <source_data_dir>
```

## Inference
```
python tools/test.py --task "inference" --config_file <config_file> --data_dir <source_data_dir> --save_path <save_path>
```

## [Colab Practice](https://colab.research.google.com/drive/1oJ_dvs09nrI1lQ6PHepbt6Rn5BOqhXCW#scrollTo=CoV_uZZN9l9C) 


## Reference
1. [AST](https://github.com/YuanGongND/ast)
2. [SSAST](https://github.com/YuanGongND/ssast)
3. [HTSAT](https://github.com/RetroCirce/HTS-Audio-Transformer)
4. [EfficientAT](https://github.com/fschmid56/EfficientAT)
5. [M2D](https://github.com/nttcslab/m2d)
6. [PANN](https://github.com/qiuqiangkong/audioset_tagging_cnn)