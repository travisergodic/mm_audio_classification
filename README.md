## 安裝
```
pip install -r requirements.txt
```

## 訓練
```
python tools/train.py --config_file <config_file>
```

## 評估
```
python tools/test.py --task "evaluate" --config_file <config_file> --data_dir <source_data_dir>
```

## 預測
```
python tools/test.py --task "inference" --config_file <config_file> --data_dir <source_data_dir> --save_path <save_path>
```

## colab 實作
https://colab.research.google.com/drive/15pxVJa4t1m0c4bY31pFxmOZjEPf4u5Rp?usp=sharing


## 待辦事項
1. unify datasets