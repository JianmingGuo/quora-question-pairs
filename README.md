

# NLP-3

## 文件说明

```shell
.
├── custom_model.py //在RoBERTa等预训练模型的输出层后加上Dropout层和线性层完成分类
├── finetune.py //不使用trick进行finetune
├── finetune_trick.py //使用trick进行finetune
├── predict.py //调用无trick训练的模型进行预测
├── predict_trick.py //调用trick的模型进行预测
├── pretrain.py //根据无标签数据集进行预训练
├── readme.md
└── utils //训练所用trick的函数
    ├── attack.py  //包括fgm,pdg
    ├── ema.py   //
    └── focal_loss.py  //利用focal loss训练
```

本组最好结果的模型及相关参数存放于云盘，该模型在test集的测试结果F1 score能达到**0.902280396** 

分享内容: [pretrain_20_fgm](https://jbox.sjtu.edu.cn/l/h1y47k)         

## 预测

> python predict_trick.py --model-dir ./pretrain_20_fgm

## 结果

本小组实验结果如下：

| **Model** | **Pretrain** | **Trick**  |    **F1 Score**    |      **Accuracy**      |
| :-------: | :----------: | :--------: | :----------------: | :--------------------: |
|  Roberta  |     None     |     -      | 0.897605291444648  |   0.904202428949516    |
|  Roberta  |  100 epochs  |     -      | 0.893969645685547  |       0.90101165       |
|  Roberta  |  20 epochs   |     -      |    0.898743361     |      0.904919736       |
|  Roberta  |     None     |    fgm     |    0.901721227     |         **-**          |
|  Roberta  |  20 epochs   |    fgm     |  **0.902280396**   |         **-**          |
|  Roberta  |  20 epochs   |    pdg     |    0.899325729     |         **-**          |
|  Roberta  |  20 epochs   | focal loss |     0.89777293     |         **-**          |
|  Albert   |     None     |    None    | 0.8800502735119846 |   0.8873580845432734   |
|  Albert   |  10 epochs   |    None    | 0.8812097440231119 | 0**.**8892131885527715 |

