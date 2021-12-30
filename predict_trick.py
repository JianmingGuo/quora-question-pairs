import argparse
from datasets import load_dataset
from transformers import RobertaTokenizer,DataCollatorWithPadding,RobertaConfig,RobertaForSequenceClassification\

from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import torch
import torch.nn.functional as F
from custom_model import ModelParamConfig, CustomModel
import os
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--max_len', type=int, help='pair文本最大长度', default=32)
parser.add_argument('--batch_size', type=int, help='batch_size', default=64)
parser.add_argument('--model_dir', type=str, help='加载模型的地址', default='./finetune_trick')
parser.add_argument('--output_dir', type=str, help='测试结果写入文件夹', default='test_results')
parser.add_argument('--run_mode', type=str, help='运行模式normal or test', default='normal')
args = parser.parse_args()

data_files ={"test":"./data/test.csv"}
dataset = load_dataset('csv',data_files=data_files,delimiter='\t',quoting=3)
tokenizer = RobertaTokenizer.from_pretrained('./roberta-base')
data_collator = DataCollatorWithPadding(tokenizer=tokenizer,padding='max_length',max_length=args.max_len)


def tokenize_function(example):
    return tokenizer(example["question1"],example["question2"],truncation=True,max_length=32)





tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["question1", "question2"])
tokenized_datasets.rename_column("label", "labels")
test_dataloader = DataLoader(
    tokenized_datasets["test"], shuffle=False, batch_size=args.batch_size, collate_fn=data_collator
)

# config_origin = RobertaConfig.from_pretrained('./roberta-base')
model_param_config = ModelParamConfig()
model = CustomModel(args.model_dir, model_param_config)
model.load_state_dict(torch.load(os.path.join(args.model_dir, 'pytorch_model.bin')))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

test_y_true = []
test_predictions = []
test_label_ids = []
test_metrics = 0.0
# logger.info('\n------------start to predict------------')
with torch.no_grad():
    for step, batch_data in enumerate(test_dataloader):
        for key in batch_data.keys():
            batch_data[key] = batch_data[key].to(device)
        labels = batch_data['labels']
        test_y_true.extend(labels.cpu().numpy())
        logits = model(**batch_data)
        predict_scores = logits.softmax(-1)
        test_label_ids.extend(predict_scores.argmax(dim=1).detach().to("cpu").numpy())
        # 预测为1的概率值
        predict_scores = predict_scores[:, 1]
        test_predictions.extend(predict_scores.detach().cpu().numpy())


# 计算metric
test_f1 = f1_score(test_y_true, test_label_ids, average='macro')

print(test_f1)

