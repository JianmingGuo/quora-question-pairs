from transformers import AutoTokenizer,AutoModelForMaskedLM,RobertaTokenizer,AlbertTokenizer,AlbertForMaskedLM, RobertaModel,DataCollatorForLanguageModeling,\
RobertaForMaskedLM,Trainer,TrainingArguments
from datasets import load_dataset
import argparse
from transformers.trainer_utils import SchedulerType
import os

# tokenizer = AutoTokenizer.from_pretrained("./bert-base-uncased")
# model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

parser = argparse.ArgumentParser()
parser.add_argument('--max_len', type=int, help='pair文本最大长度', default=32)
parser.add_argument('--batch_size', type=int, help='batch_size', default=64)
parser.add_argument('--pretrained_model_dir', type=str, help='预训练模型地址', default='./pretrained_model/bert')
parser.add_argument('--mlm_probability', type=float, help='mask概率', default=0.15)
parser.add_argument('--num_train_epochs', type=int, help='训练epoch数', default=10)
parser.add_argument('--max_steps', type=int, help='最大训练步数，如果设置了，则覆盖num_train_epochs', default=-1)
parser.add_argument('--warmup_ratio', type=float, help='warmup比例', default=0.05)
parser.add_argument('--learning_rate', type=float, help='学习率', default=5e-5)
parser.add_argument('--save_strategy', type=str, help='保存模型策略', default='epoch')
parser.add_argument('--save_steps', type=int, help='多少步保存模型，如果save_strategy=epoch，则失效', default=100)
parser.add_argument('--save_total_limit', type=int, help='checkpoint数量', default=1)
parser.add_argument('--logging_steps', type=int, help='多少步日志打印', default=100)
parser.add_argument('--output_model_dir', type=str, help='模型保存地址', default='./albert_pretrain_10')
parser.add_argument('--tar_name_prefix', type=str, help='最终模型打包名称', default='oppo_pretrained_bert')
parser.add_argument('--run_mode', type=str, help='运行模式normal or test', default='normal')
parser.add_argument('--seed', type=int, help='随机种子', default=2021)
args = parser.parse_args()


data_files = {"train":"./data/train.csv" ,"validation":"./data/valid.csv" , "test":"./data/test.csv" }
dataset = load_dataset('csv',data_files=data_files,delimiter='\t', quoting=3)

vocab_file_dir = './roberta-base/vocab.json'
tokenizer = AlbertTokenizer.from_pretrained('./albert-base-v2')

# print(dataset['train'][2]['question1'])

def tokenize_function(example):
    return tokenizer(example["question1"],example["question2"],truncation=True,padding="max_length",max_length=32)

# for i in range(282994):
#     print(i)
#     a = tokenize_function(dataset['train'][i+1])

# print(dataset['train'][150490])
# a = tokenize_function(dataset['train'][150490])
# print(a)
tokenized_datasets = dataset['train'].map(tokenize_function,batched=True,remove_columns=['label'])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=True,mlm_probability=0.15)

model = AlbertForMaskedLM.from_pretrained('./albert-base-v2')

model.resize_token_embeddings(len(tokenizer))

training_args = TrainingArguments(
    num_train_epochs=args.num_train_epochs,
    max_steps=args.max_steps,
    per_device_train_batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    lr_scheduler_type=SchedulerType.LINEAR,
    warmup_ratio=args.warmup_ratio,
    output_dir=args.output_model_dir,
    overwrite_output_dir=True,
    save_strategy=args.save_strategy,
    save_total_limit=args.save_total_limit,
    logging_steps=args.logging_steps,
    logging_first_step=True,
    seed=args.seed,
)

trainer = Trainer(
    model = model,
    args = training_args,
    data_collator = data_collator,
    train_dataset = tokenized_datasets,

)

trainer.train()
trainer.save_model(args.output_model_dir)
trainer.state.save_to_json(os.path.join(training_args.output_dir,"trainer_state.json"))
