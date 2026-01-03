from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch
import numpy as np
from tqdm import tqdm
from peft import PeftModel
os.environ["CUDA_VISIBLE_DEVICES"] = "1"## 设置使用哪块GPU
#加载预训练模型
teacher_dir = "/home/fhj/hf_cache/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/63a8b081895390a26e140280378bc85ec8bce07a"
model = AutoModelForCausalLM.from_pretrained("/home/fhj/hf_cache/Mistral-7B-Instruct-v0.2/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/63a8b081895390a26e140280378bc85ec8bce07a")
tokenizer = AutoTokenizer.from_pretrained(teacher_dir)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

#配置并插入LoRA模块
from peft import LoraConfig, get_peft_model
lora_config = LoraConfig(
   r=8, # 秩
   lora_alpha=32, # 缩放系数
   target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # 目标层
   lora_dropout=0.05,
   bias="none",
   task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.cuda()

## 数据处理 与加载
def preprocess(example):
    input_text = f"{example.get('context', '')}\nQuestion: {example.get('question', '')}"
    answer = example.get('answers', [""])
    if isinstance(answer, list) and len(answer) > 0:
        target_text = str(answer[0]) if answer[0] is not None else ""
    else:
        target_text = ""

    max_length = 128
    model_inputs = tokenizer(
        input_text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    labels = tokenizer(
        target_text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )["input_ids"]
    # pad 位置设为 -100 以忽略 loss
    model_inputs["labels"] = [
        (l if l != tokenizer.pad_token_id else -100) for l in labels
    ]
    return model_inputs

from datasets import load_dataset
train = load_dataset("parquet",
            data_files="/home/fhj/newCode/data/uld_loss_Mistral-7B-Instruct-v0.2-qed/train-00000-of-00001-00ee1a1595974b81.parquet" )
validation = load_dataset("parquet",
            data_files="/home/fhj/newCode/data/uld_loss_Mistral-7B-Instruct-v0.2-qed/validation-00000-of-00001-d14b657647a15533.parquet" )

# processed_train = train["train"].map(
#         preprocess,
#         remove_columns=train["train"].column_names,
#         load_from_cache_file=False
#     )
# processed_validation = validation["train"].map(
#         preprocess,
#         remove_columns=validation["train"].column_names,
#         load_from_cache_file=False
#     )

#评估函数
import re
import string
from collections import Counter

def normalize_answer(s):
    s = s.lower()
    s = re.sub(f"[{string.punctuation}]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def squad_f1(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return float(pred_tokens == gt_tokens)

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)

def extract_answer_text(ans):
    """从 answers 字段里抽出可比对的文本."""
    # 列表：取第一个
    if isinstance(ans, list):
        if not ans:
            return ""
        ans = ans[0]

    # 字典：尝试常见 key
    if isinstance(ans, dict):
        for k in ["text", "answer", "value"]:
            if k in ans:
                return "" if ans[k] is None else str(ans[k])
        # 都没有就直接转字符串兜底
        return str(ans)

    # 其它：直接转字符串
    if ans is None:
        return ""
    return str(ans)

@torch.no_grad()
def evaluate_qed_f1(
    model,
    dataset,
    tokenizer,
    device="cuda",
    max_new_tokens=64
):
    model.eval()
    scores = []

    for ex in tqdm(dataset):
        # 构造输入（与你 preprocess 保持一致）
        input_text = f"{ex.get('context', '')}\nQuestion: {ex.get('question', '')}"

        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=128
        ).to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

        pred_text = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        # 参考答案
        answers = ex.get("answers", [""])
        # gt = answers[0] if isinstance(answers, list) else answers
        gt = extract_answer_text(answers)
        print(pred_text,"\ngt:", gt)
        scores.append(squad_f1(pred_text, gt))

    return sum(scores) / len(scores) if scores else 0.0

#训练与保存    
from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(
   output_dir="./lora_output",
   per_device_train_batch_size=2,
   gradient_accumulation_steps=16,
   num_train_epochs=1,
    logging_steps=10,
    eval_strategy="steps",      # 新增：每隔多少步评估一次
    eval_steps=50,                   # 新增：每10步评估一次
    bf16=True
)
model = PeftModel.from_pretrained(model, "./lora_output")
val_subset = validation["train"].select(range(10))
qed_f1 = evaluate_qed_f1(
    model=model,
    dataset=val_subset,   # 原始 parquet
    tokenizer=tokenizer,
    device="cuda"
)

print("QED F1:", qed_f1)

trainer = Trainer(
   model=model,
   args=training_args,
   train_dataset=processed_train,
   eval_dataset=processed_validation,
)
trainer.train()
qed_f1 = evaluate_qed_f1(
    model=model,
    dataset=validation["train"],   # 原始 parquet
    tokenizer=tokenizer,
    device="cuda"
)

print("QED F1:", qed_f1)

model.save_pretrained("./lora_output")
tokenizer.save_pretrained("./lora_output")

#合并LoRA权重（推理加速）
# from peft import PeftModel
# base_model = AutoModelForCausalLM.from_pretrained(model_name)
# lora_model = PeftModel.from_pretrained(base_model, "./lora_output")
# merged_model = lora_model.merge_and_unload()
# merged_model.save_pretrained("./merged_model")
# tokenizer.save_pretrained("./merged_model")