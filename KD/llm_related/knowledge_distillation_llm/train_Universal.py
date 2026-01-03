from transformers import AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator
from peft import LoraConfig, get_peft_model, TaskType
from peft import PeftModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from dataset import SFTDataset
from utils import compute_fkl, compute_rkl, compute_skewed_fkl, compute_skewed_rkl
from datasets import load_dataset
from transformers import DefaultDataCollator
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"## 设置使用哪块GPU
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 新增，关闭并行警告
from sklearn.metrics import f1_score
import numpy as np

def preprocess_logits_for_metrics(logits, labels):
    # logits: (B, T, V)
    preds = torch.argmax(logits, dim=-1)
    return preds

def compute_f1(eval_preds):
    preds, labels = eval_preds

    preds  = np.asarray(preds)
    labels = np.asarray(labels)

    mask = labels != -100
    if mask.sum() == 0:
        return {"f1": 0.0}

    tp = ((preds == labels) & mask).sum()
    total = mask.sum()

    f1_micro = 2 * tp / (tp + total)

    return {"f1": float(f1_micro)}

class KGTrainer(Trainer):
    def __init__(
        self,
        model = None,
        teacher_model = None,
        if_use_entropy = False,
        args = None,
        data_collator = None, 
        train_dataset = None,
        eval_dataset = None,
        tokenizer = None,
        model_init = None, 
        compute_metrics = None, 
        callbacks = None,
        preprocess_logits_for_metrics = None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.teacher_model = teacher_model
        self.if_use_entropy = if_use_entropy
        
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        
        # student forward
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"]
        )
        # teacher forward
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=inputs["teacher_input_ids"],
                attention_mask=inputs["teacher_attention_mask"]
            )
        
        loss = outputs.loss
        logits = outputs.logits
        teacher_logits = teacher_outputs.logits
        
        # 如果教师模型和学生模型输出形状不匹配，对学生模型进行padding或对教师模型进行截断
        if logits.shape[-1] != teacher_logits.shape[-1]:
            # gap = teacher_logits.shape[-1] - logits.shape[-1]
            # if gap > 0:
            #     pad_logits = torch.zeros((logits.shape[0], logits.shape[1], gap)).to(logits.device)
            #     logits = torch.cat([logits, pad_logits], dim=-1)
            
            teacher_logits = teacher_logits[:, :, :logits.shape[-1]]
        
        labels = inputs['labels']
        # kl = compute_fkl(logits, teacher_logits, labels, padding_id=-100, temp=2.0)
        kl = loss
        if self.if_use_entropy:
            loss_total = 0.5 * kl + 0.5 * loss
        else:
            loss_total = kl
        
        return (loss_total, outputs) if return_outputs else loss_total
        
def preprocess(example):
    input_text = f"{example.get('context', '')}\n问题：{example.get('question', '')}"
    answer = example.get('answers', [""])
    if isinstance(answer, list) and len(answer) > 0:
        target_text = str(answer[0]) if answer[0] is not None else ""
    else:
        target_text = ""
    max_length = 128

    student_inputs = student_tokenizer(
        input_text,
        max_length=max_length, 
        truncation=True,
        padding='max_length'
    )
    student_labels = student_tokenizer(
        target_text,
        max_length=max_length, 
        truncation=True,
        padding='max_length'
    )
    student_inputs["labels"] = [
        (l if l != student_tokenizer.pad_token_id else -100) for l in student_labels["input_ids"]
    ]

    teacher_inputs = teacher_tokenizer(
        input_text,
        max_length=max_length, 
        truncation=True,
        padding='max_length'
    )
    # 保证字段一定存在
    student_inputs["teacher_input_ids"] = teacher_inputs.get("input_ids", [0]*max_length)
    student_inputs["teacher_attention_mask"] = teacher_inputs.get("attention_mask", [0]*max_length)

     # 检查
    if "teacher_input_ids" not in student_inputs or "teacher_attention_mask" not in student_inputs:
        print("preprocess error:", example)
    return student_inputs

def custom_data_collator(features):
    if len(features) == 0:
        return {}

    batch = DefaultDataCollator()(features)

    # teacher 字段一定存在（因为 remove_unused_columns=False）
    batch["teacher_input_ids"] = torch.tensor(
        [f["teacher_input_ids"] for f in features],
        dtype=torch.long
    )
    batch["teacher_attention_mask"] = torch.tensor(
        [f["teacher_attention_mask"] for f in features],
        dtype=torch.long
    )
    return batch

if __name__ == '__main__': 
    # 学生模型
    # model = AutoModelForCausalLM.from_pretrained("mistralai/Ministral-3-3B-Instruct-2512")
    # Load model directly
    from transformers import AutoTokenizer, AutoModelForCausalLM
    student_model_dir = "/home/fhj/hf_cache/models--bigscience--bloomz-560m/snapshots/a2845d7e13dd12efae154a9f1c63fcc2e0cc4b05"
    model = AutoModelForCausalLM.from_pretrained(student_model_dir)
    #model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m", cache_dir="/home/fhj/hf_cache")
    for name, module in model.named_modules():
        print(name)
    lora_config = LoraConfig(
    r=8,  
    lora_alpha=256,  
    target_modules=[ "query_key_value", "self_attention.dense","mlp.dense_h_to_4h","mlp.dense_4h_to_h"],
    # target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # 目标层
    lora_dropout=0.1, 
    task_type=TaskType.CAUSAL_LM)
    # 使用lora方法训练
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads() 
    model.cuda()
    print(model.print_trainable_parameters())
    
    # 教师模型，在给定数据上通过lora微调
    teacher_dir = "/home/fhj/hf_cache/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/63a8b081895390a26e140280378bc85ec8bce07a"
    teacher_model = AutoModelForCausalLM.from_pretrained("/home/fhj/hf_cache/Mistral-7B-Instruct-v0.2/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/63a8b081895390a26e140280378bc85ec8bce07a")
    #teacher_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2",
    #            cache_dir="/home/fhj/hf_cache/Mistral-7B-Instruct-v0.2")
    # 是否加载lora模型
    # lora_path = 'qwen2.5_7b/lora/sft'
    # teacher_model = PeftModel.from_pretrained(teacher_model, lora_path)
    teacher_model.cuda()
    teacher_model.eval()
    # 初始化两个tokenizer
    student_tokenizer = AutoTokenizer.from_pretrained(student_model_dir)
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_dir)
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    args = TrainingArguments(output_dir='/home/fhj/newCode/KD/llm_related/knowledge_distillation_llm/results',
                            num_train_epochs=4, 
                            do_train=True, 
                            per_device_train_batch_size=16,
                            gradient_accumulation_steps=2,
                            gradient_checkpointing=True,
                            optim='adafactor',
                            logging_steps=10,
                            report_to='tensorboard',
                            save_strategy='epoch',
                            save_total_limit=10,
                            bf16=True,
                            learning_rate=0.0005,
                            lr_scheduler_type='cosine',
                            dataloader_num_workers=8,
                            remove_unused_columns=False,
                            eval_strategy="steps",      # 新增：每隔多少步评估一次
                            eval_steps=100,                   # 新增：每10步评估一次
                            dataloader_pin_memory=True)


    # 替换原有 data_collator
    data_collator = custom_data_collator
    # data_collator = DefaultDataCollator()
    # dataset = SFTDataset('/home/fhj/code/KD/llm_related/knowledge_distillation_llm/alpaca_data.json', tokenizer=tokenizer, max_seq_len=512)


    train = load_dataset("parquet",
            data_files="/home/fhj/newCode/data/uld_loss_Mistral-7B-Instruct-v0.2-qed/train-00000-of-00001-00ee1a1595974b81.parquet" )
    validation = load_dataset("parquet",
            data_files="/home/fhj/newCode/data/uld_loss_Mistral-7B-Instruct-v0.2-qed/validation-00000-of-00001-d14b657647a15533.parquet" )
    
    # 处理数据
    processed_train = train["train"].map(
        preprocess,
        remove_columns=train["train"].column_names,
        load_from_cache_file=False
    )
    processed_validation = validation["train"].map(
        preprocess,
        remove_columns=validation["train"].column_names,
        load_from_cache_file=False
    )
    print(processed_train[0])
    print(processed_train.features)
    # 构造 DatasetDict
    processed_dataset = {"train": processed_train, "validation": processed_validation}

    
    trainer = KGTrainer(model=model,
                        teacher_model=teacher_model, 
                        if_use_entropy = True,
                        args=args, 
                        train_dataset=processed_dataset['train'], 
                        eval_dataset=processed_dataset['validation'],
                        tokenizer=student_tokenizer, 
                        # compute_metrics=compute_f1,
                        # data_collator=custom_data_collator,
                        compute_metrics=compute_f1,
                        callbacks=None ,
                        preprocess_logits_for_metrics=preprocess_logits_for_metrics)
    # 如果是初次训练resume_from_checkpoint为false，接着checkpoint继续训练，为True
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('./saves')
    trainer.save_state()
    
      
    