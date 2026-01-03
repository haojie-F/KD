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

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"## 设置使用哪块GPU

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
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            preprocess_logits_for_metrics,
        )
        self.teacher_model = teacher_model
        self.if_use_entropy = if_use_entropy
        
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        
        outputs = model(**inputs)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
        
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
        kl = compute_fkl(logits, teacher_logits, labels, padding_id=-100, temp=2.0)
        
        if self.if_use_entropy:
            loss_total = 0.5 * kl + 0.5 * loss
        else:
            loss_total = kl
        
        return (loss_total, outputs) if return_outputs else loss_total
        
def preprocess(example):
    input_text = f"{example['context']}\n问题：{example['question']}"
    answer = example.get('answers', [""])
    if isinstance(answer, list) and len(answer) > 0:
        target_text = str(answer[0]) if answer[0] is not None else ""
    else:
        target_text = ""
    max_length = 64##节省内存，后面可以改大一些
    model_inputs = tokenizer(
        input_text,
        max_length=max_length, 
        truncation=True,
        padding='max_length'  # 强制补齐
    )
    labels = tokenizer(
        target_text,
        max_length=max_length, 
        truncation=True,
        padding='max_length'  # 强制补齐
    )
    model_inputs["labels"] = labels["input_ids"]
    # 将 padding id 替换为 -100
    model_inputs["labels"] = [
        (l if l != tokenizer.pad_token_id else -100) for l in labels["input_ids"]
    ]
    # 检查labels
    for l in model_inputs["labels"]:
        if l != -100 and (l < 0 or l >= tokenizer.vocab_size):
            print("Label 越界:", l)
    return model_inputs

if __name__ == '__main__': 
    # 学生模型
    # model = AutoModelForCausalLM.from_pretrained("mistralai/Ministral-3-3B-Instruct-2512")
    # Load model directly
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")
    model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # 如果没有pad_token，设置为eos_token
    # 显式打印和确认padding id
    print("pad_token_id:", tokenizer.pad_token_id)
    # for name, module in model.named_modules():
    #     print(name)
    lora_config = LoraConfig(
    r=8,  
    lora_alpha=256,  
    target_modules=[ "self_attention.query_key_value", "self_attention.dense","mlp.dense_h_to_4h","mlp.dense_4h_to_h"],
    lora_dropout=0.1, 
    task_type=TaskType.CAUSAL_LM)
    # 使用lora方法训练
    model = get_peft_model(model, lora_config)
    model.cuda()
    print(model.print_trainable_parameters())
    
    # 教师模型，在给定数据上通过lora微调
    teacher_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    # 是否加载lora模型
    # lora_path = 'qwen2.5_7b/lora/sft'
    # teacher_model = PeftModel.from_pretrained(teacher_model, lora_path)
    teacher_model.cuda()
    teacher_model.eval()
    
    args = TrainingArguments(output_dir='/home/fhj/newCode/KD/llm_related/knowledge_distillation_llm/results',
                            num_train_epochs=4, 
                            do_train=True, 
                            per_device_train_batch_size=1,
                            gradient_accumulation_steps=32,
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
                            dataloader_pin_memory=True)
    data_collator = DefaultDataCollator()
    # dataset = SFTDataset('/home/fhj/code/KD/llm_related/knowledge_distillation_llm/alpaca_data.json', tokenizer=tokenizer, max_seq_len=512)


    train = load_dataset("parquet",
            data_files="/home/fhj/newCode/data/uld_loss_Mistral-7B-Instruct-v0.2-qed/train-00000-of-00001-00ee1a1595974b81.parquet" )
    validation = load_dataset("parquet",
            data_files="/home/fhj/newCode/data/uld_loss_Mistral-7B-Instruct-v0.2-qed/validation-00000-of-00001-d14b657647a15533.parquet" )
    
    # 处理数据
    processed_train = train["train"].map(preprocess)
    processed_validation = validation["train"].map(preprocess)
    # 构造 DatasetDict
    processed_dataset = {"train": processed_train, "validation": processed_validation}

    
    trainer = KGTrainer(model=model,
                        teacher_model=teacher_model, 
                        if_use_entropy = True,
                        args=args, 
                        train_dataset=processed_dataset['train'], 
                        eval_dataset=processed_dataset['validation'],
                        tokenizer=tokenizer, 
                        data_collator=data_collator)
    # 如果是初次训练resume_from_checkpoint为false，接着checkpoint继续训练，为True
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('./saves')
    trainer.save_state()
    
      
    