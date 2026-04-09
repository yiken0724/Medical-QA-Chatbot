import re
import pandas as pd
import torch
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, T5ForConditionalGeneration, AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer
from peft import LoraConfig, get_peft_model, TaskType

def main():
    torch.cuda.empty_cache()

    # PREPROCESSING
    medical_train = pd.read_csv("medical_qa_train.csv")
    medical_train['Formatted_Input'] = medical_train['Formatted_Input'].str.replace('\n', " ", regex=False)
    medical_train = medical_train.drop(columns='Question')

    medical_val = pd.read_csv("medical_qa_val.csv")
    medical_val['Formatted_Input'] = medical_val['Formatted_Input'].str.replace('\n', ' ', regex=False)
    medical_val = medical_val.drop(columns='Question')

    medical_train["Formatted_Input"] = medical_train["Formatted_Input"].astype(str).fillna("").apply(lambda x: x if isinstance(x, str) else " ".join(map(str, x)))
    medical_train["Answer"] = medical_train["Answer"].astype(str).fillna("").apply(lambda x: x if isinstance(x, str) else " ".join(map(str, x)))
    medical_val["Formatted_Input"] = medical_val["Formatted_Input"].astype(str).fillna("").apply(lambda x: x if isinstance(x, str) else " ".join(map(str, x)))
    medical_val["Answer"] = medical_val["Answer"].astype(str).fillna("").apply(lambda x: x if isinstance(x, str) else " ".join(map(str, x)))

    train_ds = Dataset.from_pandas(medical_train)
    val_ds = Dataset.from_pandas(medical_val)


    model_name = "google/flan-t5-base"  # or "t5-base"
    base = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # LoRA config: decoder-only, attention Q/V, seq2seq LM
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,   # <- important for seq2seq, not classification
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        # target_modules=r".*(encoder|decoder)\..*(SelfAttention|EncDecAttention)\.(q|k|v|o)$"
        target_modules=r"(encoder|decoder)\..*((SelfAttention|EncDecAttention)\.(q|k|v|o)$|(DenseReluDense\.(wi|wo)$))"
    )

    model = get_peft_model(base, lora_cfg)


    args = Seq2SeqTrainingArguments(
        output_dir="./flan_t5_med_lora_v9",
        # eval/save cadence
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,

        # optimization
        # learning_rate=1e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",

        # batches
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=8,

        # generation + logging
        predict_with_generate=True,
        logging_steps=50,
        generation_max_length=256,       # or whatever max total length you want
        generation_num_beams=4,

        # precision & stability
        fp16=False,
        label_smoothing_factor=0.1,

        # training loop
        num_train_epochs=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Adafactor + its own clipping (no separate max_grad_norm)
        optim="adafactor",
        optim_args="clip_threshold=1.0,relative_step=True,scale_parameter=True,warmup_init=True"
    )

    
    def preprocess(batch):
        # Encode the input and the answer
        model_inputs = tokenizer(
            batch["Formatted_Input"],
            max_length=256,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["Answer"],
            max_length=256,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_ds = train_ds.map(preprocess, batched=True, remove_columns=["Formatted_Input", "Answer"])
    val_ds = val_ds.map(preprocess, batched=True, remove_columns=["Formatted_Input", "Answer"])

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, pad_to_multiple_of=8)

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()

    model.save_pretrained("./flan_t5_med_lora_v9")
    tokenizer.save_pretrained("./flan_t5_med_lora_v9")


if __name__ == '__main__':
    main()