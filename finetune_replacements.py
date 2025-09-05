import os, json, logging, math, torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq, BitsAndBytesConfig, EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model


def is_distributed():
    return int(os.environ.get("WORLD_SIZE", 1)) > 1


MODELS = {
    "m2m100_418m": {"model_id": "../Data/merged/m2m100_418m", "language_map": {"en": "en", "fr": "fr"}},
    "mbart50_mmt_fr": {"model_id": "../Data/merged/mbart50_mmt_fr",
                       "language_map": {"en": "en_XX", "fr": "fr_XX"}, "restrict_source_language": "en"},
    "mbart50_mmt_en": {"model_id": "../Data/merged/mbart50_mmt_en",
                       "language_map": {"en": "en_XX", "fr": "fr_XX"}, "restrict_source_language": "fr"},
    "opus_mt_en_fr": {"model_id": "../Data/merged/opus_mt_en_fr", "language_map": {"en": "en", "fr": "fr"},
                      "restrict_source_language": "en"},
    "opus_mt_fr_en": {"model_id": "../Data/merged/opus_mt_fr_en", "language_map": {"en": "en", "fr": "fr"},
                      "restrict_source_language": "fr"},
}


def setup_logging(output_directory, to_file=True):
    os.makedirs(output_directory, exist_ok=True)
    handlers = [logging.StreamHandler()]
    if to_file:
        handlers.append(logging.FileHandler(os.path.join(output_directory, "console_output.txt"), encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", handlers=handlers)


def load_tokenizer_and_model(model_id, use_qlora, use_bfloat16, device_map):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {"torch_dtype": torch.bfloat16 if use_bfloat16 else torch.float16, "trust_remote_code": True}
    
    if use_qlora:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if use_bfloat16 else torch.float16
        )
    
    is_opus_model = "opus-mt" in model_id.lower() or "helsinki" in model_id.lower()
    if device_map is not None and not is_opus_model:
        model_kwargs["device_map"] = device_map
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **model_kwargs)
    if hasattr(model.config, "vocab_size") and len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model.config.use_cache = False
    if use_qlora and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return tokenizer, model


def attach_lora(model, r, alpha, dropout):
    names = ["q", "k", "v", "o", "q_proj", "k_proj", "v_proj", "o_proj", "in_proj_weight"]
    detected = [n for n in names if
                any(hasattr(m, n) or n in type(m).__name__.lower() for _, m in model.named_modules())]
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none", task_type="SEQ_2_SEQ_LM",
                     target_modules=list(set(detected)) or None)
    peft_model = get_peft_model(model, cfg)
    peft_model.train()
    peft_model.print_trainable_parameters()  # Log trainable parameters for debugging
    return peft_model


class Preprocessor:
    def __init__(self, model_name, tokenizer, language_map, max_source_length, max_target_length,
                 restrict_source_language=None):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.language_map = language_map
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.restrict_source_language = restrict_source_language
        
        if "en" not in language_map or "fr" not in language_map:
            raise ValueError(f"Language map for {model_name} must contain both 'en' and 'fr' keys")
    
    def _setup_tokenizer_languages(self, source_language, target_language):
        if not hasattr(self.tokenizer, 'src_lang'):
            return  # Tokenizer doesn't support language codes
        
        mapped_source = self.language_map.get(source_language, source_language)
        mapped_target = self.language_map.get(target_language, target_language)
        
        if self.model_name in ["m2m100_418m", "mbart50_mmt_fr", "mbart50_mmt_en"]:
            self.tokenizer.src_lang = mapped_source
            if hasattr(self.tokenizer, "tgt_lang"):
                self.tokenizer.tgt_lang = mapped_target
    
    def __call__(self, example):
        if self.restrict_source_language and example["source_lang"] != self.restrict_source_language:
            return {}
        source_text = example["source"].strip()
        target_text = example["target"].strip()
        if not target_text:
            return {}
        source_language = example["source_lang"]
        target_language = "en" if source_language == "fr" else "fr"
        self._setup_tokenizer_languages(source_language, target_language)
        source_tokens = self.tokenizer(source_text, truncation=True, max_length=self.max_source_length)
        target_tokens = self.tokenizer(text_target=target_text, truncation=True, max_length=self.max_target_length)
        if not target_tokens.get("input_ids"):
            return {}
        source_tokens["labels"] = target_tokens["input_ids"]
        
        if self.model_name == "m2m100_418m":
            mapped_target = self.language_map[target_language]
            target_language_id = self.tokenizer.get_lang_id(mapped_target)
            pad_token_id = self.tokenizer.pad_token_id
            labels = source_tokens["labels"]
            
            # Create decoder_input_ids with target language ID as first token
            decoder_input_ids = [target_language_id] + [
                (pad_token_id if token == -100 else token) for token in labels[:-1]
            ]
            source_tokens["decoder_input_ids"] = decoder_input_ids
        
        # Note: mBART50 models handle forced_bos_token_id differently during generation,
        # but during training, the DataCollatorForSeq2Seq handles this automatically
        # by shifting the labels to create decoder_input_ids
        
        return source_tokens


class M2MDataCollator:
    """Special data collator for M2M100 models that handles decoder_input_ids"""
    
    def __init__(self, tokenizer, model, label_pad_token_id=-100):
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id
        self.pad_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    def __call__(self, features):
        for f in features:
            f.pop("decoder_input_ids", None)
        
        batch = self.pad_collator(features)
        
        labels = batch["labels"]
        pad_id = self.tokenizer.pad_token_id
        labels_for_shift = torch.where(labels == -100, torch.tensor(pad_id, device=labels.device), labels)
        first_tok = labels_for_shift[:, :1]
        shifted = torch.cat([first_tok, labels_for_shift[:, :-1]], dim=1)
        batch["decoder_input_ids"] = shifted
        return batch


def filter_dataset_by_model(dataset, model_config):
    if "restrict_source_language" not in model_config:
        return dataset
    allowed_lang = model_config["restrict_source_language"]
    return dataset.filter(lambda x: x["source_lang"] == allowed_lang)


def build_trainer(which, tokenizer, model, dataset_processed, output_directory, learning_rate, batch_size, grad_accum,
                  epochs, max_steps, eval_steps, logging_steps, save_steps, bf16, fp16, seed, warmup_ratio,
                  disable_tqdm, no_qlora):
    if which == "m2m100_418m":
        data_collator = M2MDataCollator(tokenizer, model)
    else:
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_directory,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        num_train_epochs=0.0 if max_steps else epochs,
        max_steps=max_steps if max_steps else -1,
        eval_strategy="steps",
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=False,
        report_to=["none"],
        bf16=bf16,
        fp16=fp16 and not bf16,
        seed=seed,
        warmup_ratio=warmup_ratio,
        gradient_checkpointing=not no_qlora,
        label_smoothing_factor=0.1,
        dataloader_num_workers=2,
        disable_tqdm=disable_tqdm,
        lr_scheduler_type="linear",
        weight_decay=0.01,
        ddp_find_unused_parameters=False if is_distributed() else None,
        label_names=["labels"],
    )
    try:
        trainer = Seq2SeqTrainer(model=model, args=training_args,
                                 train_dataset=dataset_processed["train"],
                                 eval_dataset=dataset_processed["eval"],
                                 processing_class=tokenizer, data_collator=data_collator,
                                 callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
    except TypeError:
        trainer = Seq2SeqTrainer(model=model, args=training_args,
                                 train_dataset=dataset_processed["train"],
                                 eval_dataset=dataset_processed["eval"],
                                 tokenizer=tokenizer, data_collator=data_collator,
                                 callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
    return trainer


def finetune_model(which, data_path, output_directory,
                   learning_rate=2e-4, batch_size=8, grad_accum=2, epochs=2.0,
                   eval_steps=1000, logging_steps=50, save_steps=1000,
                   seed=42, warmup_ratio=0.03, val_ratio=0.05,
                   max_source_len=512, max_target_len=512,
                   bf16=False, fp16=False, no_qlora=False, device_map="auto", disable_tqdm=True,
                   lora_r=16, lora_alpha=32, lora_dropout=0.05):
    if which not in MODELS:
        raise ValueError(f"Model '{which}' not found. Available models: {list(MODELS.keys())}")
    
    model_info = MODELS[which]
    
    is_opus_model = "opus_mt" in which
    
    raw = load_dataset("json", data_files=data_path, split="train")
    raw = filter_dataset_by_model(raw, model_info)
    
    if len(raw) == 0:
        raise ValueError(f"No data remaining after filtering for model {which}")
    
    split = raw.train_test_split(test_size=val_ratio, seed=seed)
    train_ds = split["train"].shuffle(seed=seed)
    eval_ds = split["test"].shuffle(seed=seed)
    
    if not no_qlora:
        resolved_device_map = None if is_opus_model else device_map
    else:
        if is_distributed():
            resolved_device_map = None
        elif is_opus_model:
            resolved_device_map = None
        else:
            resolved_device_map = device_map
    
    tokenizer, _ = load_tokenizer_and_model(model_info["model_id"], use_qlora=not no_qlora, use_bfloat16=bf16,
                                            device_map=resolved_device_map)
    
    def preprocess(ds):
        pre = Preprocessor(model_name=which, tokenizer=tokenizer, language_map=model_info["language_map"],
                           max_source_length=max_source_len, max_target_length=max_target_len,
                           restrict_source_language=model_info.get("restrict_source_language"))
        out = ds.map(pre, remove_columns=ds.column_names, load_from_cache_file=False)
        out = out.filter(
            lambda x: "input_ids" in x and "labels" in x and x["labels"] is not None and len(x["labels"]) > 0,
            load_from_cache_file=False)
        return out
    
    dataset_processed = {"train": preprocess(train_ds), "eval": preprocess(eval_ds)}
    
    if len(dataset_processed["train"]) == 0:
        raise ValueError(f"No training examples remaining after preprocessing for model {which}")
    if len(dataset_processed["eval"]) == 0:
        raise ValueError(f"No evaluation examples remaining after preprocessing for model {which}")
    
    setup_logging(output_directory)
    _, base = load_tokenizer_and_model(model_info["model_id"], use_qlora=not no_qlora, use_bfloat16=bf16,
                                       device_map=resolved_device_map)
    model = attach_lora(base, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
    steps_per_epoch = math.ceil(len(dataset_processed["train"]) / (batch_size * grad_accum))
    logging.info(
        f"sizes | train={len(dataset_processed['train'])} eval={len(dataset_processed['eval'])} steps/epoch≈{steps_per_epoch}")
    
    trainer = build_trainer(which, tokenizer, model, dataset_processed, output_directory, learning_rate, batch_size,
                            grad_accum,
                            epochs, None, eval_steps, logging_steps, save_steps, bf16, fp16, seed, warmup_ratio,
                            disable_tqdm, no_qlora)
    trainer.train()
    model.save_pretrained(os.path.join(output_directory, "lora"))
    tokenizer.save_pretrained(output_directory)
    with open(os.path.join(output_directory, "finished.json"), "w", encoding="utf-8") as f:
        json.dump({"status": "ok"}, f)
