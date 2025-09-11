import os, torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

translation_models = {
    "m2m100_418m_25k": {
        "base_model": "../Data/merged/m2m100_418m",
        "lora_dir": "outputs_25k/m2m100_418m/lora",
        "out_dir": "../Data/merged_25k/m2m100_418m"
    },
    "mbart50_mmt_fr_25k": {
        "base_model": "../Data/merged/mbart50_mmt_fr",
        "lora_dir": "outputs_25k/mbart50_mmt_fr/lora",
        "out_dir": "../Data/merged_25k/mbart50_mmt_fr"
    },
    "mbart50_mmt_en_25k": {
        "base_model": "../Data/merged/mbart50_mmt_en",
        "lora_dir": "outputs_25k/mbart50_mmt_en/lora",
        "out_dir": "../Data/merged_25k/mbart50_mmt_en"
    },
    "opus_mt_en_fr_25k": {
        "base_model": "../Data/merged/opus_mt_en_fr",
        "lora_dir": "outputs_25k/opus_mt_en_fr/lora",
        "out_dir": "../Data/merged_25k/opus_mt_en_fr"
    },
    "opus_mt_fr_en_25k": {
        "base_model": "../Data/merged/opus_mt_fr_en",
        "lora_dir": "outputs_25k/opus_mt_fr_en/lora",
        "out_dir": "../Data/merged_25k/opus_mt_fr_en"
    },
    "m2m100_418m_100k": {
        "base_model": "../Data/merged/m2m100_418m",
        "lora_dir": "outputs_100k/m2m100_418m/lora",
        "out_dir": "../Data/merged_100k/m2m100_418m"
    },
    "mbart50_mmt_fr_100k": {
        "base_model": "../Data/merged/mbart50_mmt_fr",
        "lora_dir": "outputs_100k/mbart50_mmt_fr/lora",
        "out_dir": "../Data/merged_100k/mbart50_mmt_fr"
    },
    "mbart50_mmt_en_100k": {
        "base_model": "../Data/merged/mbart50_mmt_en",
        "lora_dir": "outputs_100k/mbart50_mmt_en/lora",
        "out_dir": "../Data/merged_100k/mbart50_mmt_en"
    },
    "opus_mt_en_fr_100k": {
        "base_model": "../Data/merged/opus_mt_en_fr",
        "lora_dir": "outputs_100k/opus_mt_en_fr/lora",
        "out_dir": "../Data/merged_100k/opus_mt_en_fr"
    },
    "opus_mt_fr_en_100k": {
        "base_model": "../Data/merged/opus_mt_fr_en",
        "lora_dir": "outputs_100k/opus_mt_fr_en/lora",
        "out_dir": "../Data/merged_100k/opus_mt_fr_en"
    },
}


def merge_one(base_model_id, lora_dir, out_dir, dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None): tok.pad_token = tok.eos_token
    base = AutoModelForSeq2SeqLM.from_pretrained(base_model_id, torch_dtype=dtype, trust_remote_code=True)
    if hasattr(base.config, "vocab_size") and len(tok) > base.config.vocab_size:
        base.resize_token_embeddings(len(tok), mean_resizing=False)
    peft = PeftModel.from_pretrained(base, lora_dir)
    merged = peft.merge_and_unload()
    os.makedirs(out_dir, exist_ok=True)
    merged.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)


def main():
    for name, cfg in translation_models.items():
        print(f'\nmerging {name}')
        merge_one(cfg["base_model"], cfg["lora_dir"], cfg["out_dir"])


if __name__ == "__main__":
    main()
