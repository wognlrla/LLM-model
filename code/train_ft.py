import torch
from datasets import load_dataset
from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments)
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss, MatryoshkaLoss
from sentence_transformers.sentence_transformer.training_args import BatchSamplers
from eval_ir import build_evaluator

model = SentenceTransformer("google/embeddinggemma-2", device="cuda",
    model_kwargs={"torch_dtype": torch.bfloat16},
    config_kwargs={"vision_config": None, "audio_config": None})

# gradient checkpointing은 Trainer 인자 대신 내부 HF 모델에 직접 켠다
# (Trainer 경유 시 every_n_layers TypeError 발생 → Q9)
hf = getattr(model[0], "auto_model", None) or getattr(model[0], "model", None)
hf.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

train = load_dataset("json", data_files="data/train.jsonl", split="train")

# MatryoshkaLoss로 감싸야 학습 후에도 256/128차원 절단이 유지됨
loss = MatryoshkaLoss(model, MultipleNegativesRankingLoss(model),
                      matryoshka_dims=[768, 512, 256, 128])

args = SentenceTransformerTrainingArguments(
    output_dir="runs/eg2-classroom",
    num_train_epochs=3,
    per_device_train_batch_size=8,        # OOM이면 4로
    gradient_accumulation_steps=2,
    learning_rate=2e-5, warmup_steps=0.1,  # Transformers v5: 비율을 float로
    bf16=True,
    batch_sampler=BatchSamplers.NO_DUPLICATES,   # 배치 내 중복 문서 방지
    prompts={"anchor": model.prompts["SearchQuery"],
             "positive": model.prompts["Document"]},
    eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
    logging_steps=10, report_to="none",
)

trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=train,
                                     loss=loss, evaluator=build_evaluator())
trainer.train()
model.save_pretrained("models/eg2-classroom")
