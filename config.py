import ml_collections as mlc

text_cfg = mlc.ConfigDict()
text_cfg.project = "Custom_Accelerate_Trainer_Tests" # wandb project name
text_cfg.log_to_wandb = True
text_cfg.train_batch_size = 32
text_cfg.val_batch_size = 32
text_cfg.model_ckpt = "roberta-base"
text_cfg.learning_rate = 3e-5

text_cfg.trainer_args = dict(
    output_dir="./outputs",
    num_train_epochs=5,
    gradient_accumulation_steps=1,
    max_grad_norm=1,
    mixed_precision="fp16",
    scheduler_type="cosine",
    num_warmup_steps=0.1,
    save_best_checkpoint=True,
    save_last_checkpoint=True,
    save_weights_only=True,
    metric_for_best_model="val/accuracy"
)
