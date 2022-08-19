accelerate config # this should be run first
accelerate launch text_classification.py \
            --config text_cls_config.py \
            --config.trainer_args.num_train_epochs 2
