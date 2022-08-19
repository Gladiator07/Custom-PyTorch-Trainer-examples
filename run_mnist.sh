accelerate config # this should be run first
accelerate launch mnist.py \
            --config mnist_config.py \
            --config.trainer_args.num_train_epochs 5
