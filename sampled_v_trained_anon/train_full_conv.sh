CUDA_DEVICE=4
DEPTH=3
LOSS_THRESH=0.000001
SEED=175

for i in 3,500 5,500 10,500 20,500 40,500 80,500 160,500 320,500; 
do
    IFS=',' read conv_width lin_width <<< "${i}"
    echo "conv width is $conv_width"
    echo "lin width is $lin_width"
    echo "with seed: $SEED"
    echo "with depth: $DEPTH"
    echo "loss thresh: $LOSS_THRESH"
    env CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 holdout_twin_sanity_check_train.py \
        --binary_digits=True \
        --num_train_epochs=10000 \
        --num_perfect_models=1000 \
        --iter_until_exit=20000 \
        --lr_scheduling=True \
        --width=$lin_width \
        --depth=$DEPTH \
        --conv_width=$conv_width \
        --conv_depth=$DEPTH \
        --optimizer=Nero \
        --init_lr=0.001 \
        --seed=$SEED \
        --mode=conditional_sampling_seed \
        --loss_threshold=$LOSS_THRESH \
        --trained_constraints=True \
        --sampled_constraints=True \
        --num_train_examples=5 \
        --num_test_examples=50 \
        --directly_compute_measures=False \
        --save_models=False \
        --dataset_type=cifar10 \
        --network_arch=conv_adaptive \
        --result_dir=temp
done
