CUDA_DEVICE=0
DEPTH=7
LOSS_THRESH=0.000001
SEED=175

for width in 500 1000 2000 3000 4000 5000 6000 7000
do
    echo "width is $width"
    echo "with seed: $SEED"
    echo "with depth: $DEPTH"
    echo "loss thresh: $LOSS_THRESH"
    env CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 holdout_twin_sanity_check_train.py --binary_digits=True \
                                                                --num_train_epochs=10000 \
                                                                --num_perfect_models=1000 \
                                                                --iter_until_exit=20000 \
                                                                --lr_scheduling=True \
                                                                --width=$width \
                                                                --depth=$DEPTH \
                                                                --optimizer=Nero \
                                                                --init_lr=0.001 \
                                                                --seed=$SEED \
                                                                --mode=conditional_sampling_seed \
                                                                --trained_constraints=True \
                                                                --sampled_constraints=True \
                                                                --num_train_examples=5 \
                                                                --num_test_examples=50 \
                                                                --directly_compute_measures=False \
                                                                --save_models=False \
                                                                --dataset_type=mnist \
                                                                --network_arch=linear \
                                                                --result_dir=temp \
                                                                --mnist_first_class=4 \
                                                                --mnist_second_class=7
done
