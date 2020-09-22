python train_baseline.py --gpunum 6 --dataset ml1m --basemodel autoint \
    --embedding_dim 32 --headnum 8 --attention_dim 64 --lr "[0.05, 0.05]" \
    --pkl_file test2.pkl --seed 81192