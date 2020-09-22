python train.py --gpunum 7 --dataset ml1m --basemodel autoint --embedding_dim 32 \
    --headnum 8 --attention_dim 64 --embedding_dim_meta 32 --cluster_d 32 \
    --clusternum "[1, 3, 2, 1]" --inner_steps 3 --batchsize 32 --learning_rate "[0.07, 0.7]" \
    --update_lr "[0.05, (0, 0)]" --seed 81192 --spt_qry_split "max(1/8, 4)" \
    --pkl_file test.pkl