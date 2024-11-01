
# for run in {1..5}
# do 
#     # python train_ts_cot.py --dataset HAR --gpu 0 --run_desc three_view_1p --data_perc 'perc' --backbone_type TS_SEA
#     # python train_ts_cot.py --dataset Epilepsy --gpu 0 --run_desc three_view_1p --data_perc '1perc' --backbone_type TS_SEA
#     python train_ts_cot.py --dataset ISRUC --gpu 0 --run_desc few_label --data_perc '1perc' --backbone_type TS_SEA --eval --model_path save_dir/test/ISRUC/TS_SEA/20240408_040456/model.pkl
# done  "50p" "75p"


# lab_percs=("32" "64" "128" "256" "512" "1024")

# for lab_perc in "${lab_percs[@]}"
# do
#     # 
#     python train_ts_cot.py --dataset HAR --gpu 2 --run_desc repr_dim_$lab_perc --backbone_type TS_SEA   --batch_size  256 --decomp_mode generate --repr-dims $lab_perc
#     python train_ts_cot.py --dataset Epilepsy --gpu 2 --run_desc repr_dim_$lab_perc --backbone_type TS_SEA   --batch_size  256 --decomp_mode generate --repr-dims $lab_percv
#     # python train_ts_cot.py --dataset ISRUC --gpu 2 --run_desc mov_multi_$lab_perc --backbone_type TS_SEA --batch_size  256 --decomp_mode generate --repr-dims $lab_perc
#     # python train_ts_cot.py --dataset ISRUC --gpu 1 --run_desc tri_view_$lab_perc --data_perc $lab_perc"erc"  --backbone_type TS_SEA
#     # python train_ts_cot.py --dataset Epilepsy --gpu 1 --run_desc tri_view_$lab_perc --data_perc $lab_perc"erc"  --backbone_type TS_SEA
    
# done

python train_ts_cot.py --dataset RoadBank --gpu 1 --run_desc temperature --backbone_type TS_SEA   --batch_size 128 --decomp_mode generate  
# lab_percs=( "0.001" "0.01" "0.05" "0.1" "0.5" "1")

# for lab_perc in "${lab_percs[@]}"
# do
#     # 
#     # python train_ts_cot.py --dataset HAR --gpu 2 --run_desc prototype_lambda$lab_perc --backbone_type TS_SEA   --batch_size  256 --decomp_mode generate --prototype-lambda $lab_perc
#     # python train_ts_cot.py --dataset Epilepsy --gpu 2 --run_desc prototype_lambda$lab_perc --backbone_type TS_SEA   --batch_size  256 --decomp_mode generate --prototype-lambda $lab_perc
#     # python train_ts_cot.py --dataset Epilepsy --gpu 2 --run_desc temperature --backbone_type TS_SEA   --batch_size 256 --decomp_mode generate --temperature $lab_perc
#     # python train_ts_cot.py --dataset HAR --gpu 2 --run_desc temperature --backbone_type TS_SEA   --batch_size 256 --decomp_mode generate  --temperature $lab_perc
#     # python train_ts_cot.py --dataset ISRUC --gpu 2 --run_desc temperature --backbone_type TS_SEA   --batch_size 256 --decomp_mode generate --temperature $lab_perc
#     # python train_ts_cot.py --dataset ISRUC --gpu 2 --run_desc mov_multi_$lab_perc --backbone_type TS_SEA --batch_size  256 --decomp_mode generate --repr-dims $lab_perc
#     # python train_ts_cot.py --dataset ISRUC --gpu 1 --run_desc edf --decomp_mode generate --batch_size 512 --backbone_type TS_SEA --repr-dim 512
#     # python train_ts_cot.py --dataset Epilepsy --gpu 1 --run_desc tri_view_$lab_perc --data_perc $lab_perc"erc"  --backbone_type TS_SEA
    
# done

#python train_ts_cot_semi.py --dataset HAR --dataloader HAR  --gpu 2 --run_desc semi --data_perc 10perc