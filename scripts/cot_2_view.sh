
# for run in {1..1}
# do 
#     #python train_ts_cot_two.py --dataset ISRUC --gpu 2 --run_desc two_view_trend 
#     python train_ts_cot_two.py --dataset HAR --gpu 1 --run_desc two_view_1p --data_perc '1perc'
#     python train_ts_cot_two.py --dataset Epilepsy --gpu 1 --run_desc two_view_1p --data_perc '1perc'
#     python train_ts_cot_two.py --dataset ISRUC --gpu 1 --run_desc two_view_1p --data_perc '1perc'
    
# done


lab_percs=("train")

for lab_perc in "${lab_percs[@]}"
do
    # 
    # python train_ts_cot_two.py --dataset ECG --gpu 1 --run_desc test$lab_perc #--data_perc $lab_perc"erc" 
    python train_ts_cot_transfer.py --dataset HAR --gpu 1 --run_desc test$lab_perc
    # python train_ts_cot_two.py --dataset ISRUC --gpu 1 --run_desc dual_view_$lab_perc --data_perc $lab_perc"erc" 
    # python train_ts_cot_two.py --dataset Epilepsy --gpu 1 --run_desc dual_view_$lab_perc --data_perc $lab_perc"erc" 
    # python3 main.py --experiment_description $exp_desc --run_description $run --seed $i --selected_dataset $dataset --training_mode "ft_$lab_perc"   --device $device 
    # python3 main.py --experiment_description $exp_desc --run_description $run --seed $i --selected_dataset $dataset --training_mode "gen_pseudo_labels" 
    # python3 main.py --experiment_description $exp_desc --run_description $run --seed $i --selected_dataset $dataset --training_mode "SupCon" 
    # python3 main.py --experiment_description $exp_desc --run_description $run --seed $i --selected_dataset $dataset --training_mode "train_linear_SupCon_$lab_perc" 
    
done