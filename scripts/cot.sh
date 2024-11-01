# python train_ts_cot_two.py --dataset ISRUC  # --eval  --model_path /workspace/TS-CoT/pretrained_model/two_view/EDF_model.pkl
# python train_ts_cot_two.py --dataset HAR #  --eval  --model_path /workspace/TS-CoT/pretrained_model/two_view/EDF_model.pkl
#python train_ts_cot_two.py --dataset SleepEDF # --eval  --model_path /workspace/TS-CoT/save_dir/test/SleepEDF/TS_CoT/20240321_123957/model.pkl
#python train_ts_cot.py --dataset SleepEDF # --eval  --model_path /workspace/TS-CoT/save_dir/test/SleepEDF/TS_CoT/20240321_151831/model.pkl
#python train_ts_cot_copy.py --dataset EDF --eval   --model_path /workspace/TS-CoT/pretrained_model/two_view/Epi_model.pkl
# python train_ts_cot.py --dataset HAR --eval --model_path /workspace/TS-CoT/save_dir/test/HAR/TS_CoT/20231229_115509/model.pkl
# python train_ts_cot.py --dataset Epi --eval --model_path /workspace/TS-CoT/pretrained_model/best/Epi_three_model.pkl --gpu 1

for run in {1..4}
do 
    # python train_ts_cot_transfer.py --dataset FD-A --gpu 0 --run_desc FD-A-2-FD-B --target_dataset FD-B --seed $run 
    python train_ts_cot_transfer.py --dataset HAR --gpu 0 --run_desc HAR-2-Gesture --target_dataset Gesture --seed $run 
    # python train_ts_cot_transfer.py --dataset SleepEEG --gpu 0 --run_desc SleepEEG-2-Epilepsy --target_dataset Epilepsy --seed $run 
    # python train_ts_cot_transfer.py --dataset ECG --gpu 0 --run_desc ECG-2-EMG --target_dataset EMG --seed $run 
    # python train_ts_cot_transfer.py --dataset SleepEEG --gpu 0 --run_desc SleepEEG-2-FD-B --target_dataset FD-B --seed $run
    # python train_ts_cot_transfer.py --dataset SleepEEG --gpu 0 --run_desc SleepEEG-2-EMG --target_dataset EMG --seed $run
    # python train_ts_cot_transfer.py --dataset SleepEEG --gpu 0 --run_desc SleepEEG-2-Gesture --target_dataset Gesture --seed $run

done