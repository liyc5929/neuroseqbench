
# PennTreeBank
python runner.py --experiment_name 02_training_algo_on_benchmarks --experiment_item PTB_TSTBP_feedforward --data_root /benchmark_data --device 0
python runner.py --experiment_name 02_training_algo_on_benchmarks --experiment_item PTB_TSTBP_recurrent --data_root /benchmark_data --device 0

python runner.py --experiment_name 02_training_algo_on_benchmarks --experiment_item PTB_Eprop_recurrent --data_root /benchmark_data --device 0

python runner.py --experiment_name 02_training_algo_on_benchmarks --experiment_item PTB_OTTT_feedforward --data_root /benchmark_data --device 0

python runner.py --experiment_name 02_training_algo_on_benchmarks --experiment_item PTB_SLTT_feedforward --data_root /benchmark_data --device 0

# PSMNIST
python runner.py --experiment_name 02_training_algo_on_benchmarks --experiment_item PSMNIST_TSTBP_feedforward --data_root /benchmark_data --device 0
python runner.py --experiment_name 02_training_algo_on_benchmarks --experiment_item PSMNIST_TSTBP_recurrent --data_root /benchmark_data --device 0

python runner.py --experiment_name 02_training_algo_on_benchmarks --experiment_item PSMNIST_Eprop_recurrent --data_root /benchmark_data --device 0

python runner.py --experiment_name 02_training_algo_on_benchmarks --experiment_item PSMNIST_OTTT_feedforward --data_root /benchmark_data --device 0

python runner.py --experiment_name 02_training_algo_on_benchmarks --experiment_item PSMNIST_SLTT_feedforward --data_root /benchmark_data --device 0

# Binary Adding
python runner.py --experiment_name 02_training_algo_on_benchmarks --experiment_item BinaryAdding_TSTBP_feedforward --data_root /benchmark_data --device 0
python runner.py --experiment_name 02_training_algo_on_benchmarks --experiment_item BinaryAdding_TSTBP_recurrent --data_root /benchmark_data --device 0

python runner.py --experiment_name 02_training_algo_on_benchmarks --experiment_item BinaryAdding_Eprop_recurrent --data_root /benchmark_data --device 0

python runner.py --experiment_name 02_training_algo_on_benchmarks --experiment_item BinaryAdding_OTTT_feedforward --data_root /benchmark_data --device 0

python runner.py --experiment_name 02_training_algo_on_benchmarks --experiment_item BinaryAdding_SLTT_feedforward --data_root /benchmark_data --device 0
