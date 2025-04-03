import os
import argparse
import subprocess


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_root", type=str, default="./experiments", help="Experiment code root")
    parser.add_argument("--paper_name",      type=str, default="segregated_temporal_probe", help="Paper name")
    parser.add_argument("--experiment_name", type=str, default="01_STP_on_benchmarks", help="Experiment name")
    parser.add_argument("--experiment_item", type=str, default="SHD_STBP", help="Item for an experiment")
    parser.add_argument("--data_root",       type=str, default="/benchmark_data", help="Dataset root")
    parser.add_argument("--device",          type=str, default="0")
    args = parser.parse_args()
    
    experiment_path = os.path.join(args.experiment_root, args.paper_name, args.experiment_name, args.experiment_item)
    py_file         = os.path.join(experiment_path, "main.py")
    config_file     = os.path.join(experiment_path, "config.toml")

    command = [
        "python3",     py_file, 
        "--config",    config_file, 
        "--device",    args.device, 
        "--data_root", args.data_root,
    ]
    os.environ["PYTHONUNBUFFERED"] = "1"
    log_root = "./experiments/logs"
    log_path = os.path.join(log_root, args.paper_name, args.experiment_name)
    os.makedirs(log_path, exist_ok=True)
    log_file = os.path.join(log_path, f"log_{args.experiment_item}.txt")
    print(f"The experiment is about to run. Check log at `{log_file}` for details.")
    with open(log_file, "w") as log_fp:
        process = subprocess.run(
            args   = command, 
            stdout = log_fp, 
            stderr = log_fp,
            text   = True,
        )

    if process.returncode == 0:
        print(f"Experiment `{py_file}` completed.")
        print(f"please see the result log at `{log_file}`.")
    else:
        print(f"Experiment `{py_file}` failed.")
        print(f"please check the log at `{log_file}`.")
