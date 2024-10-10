import os
import argparse
import subprocess


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_root", type=str, default="src/benchmark/experiments", help="Experiment code root")
    parser.add_argument("--experiment",      type=str, default="01_STP_on_benchmarks", help="Experiment name")
    parser.add_argument("--config",          type=str, default="SHD_STBP.toml", help="Argument config file name")
    parser.add_argument("--data_root",       type=str, default="/benchmark_data", help="Dataset root")
    parser.add_argument("--device",          type=str, default="0")
    args = parser.parse_args()
    
    py_file     = os.path.join(args.experiment_root, args.experiment, "main.py")
    config_file = os.path.join(args.experiment_root, args.experiment, "configs", args.config)

    command = [
        "python3",     py_file, 
        "--config",    config_file, 
        "--device",    args.device, 
        "--data_root", args.data_root,
    ]
    os.environ["PYTHONUNBUFFERED"] = "1"
    log_file = os.path.join(".", f"log__{args.experiment}__{args.config.split('.')[0]}.txt")
    with open(log_file, "w") as log_fp:
        process = subprocess.run(
            args   = command, 
            stdout = log_fp, 
            stderr = log_fp,
            text   = True,
        )

    if process.returncode == 0:
        print(f"Experiment `{py_file}` with `{config_file}` completed successfully.")
    else:
        print(f"Experiment `{py_file}` with `{config_file}` failed,")
        print(f"please check logs in `{log_file}`.")
