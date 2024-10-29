import argparse
import os
import subprocess

import config


def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    while True:
        output = process.stdout.readline()
        if output == "" and process.poll() is not None:
            break
        if output:
            print(output.strip())
    rc = process.poll()
    return rc


def main():
    parser = argparse.ArgumentParser(description="Run experiments on specified dataset")
    parser.add_argument("--output_dir", default="final_results", help="Output directory for all experiments")
    parser.add_argument("--dataset_name", default="eelgrass", choices=config.dataset_names, help="Dataset name")
    parser.add_argument("--percent_labeled", type=int, default=20, help="Percent of labeled data")
    parser.add_argument("--run_knn", action="store_true", help="Run KNN baseline")
    parser.add_argument("--run_clip_zero", action="store_true", help="Run clip zeroshot baseline")
    parser.add_argument("--run_clip_supervised", action="store_true", help="Run CLIP supervised")
    parser.add_argument("--run_gpt_alone", action="store_true", help="Run GPT alone")
    parser.add_argument("--run_gpt_tools", action="store_true", help="Run GPT with tools")
    parser.add_argument("--run_gpt_visrag", action="store_true", help="Run GPT with Vis RAG")
    parser.add_argument("--run_gpt_visrag_tools", action="store_true", help="Run GPT with Vis RAG and tools")
    parser.add_argument("--number_of_test_samples", type=int, default=100, help="Number of test samples to process")
    parser.add_argument("--seed", type=int, default=1994, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Create test set selection
    test_set_file = os.path.join(args.output_dir, "metadata", f"{args.dataset_name}_test_indices.json")
    if os.path.exists(test_set_file):
        print(f"Test set selection file already exists: {test_set_file}")
    else:
        print("Creating test set selection...")
        command = (
            f"python create_test_set_selection.py --dataset_name {args.dataset_name} --output_dir {args.output_dir}"
        )
        run_command(command)

    # Define variables common to many experiments

    # 1. Create flags for dataset specs
    common_dataset_flags: list[str] = [
        f"--dataset_name {args.dataset_name}",
        f"--percent_labeled {args.percent_labeled}",
        f"--test_order_path {args.output_dir}/metadata/",
        f"--seed {args.seed}",
        f"--number_of_test_samples {args.number_of_test_samples}",
        "--evaluate_on_test",
        "--use_cuda",
    ]
    log_folder_pre: str = f"{args.output_dir}/{args.dataset_name}__percent_labeled={args.percent_labeled}"

    # 2. Define CLIP specs
    embedding_model_name = "clip"
    clip_model_dir = f"{args.output_dir}/{args.dataset_name}__percent_labeled={args.percent_labeled}__clip_supervised"
    clip_model_path = f"{clip_model_dir}/best_model.pth"

    # 3. For GPT experiments: get tools, create flags
    tools: list[str] = config.dataset_name2tool_list[args.dataset_name]
    common_gpt_flags: list[str] = [
        f"--clip_model_path {clip_model_path}",
        f"--prompt_schema_name {args.dataset_name}",
        f"--tools {' '.join(tools)}",
    ]

    if args.run_knn:
        print("Running KNN baseline...")
        command = " ".join(
            [
                "python main_knn.py",
                *common_dataset_flags,
                f"--log_folder {log_folder_pre}__knn__embedding={embedding_model_name}__k=3",
                f"--embedding_model_name {embedding_model_name}",
                "--n_neighbors 3",
            ]
        )
        run_command(command)

    if args.run_clip_zero:
        print("Running CLIP zero shot...")
        command = " ".join(
            [
                "python main_clip_zero_shot.py",
                *common_dataset_flags,
                f"--log_folder {log_folder_pre}__clip_zeroshot",
            ]
        )
        run_command(command)

    if args.run_clip_supervised:
        print("Running CLIP supervised...")
        command = " ".join(
            [
                "python main_clip_supervised.py",
                *common_dataset_flags,
                f"--log_folder {clip_model_dir}",
                "--num_epochs 10",
            ]
        )
        run_command(command)

    if args.run_gpt_alone:
        print("Running GPT-alone...")
        command = " ".join(
            [
                "python main.py",
                *common_dataset_flags,
                f"--log_folder {log_folder_pre}__gpt_alone",
                *common_gpt_flags,
                "--rag_type NoContext",
                "--num_tool_rounds 0",
            ]
        )
        run_command(command)

    if args.run_gpt_tools:
        print("Running GPT with tools...")
        command = " ".join(
            [
                "python main.py",
                *common_dataset_flags,
                f"--log_folder {log_folder_pre}__gpt_tools",
                *common_gpt_flags,
                "--rag_type NoContext",
            ]
        )
        run_command(command)

    if args.run_gpt_visrag:
        print("Running GPT with Vis RAG...")
        command = " ".join(
            [
                "python main.py",
                *common_dataset_flags,
                f"--log_folder {log_folder_pre}__gpt_visrag",
                *common_gpt_flags,
                "--num_tool_rounds 0",
            ]
        )
        run_command(command)

    if args.run_gpt_visrag_tools:
        print("Running GPT with Vis RAG and tools...")
        command = " ".join(
            [
                "python main.py",
                *common_dataset_flags,
                f"--log_folder {log_folder_pre}__gpt_visrag_tools",
                *common_gpt_flags,
            ]
        )
        run_command(command)


if __name__ == "__main__":
    main()
