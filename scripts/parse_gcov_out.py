import json
import os
import argparse
import re


def parse_coverage_from_gcov(gcov_file):
    percentages_lines = []
    percentages_branches = []
    num_files = 0

    with open(gcov_file, "r") as f:
        for line in f:
            if "Lines executed:" in line:
                # extract the percentage from the line
                coverage_percentage = line.split(":")[1].split("%")[0].strip()
                percentages_lines.append(float(coverage_percentage))
                num_files += 1
            if "Branches executed" in line:
                branch_percentage = line.split(":")[1].split("%")[0].strip()
                percentages_branches.append(float(branch_percentage))

    if num_files:

        total_line_coverage = sum(percentages_lines) / num_files
        total_branches_coverage = sum(percentages_branches) / num_files

    else:
        total_line_coverage = 0
        total_branches_coverage = 0

    return total_line_coverage, total_branches_coverage


def parse_all_gcov_files(directory, start_time, afl_flag, ppo_flag: bool = False):
    files = os.listdir(directory)
    files.sort(key=lambda x: os.path.getctime(os.path.join(directory, x)))

    for file in files:
        if afl_flag and not ppo_flag:
            match = re.search(r"time:(\d+)", file)
            if match:
                time_value = int(match.group(1))

        elif not afl_flag and not ppo_flag:
            timestamp_str = ".".join(file.split(".", 2)[:2])

        gcov_file_path = os.path.join(directory, file)
        if os.path.isfile(gcov_file_path):
            line_coverage, branch_coverage = parse_coverage_from_gcov(gcov_file_path)
            if line_coverage is not None and branch_coverage is not None:
                output_file = f"{gcov_file_path}_coverage.json"

                # create a dictionary to store the coverage information
                arithmetic_mean = (line_coverage + branch_coverage) / 2
                if afl_flag and not ppo_flag:
                    time_delta = time_value
                elif not afl_flag and not ppo_flag:
                    time_delta = float(timestamp_str) - start_time

                if ppo_flag:
                    time_delta = -1

                coverage_data = {
                    "time": time_delta,
                    "line_coverage": line_coverage,
                    "branch_coverage": branch_coverage,
                    "total_coverage": arithmetic_mean,
                }

                # write the dictionary to a json file
                with open(output_file, "w") as out_f:
                    json.dump(coverage_data, out_f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse gcov files in a specified directory.")
    parser.add_argument("gcov_directory", type=str, help="Path to the gcov output directory")
    parser.add_argument(
        "--start_time",
        type=float,
        required=False,
        help="UNIX timestamp for the start time (format: time.time())",
    )
    parser.add_argument(
        "--afl", action="store_true", help="Flag to specify AFL coverage (default: False)."
    )
    args = parser.parse_args()

    args = parser.parse_args()
    gcov_directory = args.gcov_directory
    start_time = args.start_time
    afl_flag = args.afl
    parse_all_gcov_files(gcov_directory, start_time, afl_flag)
