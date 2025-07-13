import json
import os
import re
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.classes.controller.training_env import TrainingEnv


def get_coverage_data(
    json_files: List[str], afl: bool = False
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Extracts coverage data from a list of JSON files and returns timestamps, line coverage,
    branch coverage, and arithmetic mean coverage. Files can be sorted based on a unique ID
    or a timestamp for accurate plotting.

    @param json_files: A list of paths to JSON files containing coverage data.
    @type json_files: List[str]

    @param afl: If True, sorts files based on an 'id' in the filename. Otherwise, sorts by 'time'
                in the JSON content.
    @type afl: Bool

    @return: A tuple containing four lists:
             - timestamps: List of time values in seconds (converted from ms if afl=True)
             - line_coverage: List of line coverage values from each JSON file
             - branch_coverage: List of branch coverage values from each JSON file
             - coverage_mean: List of arithmetic mean coverage values from each JSON file
    @rtype: Tuple[List[float], List[float], List[float], List[float]]
    """
    timestamps = []
    line_coverage = []
    branch_coverage = []
    coverage_mean = []

    # sort files for plotting
    if afl:

        def extract_id(file_name: str) -> int | float:
            match = re.search(r"id:(\d+)", file_name)
            return int(match.group(1)) if match else float("inf")

        json_files = sorted(json_files, key=extract_id)
    else:

        def extract_time(p_input: str) -> float:
            try:
                with open(p_input, "r") as j_f:
                    j_data = json.load(j_f)
                    return float(j_data.get("time", float("inf")))
            except (json.JSONDecodeError, FileNotFoundError, PermissionError):
                return float("inf")

        json_files = sorted(json_files, key=extract_time)

    # extract report data
    for file in json_files:
        try:
            with open(file, "r") as f:
                try:
                    data = json.load(f)

                    if afl:
                        timestamps.append(data["time"] / 1000)  # afl queue timestamp is in ms
                    else:
                        timestamps.append(data["time"])

                    line_coverage.append(data["line_coverage"])
                    branch_coverage.append(data["branch_coverage"])
                    coverage_mean.append(data["total_coverage"])
                except json.JSONDecodeError:
                    continue
                except KeyError as e:
                    print(f"Warning: Missing key {e} in file {file}")
                    continue
        except (FileNotFoundError, PermissionError) as e:
            print(f"Warning: Could not read file {file}: {e}")
            continue

    return timestamps, line_coverage, branch_coverage, coverage_mean


class AflEvalView:

    def __init__(self, p_env: TrainingEnv):
        """
        Initializes the class with the given training environment.

        @param p_env: The training environment to be used.
        @type p_env: TrainingEnv
        """
        self.env = p_env

    def start_model_eval_process(self, t_limit: str = "24h") -> None:
        """
        Starts the model evaluation process by invoking the environment's model evaluation function.
        Generates and plots coverage data and visualizes vulnerabilities for the model.

        @param t_limit: The time limit for the evaluation, specified as a string (e.g., '24h').
                        Defaults to '24h'.
        @type t_limit: Str
        """
        json_files, monitor_reports, eval_d = self.env.start_model_eval(t_limit)
        self.plot_neo_coverage(json_files, eval_d)
        self.plot_reached_triggered_vulnerabilities_lua("GPT Neo", monitor_reports, eval_d)

    def start_afl_eval_process(self, time_limit: str = "24h") -> None:
        """
        Starts the AFL++ evaluation process by invoking the environment's AFL evaluation function.
        Generates and plots coverage data and visualizes vulnerabilities for AFL++.

        @param time_limit: The time limit for the evaluation, specified as a string (e.g., '24h').
                           Defaults to '24h'.
        @type time_limit: Str
        """
        cov_reports, monitor_reports, eval_dir = self.env.start_afl_eval(time_limit)
        self.plot_afl_coverage(cov_reports, eval_dir)
        self.plot_reached_triggered_vulnerabilities_lua("AFL++", monitor_reports, eval_dir)

    @staticmethod
    def plot_reached_triggered_vulnerabilities_lua(
        fuzzer: str, monitor_reports: List[str], eval_dir: str
    ) -> None:

        # sort files numerically based on the filename (convert the basename to int for sorting)
        try:
            csv_files = sorted(monitor_reports, key=lambda x: int(os.path.basename(x)))
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not sort CSV files: {e}")
            csv_files = monitor_reports

        # create an empty list to store dataframes
        dfs = []

        # iterate through each file, check if it's empty, and load the data
        for file in csv_files:
            try:
                if os.path.getsize(file) > 0:  # check if the file is not empty
                    try:
                        df = pd.read_csv(file)  # load each file
                        df["timestamp"] = os.path.basename(
                            file
                        )  # add a column for the timestamp from the filename
                        dfs.append(df)  # append the dataframe to the list
                    except pd.errors.EmptyDataError:
                        continue
                    except Exception as e:
                        print(f"Warning: Could not read CSV file {file}: {e}")
                        continue
                else:
                    print(f"File {file} is empty and was skipped.")
            except (FileNotFoundError, PermissionError, OSError) as e:
                print(f"Warning: Could not access file {file}: {e}")
                continue

        # concatenate all dataframes into one if any files were loaded
        if dfs:
            full_df = pd.concat(dfs, ignore_index=True)
        else:
            print("No vulnerabilities were reached or triggered.")
            return  # exit if no data was loaded

        # ensure all vulnerabilities (e.g., LUA001, LUA002, LUA003, LUA004) are in the DataFrame
        all_bugs = ["LUA001", "LUA002", "LUA003", "LUA004"]

        # add missing vulnerabilities to the DataFrame if they haven't been reached or triggered
        for bug in all_bugs:
            if f"{bug}_R" not in full_df.columns:
                full_df[f"{bug}_R"] = 0  # fill with 0 if not reached
            if f"{bug}_T" not in full_df.columns:
                full_df[f"{bug}_T"] = 0  # fill with 0 if not triggered

        # sorting the bugs
        bug_columns_r = [f"{bug}_R" for bug in all_bugs]
        bug_columns_t = [f"{bug}_T" for bug in all_bugs]

        # define fixed colors for each vulnerability
        color_map = {
            "LUA001": "blue",
            "LUA002": "green",
            "LUA003": "red",
            "LUA004": "orange",
        }

        # Convert the timestamp to hours (if it's not already)
        full_df["timestamp"] = (
            pd.to_numeric(full_df["timestamp"]) / 3600
        )  # Convert seconds to hours

        # set the x-axis ticks and format manually
        def set_x_axis_format(ax: Any) -> None:
            ax.set_xlim([0, 25])  # limit x-axis to 25 hours
            ax.set_xticks(np.arange(0, 25, 1))  # set major ticks every 1 hour
            ax.set_xticklabels(
                [f"{int(x)}" if x < 25 else "" for x in np.arange(0, 25, 1)]
            )  # label ticks as 0h to 24h
            ax.set_ylim([0, None])  # ensure y-axis starts from 0

        os.makedirs(f"{eval_dir}/diagrams", exist_ok=True)

        # Plot 1: Combined Reached vs Triggered for Vulnerabilities
        plt.figure(figsize=(10, 6))
        for bug_r, bug_t in zip(bug_columns_r, bug_columns_t):
            bug_name = bug_r.replace("_R", "")  # Extract the bug name without _R or _T
            plt.plot(
                full_df["timestamp"],
                full_df[bug_r],
                label=f"{bug_name} Reached",
                linestyle=":",
                color=color_map[bug_name],
            )
            plt.plot(
                full_df["timestamp"],
                full_df[bug_t],
                label=f"{bug_name} Triggered",
                linestyle="-",
                color=color_map[bug_name],
            )

        plt.title(f"Reached vs Triggered Vulnerabilities during {fuzzer} Evaluation")
        plt.xlabel("Time (Hours)")
        plt.ylabel("Count")
        plt.grid(True)

        # position the legend outside the plot to avoid collision
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))  # moves the legend outside the graph

        # set x-axis format and ensure y starts from 0
        set_x_axis_format(plt.gca())
        plt.tight_layout()

        plt.savefig(f"{eval_dir}/diagrams/{fuzzer}_reached_vs_triggered_plot.png", format="png")

        plt.show()

        # Plot 2: Reached for Vulnerabilities Only
        plt.figure(figsize=(10, 6))
        for bug_r in bug_columns_r:
            bug_name = bug_r.replace("_R", "")  # extract the bug name without _R or _T
            plt.plot(
                full_df["timestamp"],
                full_df[bug_r],
                label=f"{bug_name} Reached",
                linestyle=":",
                color=color_map[bug_name],
            )

        plt.title(f"Reached Vulnerabilities during {fuzzer} Evaluation")
        plt.xlabel("Time (in Hours)")
        plt.ylabel("Reached Count")
        plt.grid(True)

        # position the legend outside the plot to avoid collision
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

        # set x-axis format and ensure y starts from 0
        set_x_axis_format(plt.gca())
        plt.tight_layout()
        plt.savefig(f"{eval_dir}/diagrams/{fuzzer}_reached_plot.png", format="png")
        plt.show()

        # Plot 3: Triggered for Vulnerabilities Only
        plt.figure(figsize=(10, 6))
        for bug_t in bug_columns_t:
            bug_name = bug_t.replace("_T", "")  # extract the bug name without _R or _T
            plt.plot(
                full_df["timestamp"],
                full_df[bug_t],
                label=f"{bug_name} Triggered",
                linestyle="-",
                color=color_map[bug_name],
            )

        plt.title(f"Triggered Vulnerabilities during {fuzzer} Evaluation")
        plt.xlabel("Time (in Hours)")
        plt.ylabel("Triggered Count")
        plt.grid(True)

        # position the legend outside the plot to avoid collision
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

        # set x-axis format and ensure y starts from 0
        set_x_axis_format(plt.gca())
        plt.tight_layout()
        plt.savefig(f"{eval_dir}/diagrams/{fuzzer}_triggered_plot.png", format="png")
        plt.show()

    @staticmethod
    def plot_afl_coverage(jsonfiles: List[str], eval_dir: str) -> None:
        """
        Plots coverage metrics (line, branch, and arithmetic mean coverage) over time for AFL++.

        @param jsonfiles: A list of paths to JSON files containing coverage data.
        @type jsonfiles: List

        @param eval_dir: The directory where the plot image will be saved.
        @type eval_dir: Str
        """

        # Get data
        timestamps, line_coverage, branch_coverage, coverage_mean = get_coverage_data(
            jsonfiles, afl=True
        )

        # Convert timestamps from seconds to hours
        hours = [ts / 3600 for ts in timestamps]

        # Check if we have data to plot
        if not line_coverage or not branch_coverage or not coverage_mean:
            print("Warning: No coverage data available for plotting")
            return

        # find the maximum coverage values across all metrics
        max_coverage = max(max(line_coverage), max(branch_coverage), max(coverage_mean))

        # create a figure and plot the metrics
        plt.figure(figsize=(10, 6))

        plt.title("Coverage Metrics for AFL++ Evaluation")
        # plot line coverage
        plt.plot(hours, line_coverage, label="Line Coverage")

        # plot branch coverage
        plt.plot(hours, branch_coverage, label="Branch Coverage")

        # plot arithmetic mean
        plt.plot(hours, coverage_mean, label="Arithmetic Mean Coverage", linestyle="--")

        plt.grid(True)
        plt.xlabel("Time (hours)")  # x-axis label
        plt.ylabel("Coverage (%)")  # y-axis label

        # limit x-axis to slightly above 24 to keep the spacing
        plt.xlim(0, 24.5)

        # set the x-ticks explicitly to show only 0 to 24 hours
        plt.xticks(range(0, 25, 1))  # showing every 2 hours

        # set the y-axis limits, with a hard lower limit of 82%
        plt.ylim(81.5, max_coverage + 0.5)

        # show legend
        plt.legend()

        os.makedirs(f"{eval_dir}/diagrams", exist_ok=True)
        plt.savefig(f"{eval_dir}/diagrams/afl_coverage.png", format="png")

        # display the plot
        plt.show()

    @staticmethod
    def plot_neo_coverage(jsonfiles: List[str], eval_dir: str) -> None:
        """
        Plots coverage metrics (line, branch, and arithmetic mean coverage) over time for the GPT Neo evaluation.

        @param jsonfiles: A list of paths to JSON files containing coverage data.
        @type jsonfiles: list

        @param eval_dir: The directory where the plot image will be saved.
        @type eval_dir: str
        """
        timestamps, line_coverage, branch_coverage, coverage_mean = get_coverage_data(jsonfiles)

        # convert timestamps (time deltas in seconds) to hours
        hours = [ts / 3600 for ts in timestamps]

        # Check if we have data to plot
        if not line_coverage or not branch_coverage or not coverage_mean:
            print("Warning: No coverage data available for plotting")
            return

        # find the maximum coverage values across all metrics
        max_coverage = max(max(line_coverage), max(branch_coverage), max(coverage_mean))

        # create a figure and plot the metrics
        plt.figure(figsize=(10, 6))

        # plot line coverage
        plt.plot(hours, line_coverage, label="Line Coverage")

        # plot branch coverage
        plt.plot(hours, branch_coverage, label="Branch Coverage")

        # plot arithmetic mean
        plt.plot(hours, coverage_mean, label="Arithmetic Mean Coverage", linestyle="--")

        plt.grid(True)
        plt.xlabel("Time (hours)")  # x-axis label
        plt.ylabel("Coverage (%)")  # y-axis label

        # limit x-axis to slightly above 24 to keep the spacing
        plt.xlim(0, 24.5)

        # set the x-ticks explicitly to show only 0 to 24 hours
        plt.xticks(range(0, 25, 1))  # showing every 2 hours

        # set the y-axis limits, with a hard lower limit of 82%
        plt.ylim(81.5, max_coverage + 0.5)

        # show legend
        plt.legend()

        os.makedirs(f"{eval_dir}/diagrams", exist_ok=True)
        plt.savefig(f"{eval_dir}/diagrams/neo_coverage_plot.png", format="png")

        # display the plot
        plt.show()


def load_json_data(directory: str) -> List[str]:
    """
    Loads all JSON files from the given directory, sorts them by the 'id' in the filename,
    and returns a list of dictionaries with the data.

    :param directory: Path to the directory containing JSON files.
    :return: List of dictionaries with the parsed JSON data.
    """
    json_data: list[str] = []

    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Warning: Directory {directory} does not exist")
        return json_data

    # Regex pattern to extract the numeric 'id' from the filename
    id_pattern = re.compile(r"test_case_(\d+)\.json")

    # Create a list of files and extract the 'id' from each filename
    files_with_ids = []
    try:
        for filename in os.listdir(directory):
            if filename.endswith(".json"):
                match = id_pattern.search(filename)
                if match:
                    file_id = int(match.group(1))
                    file_path = os.path.join(directory, filename)
                    files_with_ids.append((file_id, file_path))
    except (PermissionError, OSError) as e:
        print(f"Warning: Could not access directory {directory}: {e}")
        return json_data

    # Sort files by the extracted 'id'
    files_with_ids.sort(key=lambda x: x[0])

    # Load the JSON data in the sorted order
    for file_id, file_path in files_with_ids:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                json_data.append(data)
        except (json.JSONDecodeError, FileNotFoundError, PermissionError) as e:
            print(f"Warning: Could not load {file_path}: {e}")
            continue

    return json_data
