import json
import os
import subprocess


def run_command_in_container(container_name: str, command: str) -> str:
    """
    Run a command inside a Docker container.
    @param container_name: name of the container
    @param command: command string
    @return: command output
    """
    if not container_name or not command:
        raise ValueError("Container name and command cannot be empty")

    docker_command = ["docker", "exec", container_name] + command.split()

    try:
        result = subprocess.run(docker_command, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"Command failed with error: {result.stderr}")
        return result.stdout
    except subprocess.TimeoutExpired:
        print(f"Command timed out: {docker_command}")
        return ""
    except Exception as e:
        print(f"Error running command: {e}")
        return ""


def score_by_coverage(lua_str: str, current_coverage: float = 0.0) -> float:
    """
    Calculate the coverage score of a lua code snippet.
    @param lua_str: generated lua code snippet
    @param current_coverage: current coverage
    @return: tuple (float, float) containing coverage score and current coverage
    """

    # set up file paths
    test_case_path = "workdir/test.lua"
    # cov_res_path = "workdir/cov_report.json"

    # 1. write test case to temp file
    with open(test_case_path, "w") as f:
        f.write(lua_str)

    # 2. execute target on testcase
    container_name = "coverage_container"
    command = "/scripts/execute_target.sh"
    run_command_in_container(container_name, command)

    # 3. parse coverage result
    total_coverage = 0.0
    standalone_coverage = 0.0
    incremental_coverage = 0.0

    try:
        with open("temp/cov.report.json", "r") as f:
            coverage_data = json.load(f)
            if "Total Coverage" in coverage_data:
                total_coverage = coverage_data["Total Coverage"]
            elif "Standalone Coverage" in coverage_data:
                standalone_coverage = coverage_data["Standalone Coverage"]
            elif "Incremental Coverage" in coverage_data:
                incremental_coverage = coverage_data["Incremental Coverage"]
    except FileNotFoundError:
        print("Coverage file not found.")
    finally:
        os.remove(test_case_path)

    stand_alone_weight = 0.5  # weigh standalone coverage
    incremental_cov_weight = 2  # Incremental coverage might be more valuable
    total_cov_weight = 0.2  # lower weight for total coverage
    penalty_no_improvement = 1.0  # penalty for no coverage improvement

    # reward components
    standalone_reward = stand_alone_weight * standalone_coverage
    incremental_reward = incremental_cov_weight * incremental_coverage
    total_coverage_reward = total_cov_weight * total_coverage

    # penalty
    penalty = (
        penalty_no_improvement if standalone_coverage == 0 and incremental_coverage == 0 else 0
    )

    # final reward calculation
    reward = standalone_reward + incremental_reward + total_coverage_reward - penalty

    return reward
