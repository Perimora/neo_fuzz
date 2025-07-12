import fcntl
import os
import re
import select
import subprocess
import threading
from time import sleep
from typing import IO


def set_non_blocking(pipe: IO[str]) -> None:
    """
    Sets the file descriptor of the given pipe to non-blocking mode, allowing for asynchronous
    reading without blocking operations.

    @param pipe: The file descriptor or pipe to set to non-blocking mode.
    @type pipe: File-like object

    @return: None
    """
    fl = fcntl.fcntl(pipe, fcntl.F_GETFL)
    fcntl.fcntl(pipe, fcntl.F_SETFL, fl | os.O_NONBLOCK)


def extract_container_id(stdout_lines: list[str]) -> str | None:
    """
    Extracts the container ID from a list of stdout lines by searching for a specific ID pattern.

    @param stdout_lines: A list of strings representing lines of stdout output.
    @type stdout_lines: List of str

    @return: The extracted container ID if found, otherwise None.
    @rtype: Str or None
    """
    for line in stdout_lines:
        match = re.search(r"\(ID:\s([a-f0-9]+)\)", line)
        if match:
            return match.group(1)
    return None


class MagmaContainer:

    def __init__(self, name: str, time_limit: str = "24h"):
        """
        Initializes a new instance of the class, which starts a Docker container in a non-blocking subprocess.
        The initial output is read to extract the container ID.

        @param name: The name to assign to this instance, typically used for identifying or managing the container.
        @type name: Str

        @param time_limit: The time limit for a possible Fuzzer Campaign, specified as a string (e.g., '24h').
        @type time_limit: Str, optional

        @return: None
        """
        # initialize attributes
        self.docker_process: subprocess.Popen[str] | None = None
        self.container_id: str | None = None
        self.name: str = name

        # run the subprocess without blocking
        try:
            self.docker_process = subprocess.Popen(
                ["magma/tools/captain/run_container.sh", "-d", "workdir", "-t", f"{time_limit}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except (FileNotFoundError, OSError) as e:
            print(f"Error starting Docker container: {e}")
            self.docker_process = None
            self.container_id = None
            return

        # make stdout and stderr non-blocking
        if self.docker_process.stdout and self.docker_process.stderr:
            set_non_blocking(self.docker_process.stdout)
            set_non_blocking(self.docker_process.stderr)

        # sleep for setup completion
        sleep(5)

        # read the initial output from the startup
        output, _ = self.read_initial_output()
        print("Initial STDOUT:", output)

        # set internal name
        self.name = name

        # extract and set the container ID
        self.container_id = extract_container_id(output)
        if self.container_id:
            print(f"Extracted Container ID: {self.container_id}")
        else:
            print("Container ID not found.")

    def read_initial_output(self) -> tuple[list[str], list[str]]:
        """
        Reads the initial output from the Docker subprocess for a short period, capturing both stdout and stderr.
        Uses a 2-second timeout for non-blocking read operations and returns the output as lists of lines.

        @return: A tuple containing two lists: stdout lines and stderr lines.
        @rtype: Tuple of (list of str, list of str)
        """
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        # Check if docker_process and its pipes are available
        if (
            self.docker_process is None
            or self.docker_process.stdout is None
            or self.docker_process.stderr is None
        ):
            return stdout_lines, stderr_lines

        # use select to read only the initial output (for a short period)
        ready_to_read, _, _ = select.select(
            [self.docker_process.stdout, self.docker_process.stderr], [], [], 2
        )  # 2-second timeout

        # read available stdout
        if self.docker_process.stdout in ready_to_read:
            try:
                while True:
                    line = self.docker_process.stdout.readline()
                    if not line:
                        break
                    stdout_lines.append(line.strip())
            except BlockingIOError:
                pass  # No more data available

        # read available stderr
        if self.docker_process.stderr in ready_to_read:
            try:
                while True:
                    line = self.docker_process.stderr.readline()
                    if not line:
                        break
                    stderr_lines.append(line.strip())
            except BlockingIOError:
                pass  # No more data available

        # return the initial stdout and stderr as lists
        return stdout_lines, stderr_lines

    def __del__(self) -> None:
        """
        Destructor method that shuts down the Docker container associated with this instance.
        Executes a shutdown script using the container ID if available.

        @return: None
        """
        if self.container_id:
            subprocess.run(
                ["scripts/docker/shutdown_coverage_docker.sh", self.container_id],
                capture_output=True,
                text=True,
            )

    def run_command(self, command: str, time_out: int | None = None) -> str:
        """
        Run a command inside a Docker container. (BLOCKING)
        @param time_out: timeout in seconds
        @param command: command string
        @return: command output
        """
        if not hasattr(self, "container_id") or not self.container_id:
            print("Error: Container ID not available")
            return ""

        try:
            docker_command = ["docker", "exec", self.container_id] + command.split()
            result = subprocess.run(
                docker_command, capture_output=True, text=True, timeout=time_out
            )
            return result.stdout
        except subprocess.TimeoutExpired:
            print(f"Command timed out: {command}")
            return ""
        except subprocess.CalledProcessError as e:
            print(f"error: {e} while executing command: {command}")
            return ""
        except UnicodeDecodeError as e:
            print(f"error: {e} while executing command: {command}")
            return ""

    def run_command_in_thread(self, command: str) -> None:
        """
        Run a command inside a Docker container with threading. (NOT BLOCKING)
        @param command: command string
        @return: None
        """
        thread = threading.Thread(target=self.run_command, args=(command,))
        thread.start()

    def start_afl_campaign(self) -> None:
        """
        Start the AFL campaign for evaluation in the Docker container.
        @return:
        """
        self.run_command("./magma/magma/run.sh")

    def init_magma_monitor(self) -> None:
        """
        Initialize the Magma monitor inside the Docker container.
        @return:
        """
        self.run_command("./magma/magma/init_magma_monitor.sh")
