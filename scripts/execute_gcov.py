import glob
import os
import re
import subprocess


def sort_by_time_and_id(input_dir, files):
    def get_ctime_and_id(file):
        # Get the creation time of the file
        ctime = os.path.getctime(os.path.join(input_dir, file))
        # Check if the file has an id and extract it if present, else use None
        match = re.search(r"id:(\d{6})", os.path.basename(file))
        file_id = int(match.group(1)) if match else None  # If no id, fallback to None
        return file_id, ctime

    return sorted(
        files,
        key=lambda f: (
            get_ctime_and_id(f)[0] is None,
            get_ctime_and_id(f)[0],
            get_ctime_and_id(f)[1],
        ),
    )


def generate_gcov_reports():
    shared = os.getenv("SHARED")
    out = f'{os.getenv("OUT")}/afl'

    os.chdir(out)

    c_files = glob.glob("*.c")

    input_dir = f"{shared}/inputs"

    input_files = os.listdir(input_dir)
    input_files_sorted = sort_by_time_and_id(input_dir, input_files)

    print(input_files_sorted)

    for file in input_files_sorted:
        print(f"Working on {file}")

        out_dir = f"{shared}/reports"
        os.makedirs(out_dir, exist_ok=True)

        try:
            print(f"Executing {os.path.join(input_dir, file)}")
            subprocess.run(["./lua", os.path.join(input_dir, file)], timeout=15)
        except subprocess.TimeoutExpired:
            continue

        file_name = os.path.basename(file)

        with open(f"{out_dir}/{file_name}.report", "w") as f:
            for c_file in c_files:
                res = subprocess.run(["gcov", "-b", c_file], stdout=subprocess.PIPE)

                f.write(res.stdout.decode("utf-8"))


if __name__ == "__main__":
    generate_gcov_reports()
