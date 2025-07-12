import subprocess


def execute_script(script_path: str) -> bool:
    try:
        result = subprocess.run([script_path], check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error executing script {script_path}: {e}")
        return False
