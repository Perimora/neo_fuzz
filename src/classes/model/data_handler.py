import glob
import json
import os
import re
import shutil
import subprocess
from typing import Any

from datasets import Dataset, load_dataset, load_from_disk
from tqdm import tqdm
from transformers import GPT2Tokenizer
from transformers.tokenization_utils_base import BatchEncoding
from trl.core import LengthSampler

from src.classes.controller.magma_container import MagmaContainer
from src.classes.enum.targets import Target
from src.classes.enum.train_datatypes import DataSetType


class DataHandler:

    def __init__(self, target: Target = Target.LUA):
        """
        Initializes the handler with the specified target, sets up dataset attributes, and defines
        directory paths for data storage and evaluation.

        @param target: The target platform or language, such as Lua, for which the handler is configured.
                       Defaults to Lua.
        @type target: Target
        """

        # set paths and target of handler
        self.target: Target = target

        # full the stack ds
        self.ds_full: Dataset | None = None
        # ssl datasets
        self.ssl_data: Dataset | None = None
        self.ssl_data_tokenized: Dataset | None = None

        # ppo datasets
        self.ppo_data: Dataset | None = None
        self.ppo_data_cleaned: Dataset | None = None
        self.ppo_data_tokenized: Dataset | None = None

        # backup dataset
        self.backup_data: Dataset | None = None
        self.backup_data_tokenized: Dataset | None = None

        # paths to datasets
        self.path_ssl_data: Dataset | None = None
        self.path_ssl_data_tokenized: Dataset | None = None
        self.path_ppo_data: Dataset | None = None
        self.path_ppo_data_cleaned: Dataset | None = None
        self.path_ppo_data_tokenized: Dataset | None = None
        self.path_backup_data: Dataset | None = None

        # path to evaluation dir
        self.path_eval_dir = "logs/eval"

        # init path strings with given target
        self.set_path_strings()

    def load_dataset_from_hub(self, name: str) -> Dataset | None:
        """
        Loads a dataset from the Hugging Face Hub, sets it as the primary dataset for the handler,
        and returns the dataset.

        @param name: The name of the dataset to load from the Hugging Face Hub.
        @type name: Str

        @return: The loaded dataset.
        @rtype: Dataset
        """
        self.ds_full = load_dataset(name, data_dir="data/lua", token=True)  # nosec B615
        return self.ds_full

    def load_dataset_from_file(
        self, path: str, d_type: DataSetType, tokenized: bool = False, clean: bool = False
    ) -> bool:
        """
        Loads a dataset from a specified file path and assigns it to the appropriate dataset attribute
        based on the dataset type and additional flags.

        @param path: The file path from which to load the dataset.
        @type path: Str

        @param d_type: The type of dataset to load (e.g., SSL, PPO, or BACKUP).
        @type d_type: DataSetType

        @param tokenized: Indicates whether to load the tokenized version of the dataset.
        @type tokenized: Bool, optional

        @param clean: For PPO datasets, indicates whether to load the cleaned version.
        @type clean: Bool, optional

        @return: A flag indicating if the dataset was successfully loaded.
        @rtype: Bool
        """
        r_flag = False
        try:
            if d_type == DataSetType.SSL:
                if tokenized:
                    self.ssl_data_tokenized = load_from_disk(path)
                    r_flag = True
                else:
                    self.ssl_data = load_from_disk(path)
                    r_flag = True
            elif d_type == DataSetType.PPO:
                if clean:
                    if tokenized:
                        self.ppo_data_tokenized = load_from_disk(path)
                        r_flag = True
                    else:
                        self.ppo_data_cleaned = load_from_disk(path)
                        r_flag = True
                else:
                    self.ppo_data = load_from_disk(path)
                    r_flag = True
            elif d_type == DataSetType.BACKUP:
                if tokenized:
                    self.backup_data_tokenized = load_from_disk(path)
                    r_flag = True
                else:
                    self.backup_data = load_from_disk(path)
                    r_flag = True
        except FileNotFoundError as e:
            print(e)

        return r_flag

    def remove_comments(
        self, p_ds: Dataset | None = None, target: Target = Target.LUA
    ) -> Dataset | None:
        """
        Removes comments from the specified dataset based on the target language's comment syntax.
        By default, removes Lua comments from the entire dataset or from a specified subset if provided.

        @param p_ds: An optional dataset subset to clean.
                       If not provided, the method uses the full dataset.
        @type p_ds: Dataset, optional

        @param target: The target language for which comments should be removed.
                       Only 'Lua' is supported by default.
        @type target: Target

        @return: The cleaned dataset with comments removed.
        @rtype: Dataset
        """
        if not self.ds_full and not p_ds:
            print(
                "Error: no dataset specified!\n" "Please specify a dataset to remove comments from."
            )

        # set regex for comment purge
        if target == Target.LUA:
            # regex for lua comment purge
            single_line_pattern = re.compile(r"--.*")
            multi_line_pattern = re.compile(r"(?s)--\[(=*)\[(.*?)\]\1\]", re.DOTALL)
        else:
            print(
                "No comment regex for given Target found!\n"
                "Please provide a valid regex to remove comments!"
            )
            return

        def remove_sample_comments(lua_string: str) -> str:
            """
            This function removes all multiline comments from a given lua code snippet.
            @param lua_string: Lua code string
            @return: cleaned lua code string
            """
            # remove multiline comments
            lua_string = re.sub(multi_line_pattern, "", lua_string)
            # remove single and inline comments
            lua_string = re.sub(single_line_pattern, "", lua_string)
            # remove blank lines
            lua_string = re.sub(r"\n\s*\n", "\n", lua_string)
            # remove leading and tailing whitespaces
            lua_string = lua_string.strip()
            return lua_string

        def handle_cleanup_batches(example: dict[str, str]) -> dict[str, list[str]]:
            """
            Wrapper for remove_sample_comments() to handle batched samples
            @param example:
            @return:
            """
            cleaned_content = [remove_sample_comments(content) for content in example["content"]]
            return {"content": cleaned_content}

        print("Starting cleaning process...")
        # use standard member field or argument if member is not set
        target_ds = p_ds
        if not target_ds:
            target_ds = self.ds_full

        if target_ds is None:
            print("Error: No dataset available for comment removal")
            return None

        # removing all columns but 'content'
        ds = target_ds["train"].select_columns(["content"])

        # set core count and batch size for preprocessing
        # you might want to tune these settings...
        num_proc = 16
        batch_size = 5000

        # remove comments
        print("Removing comments...")
        ds = ds.map(handle_cleanup_batches, batched=True, batch_size=batch_size, num_proc=num_proc)

        # make sure that there are no empty examples in the data set
        print("Check for empty entries...")
        filtered_ds = ds.filter(lambda example: example["content"].strip() != "", num_proc=num_proc)

        print("Cleaning process completed.")
        self.ds_full = filtered_ds
        return self.ds_full

    def generate_data_split(self) -> None:
        """
        Shuffles the full dataset and splits it into three subsets for SSL, PPO, and backup data.
        The data is split as follows:
            - SSL data: First 200k samples
            - PPO data: Next 50k samples
            - Backup data: Remaining samples (~650k)

        @return: None
        """
        if self.ds_full is None:
            print("Error: No dataset available for splitting")
            return

        # shuffle data set
        shuffled_ds = self.ds_full.shuffle(seed=42)

        # split data into datasets for ssl, ppo and backup
        # samples: ssl 200k -- ppo 50k -- backup ~650k
        self.ssl_data = shuffled_ds.select(range(200000))
        self.ppo_data = shuffled_ds.select(range(200000, 250000))
        self.backup_data = shuffled_ds.select(range(250000, len(shuffled_ds)))

    def save_to_disk(self, d_type: DataSetType) -> None:
        """
        Saves a dataset to disk based on the dataset type.

        @param d_type: The type of dataset to save
        @type d_type: DataSetType
        @return: None
        """
        try:
            if d_type == DataSetType.PPO:
                if self.ppo_data is None:
                    print("Error: PPO data not available for saving")
                    return
                if self.path_ppo_data is None:
                    print("Error: PPO data path not set")
                    return
                self.ppo_data.save_to_disk(self.path_ppo_data)
            elif d_type == DataSetType.PPO_CLEAN:
                if self.ppo_data_cleaned is None:
                    print("Error: Cleaned PPO data not available for saving")
                    return
                if self.path_ppo_data_cleaned is None:
                    print("Error: Cleaned PPO data path not set")
                    return
                self.ppo_data_cleaned.save_to_disk(self.path_ppo_data_cleaned)
            elif d_type == DataSetType.SSL:
                if self.ssl_data is None:
                    print("Error: SSL data not available for saving")
                    return
                if self.path_ssl_data is None:
                    print("Error: SSL data path not set")
                    return
                self.ssl_data.save_to_disk(self.path_ssl_data)
            elif d_type == DataSetType.BACKUP:
                if self.backup_data is None:
                    print("Error: Backup data not available for saving")
                    return
                if self.path_backup_data is None:
                    print("Error: Backup data path not set")
                    return
                self.backup_data.save_to_disk(self.path_backup_data)
        except Exception as e:
            print(f"Error saving dataset: {e}")

    def ppo_data_cleanup(self) -> None:
        """
        Cleans up the PPO dataset by removing invalid Lua entries based on a custom validation function.
        Any entries that do not pass the validation are removed, and the cleaned dataset is saved.

        @return: None
        """
        print("Starting PPO data cleanup...")
        # create temp dir for data clean up
        os.makedirs("temp", exist_ok=True)
        # check if ppo data is available
        if not self.ppo_data:
            print("Error: no ppo dataset specified!\n")
            try:
                os.removedirs("temp")
            except OSError:
                pass
            return

        count_err = 0

        valid_entries = []

        for entry in tqdm(self.ppo_data["content"]):
            # Ensure entry is a string
            entry_str = str(entry) if not isinstance(entry, str) else entry
            if not self.check_lua_w_teal(entry_str):
                # error while scoring
                count_err += 1
            else:
                valid_entries.append(entry_str)

        print(f"Removed entries due to encoding: {count_err}")

        # create new dataset from valid entries
        new_dataset = Dataset.from_dict({"content": valid_entries})

        # save new dataset
        self.ppo_data_cleaned = new_dataset

        print(f"Valid entries: {len(valid_entries)}")

    def tokenize_ssl(self, tokenizer: GPT2Tokenizer, tokenizer_path: str) -> None:
        """
        Tokenizes the SSL dataset using the specified tokenizer and saves the tokenized dataset to disk.
        Sets the tokenizer padding token, applies tokenization to the content, and stores the tokenizer.

        @param tokenizer: The tokenizer to use for tokenizing the SSL dataset.
        @type tokenizer: GPT2Tokenizer

        @param tokenizer_path: The file path where the tokenizer will be saved after tokenization.
        @type tokenizer_path: Str

        @return: None
        """
        tokenizer.pad_token = "[PAD]"  # nosec

        # define tokenize function
        def tokenize_function(examples: dict[str, list[str]]) -> BatchEncoding | Any:
            return tokenizer(
                examples["content"],  # select content column to tokenize
                truncation=True,  # truncate if necessary
                max_length=512,  # set max length
                padding="max_length",  # set padding strategy
                return_tensors="pt",  # set pytorch flag
            )

        # load dataset
        if self.ssl_data:
            ssl_train_ds = self.ssl_data
        else:
            if self.path_ssl_data is None:
                print("Error: SSL data path not set")
                return
            ssl_train_ds = load_from_disk(self.path_ssl_data)

        # call tokenize function to tokenize the whole lua stack
        tokenized_ds = ssl_train_ds.map(
            tokenize_function, batched=True, remove_columns=["content"], num_proc=16
        )

        self.ssl_data_tokenized = tokenized_ds
        if self.path_ssl_data_tokenized:
            tokenized_ds.save_to_disk(self.path_ssl_data_tokenized)

        # save the tokenizer
        tokenizer.save_pretrained(tokenizer_path)

    def build_ds_ppo(
        self,
        tokenizer: GPT2Tokenizer,
        input_min_text_length: int = 5,
        input_max_text_length: int = 512,
    ) -> bool:
        """
        Prepares and tokenizes the PPO dataset using the specified tokenizer, and saves the tokenized dataset to disk.
        If necessary, it performs a cleanup of the PPO dataset before tokenization.

        @param tokenizer: The tokenizer used to tokenize the PPO dataset.
        @type tokenizer: GPT2Tokenizer

        @param input_min_text_length: The minimum text length for sampling the input sequence length.
        @type input_min_text_length: Int, optional

        @param input_max_text_length: The maximum text length for sampling the input sequence length.
        @type input_max_text_length: Int, optional

        @return: A flag indicating if the process was successful.
        @rtype: Bool
        """

        # load preprocessed ppo data
        if not self.ppo_data_cleaned:
            if self.path_ppo_data_cleaned:
                self.ppo_data_cleaned = load_from_disk(self.path_ppo_data_cleaned)

        if not self.ppo_data_cleaned:
            if self.ppo_data:
                # clean up was not yet done but can be done now
                self.ppo_data_cleanup()
            else:
                print("Error: Clean PPO data is not available!\n")
                return False

        if self.ppo_data_cleaned is None:
            print("Error: PPO data cleaning failed!")
            return False

        self.ppo_data_cleaned = self.ppo_data_cleaned.rename_columns({"content": "lua"})

        input_size = LengthSampler(input_min_text_length, input_max_text_length)

        def tokenize(sample: dict[str, Any]) -> dict[str, Any]:
            sample["input_ids"] = tokenizer.encode(sample["lua"])[: input_size()]
            sample["query"] = tokenizer.decode(sample["input_ids"])
            return sample

        self.ppo_data_tokenized = self.ppo_data_cleaned.map(tokenize, batched=False)
        self.ppo_data_tokenized.set_format(type="torch")
        if self.path_ppo_data_tokenized:
            self.ppo_data_tokenized.save_to_disk(self.path_ppo_data_tokenized)

        return True

    # setter
    def set_target(self, target: Target) -> None:
        """
        Sets the target platform or language for the handler.

        @param target: The target for which datasets and paths will be configured.
        @type target: Target

        @return: None
        """
        self.target = target

    def set_path_strings(self) -> None:
        """
        Configures directory paths based on the current target setting, creating the necessary directories
        for storing PPO, SSL, and backup data.
        Paths are initialized based on the target platform and structured within subdirectories.

        @return: None
        """

        # Base path depending on the target value
        base_path = f"data/{self.target.value}"

        # Define and create PPO directories
        ppo_data_dir = os.path.join(base_path, "ppo")
        if not os.path.exists(ppo_data_dir):
            os.makedirs(ppo_data_dir)

        self.path_ppo_data = os.path.join(ppo_data_dir, "ppo_data")
        os.makedirs(self.path_ppo_data, exist_ok=True)
        self.path_ppo_data_cleaned = os.path.join(ppo_data_dir, "ppo_data_cleaned")
        os.makedirs(self.path_ppo_data_cleaned, exist_ok=True)
        self.path_ppo_data_tokenized = os.path.join(ppo_data_dir, "ppo_data_tokenized")
        os.makedirs(self.path_ppo_data_tokenized, exist_ok=True)

        # Define and create SSL directories
        ssl_data_dir = os.path.join(base_path, "ssl")
        if not os.path.exists(ssl_data_dir):
            os.makedirs(ssl_data_dir)

        self.path_ssl_data = os.path.join(ssl_data_dir, "ssl_data")
        self.path_ssl_data_tokenized = os.path.join(ssl_data_dir, "ssl_data_tokenized")

        # Define and create Backup directory
        backup_data_dir = os.path.join(base_path, "backup")
        if not os.path.exists(backup_data_dir):
            os.makedirs(backup_data_dir)

        self.path_backup_data = os.path.join(backup_data_dir, "backup_data")

    @staticmethod
    def score_by_coverage(
        lua_str: str, container: MagmaContainer, current_coverage: float = 0.0
    ) -> tuple[float, float]:
        """
        Calculate the coverage score of a lua code snippet.
        @param container:
        @param lua_str: generated lua code snippet
        @param current_coverage: current coverage
        @return: tuple (float, float) containing coverage score and current coverage
        """
        shared = "workdir"
        # set up file paths
        test_case_path = f"{shared}/inputs"
        os.makedirs(test_case_path, exist_ok=True)
        # 1. write test case to temp file
        with open(f"{test_case_path}/test.lua", "w") as f:
            f.write(lua_str)

        # 2. execute target on testcase
        container.run_command(
            f"python3 scripts/collect_coverage_ppo.py --previous_coverage {current_coverage}"
        )

        # 3. parse coverage result
        total_coverage = 0.0
        standalone_coverage = 0.0
        incremental_coverage = 0.0

        try:
            report = glob.glob(f"{shared}/reports/*.json")[0]
            if report:

                with open(report, "r") as f:
                    report_data = json.load(f)
                    standalone_coverage = report_data["standalone_coverage"]
                    if float(report_data["incremental_coverage"]) >= 0:
                        incremental_coverage = report_data["incremental_coverage"]
                    else:
                        incremental_coverage = 0.0
                    total_coverage = report_data["total_coverage"]
        except FileNotFoundError:
            print("Coverage file not found.")
        finally:
            os.remove(f"{test_case_path}/test.lua")
            for filename in glob.glob(f"{shared}/reports/*.json"):
                os.remove(filename)

        stand_alone_weight = 0.5  # weigh standalone coverage
        incremental_cov_weight = 2  # Incremental coverage might be more valuable
        total_cov_weight = 0.2  # lower weight for total coverage
        penalty_no_improvement = 1.0  # penalty for no coverage improvement

        # reward components
        standalone_reward = stand_alone_weight * standalone_coverage
        incremental_reward = incremental_cov_weight * incremental_coverage
        total_coverage_reward = total_cov_weight * total_coverage

        # penalty
        penalty = penalty_no_improvement if incremental_coverage == 0 else 0

        # final reward calculation
        reward = standalone_reward + incremental_reward + total_coverage_reward - penalty

        return reward, total_coverage

    @staticmethod
    def check_lua_w_teal(lua_str: str) -> float:
        os.makedirs("temp", exist_ok=True)
        # path handling for temp files
        path = "temp/temp.lua"

        with open(path, "w") as f:
            f.write(lua_str)
        try:
            # run teal for static analysis
            res = subprocess.run(
                ["tl", "check", path], capture_output=True, text=True, timeout=20, errors="ignore"
            )
        except subprocess.TimeoutExpired:
            print(f'Timeout expired: the command "tl check {path}" took too long.')
            return 0
        except UnicodeDecodeError:
            print(f"Could not decode lua code for {lua_str}")
            return 0

        # define max score and penalties
        max_score = 10

        syntax_penalty = 1.0
        semantic_penalty = 0.5

        if res.stderr:
            err_lines = res.stderr.splitlines()
            syntax_error_count = 0
            semantic_error_count = 0

            # check for syntax errors
            for i, line in enumerate(err_lines):
                match = re.match(r"(\d+)\s+syntax\s+error(?:s)?", line, re.IGNORECASE)
                if match:
                    syntax_error_count = int(match.group(1))
                    break

            if syntax_error_count > 0:
                # return total penalty for syntax errors
                return max_score - (syntax_penalty * syntax_error_count)

            # check for warnings and semantic errors
            for _, line in enumerate(err_lines):
                semantic_error_match = re.match(r"(\d+)\s+error(?:s)?", line, re.IGNORECASE)

                if semantic_error_match:
                    semantic_error_count = int(semantic_error_match.group(1))

            # calculate final score
            score = max_score - (semantic_penalty * semantic_error_count)

            # remove temp dir
            shutil.rmtree("temp")

            return score

        else:
            # no penalties found
            return max_score
