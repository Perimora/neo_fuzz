import glob
import logging
import os
import random
import re
import shutil
import subprocess
import time
from datetime import datetime
from typing import Any, no_type_check

import torch
import wandb
from datasets import tqdm
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from trl.core import LengthSampler

from src.classes.controller.magma_container import MagmaContainer
from src.classes.enum.targets import Target
from src.classes.enum.train_datatypes import DataSetType
from src.classes.enum.train_types import TrainType
from src.classes.model.data_handler import DataHandler

logger = logging.getLogger("app.dev")


def parse_time_input(time_str: str) -> int:
    """
    Parses a time string with hours, minutes, and seconds, and converts it to a total in seconds.
    The time string should include values followed by units,
    such as 'h' for hours, 'm' for minutes, and 's' for seconds.

    @param time_str: A string representing time with units (e.g., '2h30m15s').
    @type time_str: Str

    @return: The total time represented in seconds.
    @rtype: Int
    """

    # initialize total time in seconds
    total_seconds = 0

    # regex for finding time values followed by their units
    matches = re.findall(r"(\d+)([hms])", time_str)

    for value, unit in matches:
        value = int(value)
        if unit == "h":
            total_seconds += value * 3600  # convert hours to seconds
        elif unit == "m":
            total_seconds += value * 60  # convert minutes to seconds
        elif unit == "s":
            total_seconds += value  # already in seconds

    return total_seconds


class TrainingEnv:
    def __init__(self, model_name: str = "EleutherAI/gpt-neo-125M", target: Target = Target.LUA):
        """
        Initializes an instance with a specified model and target language, setting up paths,
        loading the model and tokenizer, and preparing data handlers and container collections.

        @param model_name: The name or path of the pre-trained model to load.
                Defaults to 'EleutherAI/gpt-neo-125M'.
        @type model_name: str, optional

        @param target: The target platform or language for configuring data handlers.
        Defaults to Lua.
        @type target: Target

        @return: None
        """

        # Load path configuration from env file
        load_dotenv("config/path.env")

        # neo model and tokenizer
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name)  # nosec
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)  # nosec

        # DataHandler collection
        self.data_handlers = [DataHandler(target)]

        # MagmaContainer collection
        self.magma_train_docker: list[MagmaContainer] = []
        self.magma_eval_docker: list[MagmaContainer] = []

        # path variables from environment
        self.path_model_dir = os.getenv("MODEL_DIR", "models")
        self.path_tokenizer_dir = os.getenv("TOKENIZER_DIR", "tokenizer")
        self.path_ssl_model = os.getenv("SSL_MODEL_PATH", "models/ssl")
        self.path_ppo_structure_model = os.getenv(
            "PPO_STRUCTURE_MODEL_PATH", "models/ppo/structure"
        )
        self.path_ppo_coverage_model = os.getenv("PPO_COVERAGE_MODEL_PATH", "models/ppo/coverage")

        # additional path variables
        self.path_eval_dir = os.getenv("EVAL_DIR", "logs/eval")
        self.path_shared_dir = os.getenv("SHARED_DIR", "workdir")
        self.path_corpus_dir = os.getenv("CORPUS_DIR", "data/lua/corpus_used_afl")

        self.init_path_variables()

    def run_ssl_training(self) -> None:
        """
        Executes the SSL (Self-Supervised Learning) training process on the specified dataset.
        This involves tokenizing the data if needed, setting up the training arguments, configuring
        the data collator and trainer, and saving the trained model and tokenizer.

        @return: None
        """

        # get corresponding DataHandler object
        lua_dh = self.get_data_handler_by_target()
        if lua_dh is None:
            print("Error: No data handler available for SSL training")
            return

        # tokenize ssl training data if necessary
        if not lua_dh.ssl_data_tokenized:
            lua_dh.tokenize_ssl(self.tokenizer, self.path_tokenizer_dir)
        # start the actual ssl training
        # reload tokenizer (added eos token as pad)
        self.tokenizer = AutoTokenizer.from_pretrained(self.path_tokenizer_dir)  # nosec
        # load tokenized data
        ds = lua_dh.ssl_data_tokenized
        if ds is None:
            print("Error: SSL data not available")
            return
        # create train eval split
        ds = ds.train_test_split(0.2)
        # setup WandB logging
        wandb.init()
        os.environ["WANDB_PROJECT"] = "<neo_lua_ssl>"
        os.environ["WANDB_LOG_MODEL"] = "checkpoint"

        # setup data paths
        model_output = self.path_ssl_model

        # setup data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        # define training arguments
        training_args = TrainingArguments(
            output_dir=self.path_ssl_model,
            overwrite_output_dir=True,
            num_train_epochs=1,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            eval_strategy="epoch",
            save_total_limit=2,
            report_to="wandb",
            run_name="neo_lua_ssl_200k_samples",
            weight_decay=0.01,
        )
        # setup trainer object
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=ds["train"],
            eval_dataset=ds["test"],
        )
        # start the actual training
        trainer.train()
        # save the neo_model and tokenizer
        self.model.save_pretrained(model_output)

    def run_ppo_training(self, t_type: TrainType) -> None:
        """
        Executes Proximal Policy Optimization (PPO) training based on the specified training type.
        PPO training can be performed for structure or coverage-based reinforcement learning,
        depending on the given type. The method configures training, prepares the necessary datasets,
        initializes a Docker container for coverage (if needed), and saves the trained model.

        @param t_type: Specifies the type of PPO training (either structure-based or coverage-based).
        @type t_type: TrainType

        @return: None
        """
        if t_type == TrainType.PPO_COVERAGE:
            # start coverage container
            ppo_cov_docker = MagmaContainer("ppo_cov")
            self.magma_train_docker.append(ppo_cov_docker)
            ppo_cov_docker.run_command("./scripts/neo/setup_docker_monitoring.sh")
            m_name = self.path_ppo_structure_model
            current_coverage = 0.0
        else:
            ppo_cov_docker = None
            current_coverage = 0.0
            m_name = self.path_ssl_model

        config = PPOConfig(
            model_name=m_name,
            learning_rate=1.41e-5,  # default learn rate
            batch_size=16,  # memory issues -- increase if possible for better training stability
            mini_batch_size=4,
            log_with="wandb",  # remove if you don't need
            ppo_epochs=1,
        )

        wandb.init()  # remove if you don't need

        # tokenize ppo dataset
        target_dh = self.get_data_handler_by_target()
        if target_dh is None:
            print("Error: No data handler available for PPO training")
            return

        if (
            not target_dh.ppo_data_tokenized
            and target_dh.path_ppo_data_tokenized
            and not target_dh.load_dataset_from_file(
                target_dh.path_ppo_data_tokenized, DataSetType.PPO, True, True
            )
        ):
            print("Start to tokenize PPO dataset")
            if not target_dh.build_ds_ppo(
                self.tokenizer, input_min_text_length=5, input_max_text_length=512
            ):
                print("Error: PPO data could not be built.")
                return

        def collator(data: list[dict[str, Any]]) -> dict[str, list[Any]]:
            return dict((key, [d[key] for d in data]) for key in data[0])

        if t_type == TrainType.PPO_STRUCTURE:
            ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(self.path_ssl_model)
            model_w_vh = AutoModelForCausalLMWithValueHead.from_pretrained(self.path_ssl_model)
        else:
            ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
                self.path_ppo_structure_model
            )
            model_w_vh = AutoModelForCausalLMWithValueHead.from_pretrained(
                self.path_ppo_structure_model
            )

        ppo_trainer = PPOTrainer(
            config,
            model_w_vh,
            ref_model,
            self.tokenizer,
            dataset=target_dh.ppo_data_tokenized,
            data_collator=collator,
        )

        generation_kwargs = {
            "min_length": 1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "max_new_tokens": 300,
        }

        if t_type == TrainType.PPO_COVERAGE:
            current_coverage = 0

        for _, batch in tqdm(enumerate(ppo_trainer.dataloader)):
            query_tensors = batch["input_ids"]

            # Get response from GPT NEO
            response_tensors = []
            for query in query_tensors:
                response = ppo_trainer.generate(query, **generation_kwargs)
                response_tensors.append(response.squeeze()[-generation_kwargs["max_new_tokens"] :])
            batch["response"] = [self.tokenizer.decode(r.squeeze()) for r in response_tensors]

            rewards = []

            if t_type == TrainType.PPO_STRUCTURE:
                for q, r in zip(batch["query"], batch["response"]):
                    # concatenate query and response for scoring
                    combined_code = q + r

                    # call deterministic score agent to get number of errors
                    data_handler = self.get_data_handler_by_target()
                    if data_handler is None:
                        print("Error: No data handler available")
                        continue
                    error_score = data_handler.check_lua_w_teal(combined_code)

                    # calculate the length of the response (r) and apply the penalty
                    if len(r) < 50:
                        score_value = error_score - 2
                    elif 50 <= len(r) < 75:
                        score_value = error_score + 3
                    else:
                        score_value = error_score + 5

                    # convert the score to a tensor
                    score_tensor = torch.tensor(score_value, dtype=torch.float)

                    # append the score tensor to the list
                    rewards.append(score_tensor)
            else:
                for q, r in zip(batch["query"], batch["response"]):
                    combined_code = q + r
                    rewards = []
                    data_handler = self.get_data_handler_by_target()
                    if data_handler is None or ppo_cov_docker is None:
                        print("Error: No data handler or coverage docker available")
                        continue
                    score, total_coverage = data_handler.score_by_coverage(
                        combined_code, ppo_cov_docker, current_coverage
                    )
                    rewards.append(torch.tensor(score))

                    if total_coverage > current_coverage:
                        current_coverage = total_coverage

            # Run PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)

        if t_type == TrainType.PPO_STRUCTURE:
            ppo_trainer.save_pretrained(self.path_ppo_structure_model)
        else:
            ppo_trainer.save_pretrained(self.path_ppo_coverage_model)

        # shutdown container
        if t_type == TrainType.PPO_COVERAGE:
            docker_to_remove = self.get_train_docker_by_name("ppo_cov")
            if docker_to_remove:
                docker_to_remove.__del__()

    def init_data(self) -> bool:
        """
        Initializes and prepares the datasets by loading, cleaning, and splitting data from a specified source.
        This includes removing comments, generating dataset splits for SSL, PPO, and backup, and saving these splits to disk.
        A secondary cleanup is also performed on the PPO dataset, and the cleaned version is saved.

        @return: A flag indicating if the data initialization process was successful.
        @rtype: Bool
        """

        logger.info("TrainingEnv: Starting data initialization workflow")

        # get corresponding DataHandler object
        lua_dh = self.get_data_handler_by_target()
        if lua_dh is None:
            logger.error("TrainingEnv: No data handler available for data initialization")
            return False

        # load dataset via Lua DataHandler of TrainingEnv
        logger.info("TrainingEnv: Loading dataset from hub...")
        lua_dh.load_dataset_from_hub("bigcode/the-stack")
        # remove all comments and whitespaces; filter empty entries
        logger.info("TrainingEnv: Removing comments and cleaning data...")
        lua_dh.remove_comments()
        # generate ssl, ppo and backup data split
        logger.info("TrainingEnv: Generating data splits...")
        lua_dh.generate_data_split()
        # save datasets to disk
        lua_dh.save_to_disk(DataSetType.SSL)
        lua_dh.save_to_disk(DataSetType.PPO)
        lua_dh.save_to_disk(DataSetType.BACKUP)
        # secondary ppo data cleanup
        lua_dh.ppo_data_cleanup()
        lua_dh.save_to_disk(DataSetType.PPO_CLEAN)

        logger.info("TrainingEnv: Data initialization completed successfully")
        return True

    # Setter
    def set_model(self, model: str) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(model)  # nosec

    def set_tokenizer(self, tokenizer: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)  # nosec

    def add_data_handler(self, data_handler: DataHandler) -> None:
        self.data_handlers.append(data_handler)

    def remove_data_handler(self, data_handler: DataHandler) -> None:
        self.data_handlers.remove(data_handler)

    def get_data_handler_by_target(self, target: Target = Target.LUA) -> DataHandler | None:
        for handler in self.data_handlers:
            if handler.target == target:
                return handler
        return None

    def add_magma_docker(self, magma_docker: MagmaContainer) -> None:
        self.magma_train_docker.append(magma_docker)

    def remove_magma_docker(self, magma_docker: MagmaContainer) -> None:
        self.magma_train_docker.remove(magma_docker)

    def init_path_variables(self) -> None:
        """
        Initializes the directory structure for the model and tokenizer paths.
        Creates necessary directories for SSL and PPO models, including subdirectories
        for structure and coverage training, and sets class path variables accordingly.

        @return: None
        """

        # create neo_model and tokenizer directories
        if not os.path.exists(self.path_model_dir):
            os.makedirs(self.path_model_dir)

        if not os.path.exists(self.path_tokenizer_dir):
            os.makedirs(self.path_tokenizer_dir)

        # create ./models/ssl directory and set the path_ssl_model variable
        ssl_dir = os.path.join(self.path_model_dir, "ssl")
        if not os.path.exists(ssl_dir):
            os.makedirs(ssl_dir)
        self.path_ssl_model = ssl_dir

        # create ./neo_model/ppo/structure directory and set the path_ppo_structure_model variable
        ppo_structure_dir = os.path.join(self.path_model_dir, "ppo", "structure")
        if not os.path.exists(ppo_structure_dir):
            os.makedirs(ppo_structure_dir)
        self.path_ppo_structure_model = ppo_structure_dir

        # create ./neo_model/ppo/coverage directory and set the path_ppo_coverage_model variable
        ppo_coverage_dir = os.path.join(self.path_model_dir, "ppo", "coverage")
        if not os.path.exists(ppo_coverage_dir):
            os.makedirs(ppo_coverage_dir)
        self.path_ppo_coverage_model = ppo_coverage_dir

    def run_syn_sem_eval(
        self, phase: TrainType, target: Target = Target.LUA
    ) -> tuple[int, float, float, str]:
        """
        Performs a syntactic and semantic evaluation on a set of generated code mutations.
        This method generates code based on truncated prompts, evaluates them, and saves the results
        to disk. The evaluation calculates and returns the total number of examples, along with
        percentages of accepted and rejected mutations.

        @param phase: The training phase (such as PPO or SSL) associated with this evaluation.
        @type phase: TrainType

        @param target: The target language for the evaluation. Defaults to Lua.
        @type target: Target

        @return: A tuple containing the total number of examples, percentage of accepted mutations,
                 percentage of rejected mutations, and the directory path for evaluation logs.
        @rtype: Tuple(int, float, float, str)
        """

        logger.info(f"Starting {phase.value} evaluation...")

        tokenizer = AutoTokenizer.from_pretrained(self.path_tokenizer_dir)  # nosec

        # load correct model for evaluation phase
        if phase is TrainType.SSL:
            self.model = AutoModelForCausalLM.from_pretrained(self.path_ssl_model)
        elif phase is TrainType.PPO_STRUCTURE:
            self.model = AutoModelForCausalLM.from_pretrained(self.path_ppo_structure_model)
        elif phase is TrainType.PPO_COVERAGE:
            self.model = AutoModelForCausalLM.from_pretrained(self.path_ppo_coverage_model)

        # generate mutation for given prompt
        def generate_code(
            p_prompt: str,
            p_model: PreTrainedModel,
            p_tokenizer: PreTrainedTokenizerBase,
            max_new_tokens: int = 512,
            num_return_sequences: int = 1,
        ) -> list[str]:
            inputs = p_tokenizer.encode(p_prompt, return_tensors="pt")
            outputs = p_model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                pad_token_id=p_tokenizer.pad_token_id,
                temperature=0.8,
            )
            generated_codes = [
                p_tokenizer.decode(output, skip_special_tokens=True) for output in outputs
            ]
            return generated_codes

        # load backup dataset
        data_handler = self.get_data_handler_by_target(target)
        if data_handler is None:
            print("Error: No data handler available for evaluation")
            return 0, 0.0, 0.0, ""

        backup = data_handler.backup_data
        if not backup:
            if data_handler.path_backup_data:
                load_success = data_handler.load_dataset_from_file(
                    data_handler.path_backup_data, d_type=DataSetType.BACKUP
                )
                if load_success:
                    backup = data_handler.backup_data

        if not backup:
            logger.error("Could not load backup dataset")
            print("Error: Could not load backup dataset")
            return 0, 0.0, 0.0, ""

        # we use a length sampler to truncate the inputs randomly
        ls = LengthSampler(1, 512)

        # truncate input string randomly
        def truncate_input(input_str: str) -> str:
            truncated = input_str[: ls()]
            return truncated

        # number of rejected and accepted mutations
        accepted = 0
        rejected = 0

        # set sample count
        num_examples = 1000

        # create directories for the eval run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_dir = f"{self.path_eval_dir}/{timestamp}_{phase.value}"
        mutations_dir = f"{eval_dir}/mutations"

        os.makedirs(mutations_dir, exist_ok=True)

        # generate mutations and validate them
        for i in tqdm(range(num_examples), desc="processing examples"):
            if not backup or len(backup) == 0:
                print("Error: Backup dataset is empty")
                break
            # Access backup data like a regular dataset
            prompt = backup[random.randint(0, len(backup) - 1)]["content"]
            truncated_prompt = truncate_input(prompt)
            mutated_codes = generate_code(
                truncated_prompt, self.model, tokenizer, max_new_tokens=100
            )

            for j, code in enumerate(mutated_codes):
                # we are using the scoring agent of later following ppo implementation it rewards 5.0 for every
                # correct mutation, 0 if a codec was used that was not supported, -0.5 or -1.0 otherwise
                scoring_handler = self.get_data_handler_by_target()
                if scoring_handler is None:
                    print("Error: No data handler available for scoring")
                    continue
                score = scoring_handler.check_lua_w_teal(code)
                if score == 10:
                    accepted += 1
                elif score == 0.0:
                    continue
                else:
                    rejected += 1

                # save mutations in files
                mutation_file_name = f"{mutations_dir}/mutation_{i}_{j}.txt"
                with open(mutation_file_name, "w") as file:
                    file.write(f"original prompt:\n{prompt}\n\n")
                    file.write(f"truncated prompt:\n{truncated_prompt}\n\n")
                    file.write(f"mutated code:\n{code}\n\n")
                    file.write(f"score: {score}\n\n")

        # calculate percentages
        total = accepted + rejected
        accepted_percent = (accepted / total) * 100
        rejected_percent = (rejected / total) * 100

        logger.info(
            f"Evaluation completed: {total} examples processed, "
            f"{accepted_percent:.2f}% accepted, {rejected_percent:.2f}% rejected."
        )

        return total, accepted_percent, rejected_percent, eval_dir

    def get_train_docker_by_name(self, n: str) -> MagmaContainer | None:
        """
        Retrieves a training Docker container by its name from the collection of training containers.

        @param n: The name of the training Docker container to retrieve.
        @type n: Str

        @return: The training Docker container with the specified name, if found.
        @rtype: MagmaContainer or None
        """
        for docker in self.magma_train_docker:
            if docker.name == n:
                return docker
        return None

    def get_eval_docker_by_name(self, name: str) -> MagmaContainer | None:
        """
        Retrieves an evaluation Docker container by its name from the collection of evaluation containers.

        @param name: The name of the evaluation Docker container to retrieve.
        @type name: Str

        @return: The evaluation Docker container with the specified name, if found.
        @rtype: MagmaContainer or None
        """
        for docker in self.magma_eval_docker:
            if docker.name == name:
                return docker
        return None

    def start_model_eval(
        self, time_limit: str, t_type: TrainType, target: Target = Target.LUA
    ) -> tuple[list[str], list[str], str]:
        """
        Starts an evaluation process for the model within a Docker container over a specified time limit.
        The method initializes a Docker container, sets up monitoring, executes initial seed inputs,
        generates and evaluates mutations, and stores coverage data and other logs in an evaluation directory.

        @param time_limit: The total time for the evaluation, given as a string with time units (e.g., '24h').
        @type time_limit: Str

        @param target: The target language or environment for the evaluation, with Lua as the default.
        @type target: Target, optional

        @return: A tuple containing paths to coverage reports, monitor data, and the evaluation directory.
        @rtype: Tuple(list of str, list of str, str)
        """

        # load ppo model for evaluation
        if t_type == TrainType.PPO_STRUCTURE:
            self.model = AutoModelForCausalLM.from_pretrained(self.path_ppo_structure_model)
        elif t_type == TrainType.PPO_COVERAGE:
            self.model = AutoModelForCausalLM.from_pretrained(self.path_ppo_coverage_model)

        # parse given time string and get total evaluation time in seconds
        time_limit_seconds = parse_time_input(time_limit)
        neo_docker_name = f"neo_{target.value}_eval_docker"
        neo_docker = MagmaContainer(neo_docker_name)
        self.magma_eval_docker.append(neo_docker)

        # since default location is mandatory, for monitor support move the files
        neo_docker.run_command("./scripts/neo/setup_docker_monitoring.sh")
        shared_directory = self.path_shared_dir
        os.makedirs(f"{shared_directory}/inputs", exist_ok=True)

        # setup path to seed corpora
        seed_directory = self.path_corpus_dir

        time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_dir = f"{self.path_eval_dir}/{time_stamp}_neo_lua_evaluation"
        os.makedirs(eval_dir, exist_ok=True)
        os.makedirs(f"{shared_directory}/all_inputs/", exist_ok=True)

        # read initial corpus files
        seed_files = os.listdir(seed_directory)
        start_time = time.time()

        with open(f"{eval_dir}/start_time.txt", "w") as file:
            file.write(str(start_time))

        # setup magma monitor
        neo_docker.run_command_in_thread("/magma/magma/init_magma_monitor.sh")

        iteration = 0
        seeds = True
        while time.time() < start_time + time_limit_seconds:
            if seeds:
                for seed_file in seed_files:
                    path = os.path.join(seed_directory, seed_file)
                    time_stamp = str(time.time())
                    with open(path, "rb") as input_file:
                        target_path = f"{shared_directory}/all_inputs/{seed_file}_{time_stamp}"
                        with open(target_path, "wb") as output_file:
                            output_file.write(input_file.read())
                    shutil.copyfile(
                        os.path.join(seed_directory, seed_file),
                        f"{shared_directory}/inputs/{time_stamp}.lua",
                    )
                    neo_docker.run_command("python3 /scripts/execute_gcov.py", time_out=25)
                    os.remove(f"{shared_directory}/inputs/{time_stamp}.lua")
                seeds = False
            else:
                # select random input and mutate it
                selected_file = random.choice(seed_files)
                # open file and read content
                file_path = os.path.join(seed_directory, selected_file)
                # read byte string from file and decode it
                with open(file_path, "rb") as file:
                    content_bytes = file.read()
                    content = decode_str(content_bytes)
                # generate mutation
                if content is not None:
                    mutation = self.generate_mutation(content)  # type: ignore
                else:
                    continue
                time_stamp = str(time.time())
                # run target on mutation
                with open(f"{shared_directory}/inputs/{time_stamp}.lua", "w") as file:
                    file.write(mutation)
                neo_docker.run_command("python3 scripts/execute_gcov.py", time_out=25)

                with open(
                    f"{shared_directory}/all_inputs/test_{iteration}_{time_stamp}", "w"
                ) as file:
                    file.write(mutation)

                iteration += 1
                time.sleep(0.3)

        # parse coverage reports
        subprocess.run(
            [
                "python3",
                "scripts/parse_gcov_out.py",
                f"{shared_directory}/reports",
                "--start_time",
                str(start_time),
            ]
        )

        # move data to logging dir
        source_dir = f"{shared_directory}"
        target_dir = f"{eval_dir}"
        for item in os.listdir(source_dir):
            source_path = os.path.join(source_dir, item)
            if os.path.isdir(source_path) and item != "inputs":
                shutil.move(source_path, os.path.join(target_dir, item))

        # shutdown docker
        self.magma_eval_docker.remove(neo_docker)
        neo_docker.__del__()

        shutil.rmtree(shared_directory)

        # return
        return glob.glob(f"{eval_dir}/reports/*.json"), glob.glob(f"{eval_dir}/monitor/*"), eval_dir

    @no_type_check
    def generate_mutation(self, prompt: str) -> str:
        """
        Generates a mutated version of a given prompt using the model and tokenizer.
        The prompt is truncated to a random length, then passed to the model to produce
        a mutation by generating additional code.

        @param prompt: The input prompt as a string to mutate.
        @type prompt: str

        @return: A string containing the mutated prompt.
        @rtype: Str
        """

        @no_type_check
        def generate_code(
            p_prompt,
            p_model,
            p_tokenizer,
            max_new_tokens=100,
            num_return_sequences=1,
        ) -> Any:
            inputs = p_tokenizer.encode(p_prompt, return_tensors="pt")
            outputs = p_model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                pad_token_id=p_tokenizer.pad_token_id,
                temperature=0.8,
            )
            generated_codes = [
                p_tokenizer.decode(output, skip_special_tokens=True) for output in outputs
            ]
            return generated_codes

        ls = LengthSampler(1, 512)

        @no_type_check
        def truncate_input(input_str: str) -> str:
            truncated = input_str[: ls()]
            return truncated

        truncated_prompt = truncate_input(prompt)
        mutated_prompt = generate_code(
            truncated_prompt, self.model, self.tokenizer, max_new_tokens=512
        )[0]

        return mutated_prompt

    def start_afl_eval(self, time_limit: str = "24h") -> tuple[list[str], list[str], str]:
        """
        Starts an evaluation process for AFL++ within a Docker container for a specified time limit.
        The method sets up the container, initiates the AFL++ fuzzing campaign, and logs the results.
        After the campaign completes, it runs a coverage analysis on the generated inputs
        and stores the results in an evaluation directory.

        @param time_limit: The duration for which the AFL++ evaluation runs, given as a string with time units (e.g., '24h').
        @type time_limit: Str, optional

        @return: A tuple containing paths to coverage reports, monitor data, and the evaluation directory.
        @rtype: Tuple(list of str, list of str, str)
        """

        # create docker for target
        afl_docker = MagmaContainer("afl++", time_limit)
        self.magma_eval_docker.append(afl_docker)
        # create logging dir
        eval_dir = f'{self.path_eval_dir}/{datetime.now().strftime("%Y%m%d_%H%M%S")}_afl++_eval'
        os.makedirs(eval_dir, exist_ok=True)
        shared = self.path_shared_dir
        # start campaign
        afl_docker.start_afl_campaign()
        for item in os.listdir(shared):
            shutil.move(os.path.join(shared, item), eval_dir)
        # shutdown campaign docker
        self.magma_eval_docker.remove(afl_docker)
        afl_docker.__del__()
        # create fresh docker for coverage calculation
        cov_docker = MagmaContainer("afl++_cov")
        self.magma_eval_docker.append(cov_docker)
        # create new docker to recalculate queue coverage
        source_dir = f"{eval_dir}/findings/default/queue"
        target_dir = f"{shared}/inputs"
        os.makedirs(target_dir, exist_ok=True)
        for file_name in os.listdir(source_dir):
            source_file = os.path.join(source_dir, file_name)
            target_file = os.path.join(target_dir, file_name)

            # handle bug with db.lua test case
            if "orig:db.lua" in file_name:
                name, ext = os.path.splitext(file_name)
                modified_file_name = f"{name}{ext}-"
                target_file = os.path.join(target_dir, modified_file_name)

            if os.path.isfile(source_file):
                shutil.copy2(source_file, target_file)

        # calc coverage
        cov_docker.run_command("./scripts/neo/setup_docker_monitoring.sh")
        cov_docker.run_command("python3 /scripts/execute_gcov.py")
        subprocess.run(["python3", "scripts/parse_gcov_out.py", f"{shared}/reports", "--afl"])

        # free cov docker
        self.magma_eval_docker.remove(cov_docker)
        cov_docker.__del__()

        shutil.move(f"{shared}/reports", eval_dir)
        shutil.rmtree(shared)

        # return coverage reports and monitor reports
        return glob.glob(f"{eval_dir}/reports/*.json"), glob.glob(f"{eval_dir}/monitor/*"), eval_dir


def decode_str(b_s: bytes) -> str | None:
    """
    Attempts to decode a byte string using a list of common encodings.
    Returns the decoded string if successful, or None if decoding fails for all encodings.

    @param b_s: The byte string to decode.
    @type b_s: Bytes

    @return: The decoded string if decoding is successful, otherwise None.
    @rtype: Str or None
    """

    # list of common encodings
    encodings_to_try = ["utf-8", "latin-1", "iso-8859-1", "cp1252", "ascii"]
    decoded_content = None

    for encoding in encodings_to_try:
        try:
            decoded_content = b_s.decode(encoding)
            break
        except UnicodeDecodeError:
            pass

    return decoded_content
