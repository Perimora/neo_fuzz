import os
from typing import Optional

from matplotlib import pyplot as plt

from src.classes.controller.training_env import TrainingEnv
from src.classes.enum.train_types import TrainType


class EvalView:
    def __init__(self, env: TrainingEnv):
        """
        Initializes the class with the specified training environment.

        @param env: The training environment used to manage evaluation processes and configurations.
        @type env: TrainingEnv
        """
        if env is None:
            raise ValueError("Training environment cannot be None")
        self.env = env

    def evaluate_generations(self, phase: TrainType) -> Optional[str]:
        """
        Evaluates the generated code by calculating the percentage of accepted and rejected Lua code mutations.
        Generates a bar plot visualizing the results.

        @param phase: The training phase used for evaluation, such as syntax or semantic evaluation.
        @type phase: TrainType
        @return: Path to the evaluation directory if successful, None otherwise
        @rtype: Optional[str]
        """

        if phase is None:
            raise ValueError("Training phase cannot be None")

        try:
            result = self.env.run_syn_sem_eval(phase)
            if result is None or len(result) != 4:
                print(f"Error: Invalid result from run_syn_sem_eval: {result}")
                return None

            total, accepted, rejected, eval_dir = result

            if eval_dir is None:
                print("Error: Evaluation directory is None")
                return None

            # Validate the results are numeric
            if not all(isinstance(x, (int, float)) for x in [total, accepted, rejected]):
                print(
                    f"Error: Invalid numeric results - total: {total}, accepted: {accepted}, rejected: {rejected}"
                )
                return None

            # create directory for plot
            os.makedirs(f"{eval_dir}/diagrams", exist_ok=True)

            # Save results to file
            try:
                with open(f"{eval_dir}/results.txt", "w") as f:
                    f.write(f"total: {total}\n")
                    f.write(f"accepted: {accepted}\n")
                    f.write(f"rejected: {rejected}\n")
            except IOError as e:
                print(f"Error writing results file: {e}")
                return None

            # plot the results
            labels = ["accepted", "rejected"]
            percentages = [accepted, rejected]

            plt.figure(figsize=(8, 6))
            plt.bar(labels, percentages, color=["blue", "grey"])
            plt.xlabel("code evaluation")
            plt.ylabel("percentage")
            plt.title(f"accepted and rejected lua code mutations after {phase.value}")
            plt.ylim(0, 100)

            # create timestamp and file name
            diagram_file_name = f"{eval_dir}/diagrams/{phase.value}_eval.png"

            try:
                plt.savefig(diagram_file_name, dpi=300, bbox_inches="tight")
                plt.close()  # Close the figure to free memory
                print(f"Diagram saved to: {diagram_file_name}")
            except Exception as e:
                print(f"Error saving diagram: {e}")
                plt.close()  # Still close the figure even if saving failed
                return None

            print(f"Logged outputs to: {eval_dir}")
            return eval_dir

        except Exception as e:
            print(f"Error during evaluation: {e}")
            # Ensure any open plots are closed
            plt.close("all")
            return None
