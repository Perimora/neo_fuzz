import logging

from src.classes.controller.training_env import TrainingEnv
from src.classes.enum.targets import Target
from src.classes.enum.train_types import TrainType

logger = logging.getLogger("app.dev")


class TrainingView:

    def __init__(self, env: TrainingEnv, target: Target = Target.LUA) -> None:
        """
        Initializes the class with a specified training environment and target.

        @param env: The training environment used to manage data and training processes.
        @type env: TrainingEnv

        @param target: The target platform or language for training, defaulting to Lua.
        @type target: Target
        """
        self.env = env
        self.target = target

    def init_data(self) -> bool:
        """
        Initializes and prepares data within the training environment.
        """
        result = self.env.init_data()
        if result:
            logger.info("TrainingView: Data initialization successful")
        else:
            logger.error("TrainingView: Data initialization failed")
        return result

    def start_training(self, t_type: TrainType) -> None:
        """
        Starts the training process based on the specified training type.

        @param t_type: The type of training to execute, such as SSL or PPO training.
        @type t_type: TrainType
        """

        # start ssl training
        logger.info(f"Starting {t_type.value} training...")

        if t_type == TrainType.SSL:
            self.env.run_ssl_training()
        elif t_type == TrainType.PPO_STRUCTURE or t_type == TrainType.PPO_COVERAGE:
            self.env.run_ppo_training(t_type)
        else:
            print("Error: Unknown training type provided...")
