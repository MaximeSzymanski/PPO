from torch.utils.tensorboard import SummaryWriter

import os
class StaticWriter:

    writer = None

    @staticmethod
    def initalize_writer(env_name):
        # get the latest run number
        run_number = 0
        # check if the directory exists
        if not os.path.exists(f"tensorboard/{env_name}"):
            os.makedirs(f"tensorboard/{env_name}")
        for file in os.listdir(f"tensorboard/{env_name}"):
            if file.startswith("run"):
                run_number = max(run_number, int(file[3:]))
        run_number += 1
        StaticWriter.writer = SummaryWriter(f"tensorboard/{env_name}/run{run_number}")
