from src import training
import gymnasium
if __name__ == '__main__':
    gymnasium.register(
        id='StockEnv-v0',
        entry_point='gym_env:StockEnv',
    )
    training.start_training()