import argparse
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    BaseCallback,
)
from stable_baselines3.common.vec_env import VecNormalize
from brick_lift_env import BrickLiftEnv

#  PATHS
LOG_DIR   = "logs/"
CKPT_DIR  = "checkpoints/"
BEST_DIR  = "best_model/"
os.makedirs(LOG_DIR,  exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(BEST_DIR, exist_ok=True)

#  TRAINING PARAMETERS
N_ENVS        = 4        # parallel environments — increase if you have more CPU
TOTAL_STEPS   = 5_000_000
CHECKPOINT_FREQ = 100_000  # save a checkpoint every N steps


class SuccessRateCallback(BaseCallback):
    # Prints success rate to terminal every N rollouts
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.successes       = 0
        self.episodes        = 0

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episodes += 1
                ep_rew = info["episode"]["r"]
                self.episode_rewards.append(ep_rew)
                # A reward >= 100 means the success bonus was hit
                if ep_rew >= 100:
                    self.successes += 1

                if self.episodes % 50 == 0:
                    rate = self.successes / max(1, self.episodes)
                    mean = np.mean(self.episode_rewards[-50:])
                    print(f"  [ep {self.episodes:5d}] "
                          f"mean_reward={mean:7.1f}  "
                          f"success_rate={rate:.2%}")
        return True


def make_env():
    return BrickLiftEnv(xml_path="mjxpandamerged.xml")


def train(resume=False):
    print("=" * 60)
    print("  Brick Lift RL Training — PPO + SB3")
    print("=" * 60)
    print(f"  Parallel envs : {N_ENVS}")
    print(f"  Total steps   : {TOTAL_STEPS:,}")
    print(f"  Checkpoints   : {CKPT_DIR}")
    print("=" * 60)

    # Vectorised + normalised environments
    vec_env = make_vec_env(make_env, n_envs=N_ENVS)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True,
                           clip_obs=10.0)

    if resume and os.path.exists(f"{CKPT_DIR}/latest_model.zip"):
        print("  Resuming from checkpoint...")
        model = PPO.load(f"{CKPT_DIR}/latest_model", env=vec_env)
    else:
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            # Network: two hidden layers — wide enough for the tactile obs
            policy_kwargs=dict(net_arch=[512, 512]),
            learning_rate=3e-4,
            n_steps=2048,        # rollout length per env before update
            batch_size=256,
            n_epochs=10,
            gamma=0.99,          # discount — high because hold reward is delayed
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,      # small entropy bonus — encourages exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=LOG_DIR,
            verbose=1,
        )

    # Eval env must be wrapped in VecNormalize to match training env, but with training=False so it doesn't update running stats
    eval_vec_env = make_vec_env(make_env, n_envs=1)
    eval_vec_env = VecNormalize(eval_vec_env, norm_obs=True, norm_reward=False,
                                clip_obs=10.0, training=False)

    callbacks = [
        # Save a checkpoint every CHECKPOINT_FREQ steps
        CheckpointCallback(
            save_freq=CHECKPOINT_FREQ // N_ENVS,
            save_path=CKPT_DIR,
            name_prefix="ppo_brickLift",
        ),
        # Evaluate on a separate env and save the best model
        EvalCallback(
            eval_vec_env,
            best_model_save_path=BEST_DIR,
            log_path=LOG_DIR,
            eval_freq=CHECKPOINT_FREQ // N_ENVS,
            n_eval_episodes=3,
            deterministic=True,
            verbose=1,
        ),
        SuccessRateCallback(),
    ]

    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=callbacks,
        reset_num_timesteps=not resume,
    )

    model.save(f"{CKPT_DIR}/latest_model")
    vec_env.save(f"{CKPT_DIR}/vec_normalize.pkl")
    print("\n  Training complete. Model saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")
    args = parser.parse_args()
    train(resume=args.resume)