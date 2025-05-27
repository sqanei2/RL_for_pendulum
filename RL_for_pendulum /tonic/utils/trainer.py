import os
import time

import numpy as np

from tonic import logger

import tensorflow as tf


class Trainer:
    '''Trainer used to train and evaluate an agent on an environment.'''

    def __init__(
        self, steps=int(1e7), epoch_steps=int(2e4), save_steps=int(5e5),
        test_episodes=5, show_progress=True, replace_checkpoint=False,
        tf_writer=None
    ):
        """Initializes the trainer.
        :param steps: The total number of training steps.
        :param epoch_steps: The number of steps per epoch. An epoch is a period of time during which the agent interacts with the environment. It is just arbitrary and does not have any impact on the training.
        :param save_steps: The number of steps between saving checkpoints.
        :param test_episodes: The number of episodes to test the agent.
        :param show_progress: Whether to show the progress bar.
        :param tf_writer: The TensorFlow summary writer.
        """

        self.max_steps = steps
        self.epoch_steps = epoch_steps
        self.save_steps = save_steps
        self.test_episodes = test_episodes
        self.show_progress = show_progress
        self.replace_checkpoint = replace_checkpoint
        self.tf_writer = tf_writer

        self.flag_restart = False

    def initialize(self, agent, environment, test_environment=None, seed=None):
        if seed is not None:
            environment.initialize(seed=seed)
        if test_environment and seed is not None:
            test_environment.initialize(seed=seed + 10000)

        agent.initialize(
            observation_space=environment.observation_space,
            action_space=environment.action_space, seed=seed)

        self.agent = agent
        self.environment = environment
        self.test_environment = test_environment

    def restart(self, steps_restart_from):
        self.flag_restart = True
        self.steps_restart_from = steps_restart_from

    def run(self):
        '''Runs the main training loop.'''

        start_time = last_epoch_time = time.time()

        # Start the environments.
        observations = self.environment.start()

        num_workers = len(observations)  # the `num_worker` here is actually `num_process * num_thread`
        scores = np.zeros(len(observations))
        lengths = np.zeros(len(observations), int)
        self.steps, epoch_steps, epochs, episodes = 0, 0, 0, 0
        steps_since_save = 0
        # # Peng: not successful in add restarting from a checkpoint, so, just comment out the following line.
        # #       (Likely wrong and not relevant anymore. Just keep for reference.)
        # if self.flag_restart:
        #     steps = self.steps_restart_from
        # if self.save_steps:
        #     save_steps = np.ceil(self.save_steps / num_workers)
        # else:
        #     save_steps = None

        with self.tf_writer.as_default():
            while True:
                # Select actions.
                actions = self.agent.step(observations, self.steps)
                assert not np.isnan(actions.sum())
                logger.store('train/action', actions, stats=True)

                # Take a step in the environments.
                observations, infos = self.environment.step(actions)
                replayed, training_counter, actor_critic_losses = self.agent.update(**infos, steps=self.steps)
                # Tensorboard log: the training data if `replayed`, meaning trained.
                if replayed:
                    tf.summary.scalar(f'0_vs_training_counter/loss_of_actor', actor_critic_losses['actor_loss'], step=training_counter)
                    tf.summary.scalar(f'0_vs_training_counter/loss_of_critic', actor_critic_losses['critic_loss'], step=training_counter)
                    tf.summary.scalar(f'2_vs_total_steps/loss_of_actor', actor_critic_losses['actor_loss'], step=self.steps)
                    tf.summary.scalar(f'2_vs_total_steps/loss_of_critic', actor_critic_losses['critic_loss'], step=self.steps)

                scores += infos['rewards']
                lengths += 1  # an array of the length of each episode, all add one. Later they will be set to 0 if env gets reset.
                self.steps += num_workers
                epoch_steps += num_workers
                steps_since_save += num_workers
                # Tensorboard log: real-time training data, the accumulated score and length of one worker.
                #                  No need to show everything. Just show the first episode as an example, for curiosity.
                #                  Didn't test but this should slow down the simulation, so, set to write every several steps.
                if lengths[0] % 20 == 0:
                    tf.summary.scalar('3_vs_step_in_episode/current_score_worker0', scores[0], step=lengths[0])

                # Show the progress bar.
                if self.show_progress:
                    logger.show_progress(self.steps, self.epoch_steps, self.max_steps)

                # Check the finished episodes.
                for i in range(num_workers):
                    if infos['resets'][i]:
                        logger.store('train/episode_score', scores[i], stats=True)
                        logger.store('train/episode_length', lengths[i], stats=True)
                        # Tensorboard log: finished episode data, `episodes` is the global counter for episodes (either in serial or parallel).
                        tf.summary.scalar(f'1_vs_episode_id/final_episode_score_worker{i}', scores[i], step=episodes)
                        tf.summary.scalar(f'1_vs_episode_id/final_episode_length_worker{i}', lengths[i], step=episodes)
                        # Tensorboard log: End
                        scores[i] = 0
                        lengths[i] = 0
                        episodes += 1

                # End of the epoch.
                if epoch_steps == self.epoch_steps:
                    # Evaluate the agent on the test environment.
                    if self.test_environment:
                        scores_test, lengths_test = self._test()
                        # Tensorboard log: testing data, the accumulated score and length of one worker.
                        for _ in range(self.test_episodes):
                            tf.summary.scalar(f'0_vs_training_counter/test{_}_score', scores_test[_], step=training_counter)
                            tf.summary.scalar(f'0_vs_training_counter/test{_}_length', lengths_test[_], step=training_counter)
                            tf.summary.scalar(f'2_vs_total_steps/test{_}_score', scores_test[_], step=self.steps)
                            tf.summary.scalar(f'2_vs_total_steps/test{_}_length', lengths_test[_], step=self.steps)

                    # Log the data.
                    epochs += 1
                    current_time = time.time()
                    epoch_time = current_time - last_epoch_time
                    sps = epoch_steps / epoch_time
                    logger.store('train/episodes', episodes)
                    logger.store('train/epochs', epochs)
                    logger.store('train/seconds', current_time - start_time)
                    logger.store('train/epoch_seconds', epoch_time)
                    logger.store('train/epoch_steps', epoch_steps)
                    logger.store('train/steps', self.steps)
                    logger.store('train/worker_steps', self.steps // num_workers)
                    logger.store('train/steps_per_second', sps)
                    logger.dump()
                    last_epoch_time = time.time()
                    epoch_steps = 0

                # End of training.
                stop_training = self.steps >= self.max_steps

                # Tensorboard log: Save a tensorflow checkpoint.
                if stop_training or steps_since_save >= self.save_steps:
                    path = os.path.join(logger.get_path(), 'checkpoints')
                    if os.path.isdir(path) and self.replace_checkpoint:
                        for file in os.listdir(path):
                            if file.startswith('step_'):
                                os.remove(os.path.join(path, file))
                    checkpoint_name = f'step_{self.steps}'
                    save_path = os.path.join(path, checkpoint_name)
                    self.agent.save(save_path)
                    steps_since_save = self.steps % self.save_steps

                if stop_training:
                    break

                self.tf_writer.flush()

    def _test(self):
        '''Tests the agent on the test environment.'''

        # Test loop.
        scores_test = []
        lengths_test = []
        for _ in range(self.test_episodes):
            score, length = 0, 0

            # Start the environment.
            self.test_observations = self.test_environment.start()
            assert len(self.test_observations) == 1

            while True:
                # Select an action.
                actions = self.agent.test_step(
                    self.test_observations, self.steps)
                assert not np.isnan(actions.sum())
                logger.store('test/action', actions, stats=True)

                # Take a step in the environment.
                self.test_observations, infos = self.test_environment.step(
                    actions)
                self.agent.test_update(**infos, steps=self.steps)

                score += infos['rewards'][0]
                length += 1

                if infos['resets'][0]:
                    break

            # Log the data.
            logger.store('test/episode_score', score, stats=True)
            logger.store('test/episode_length', length, stats=True)

            # Append score and length to output lists
            scores_test.append(score)
            lengths_test.append(length)

        return scores_test, lengths_test
