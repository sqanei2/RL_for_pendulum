import tensorflow as tf

from tonic import explorations, logger, replays
from tonic.tensorflow import agents, models, normalizers, updaters


def default_model():
    return models.ActorCriticWithTargets(
        actor=models.Actor(
            encoder=models.ObservationEncoder(),
            torso=models.MLP((256, 256), 'relu'),
            head=models.DeterministicPolicyHead()),
        critic=models.Critic(
            encoder=models.ObservationActionEncoder(),
            torso=models.MLP((256, 256), 'relu'),
            head=models.ValueHead()),
        observation_normalizer=normalizers.MeanStd())


class DDPG(agents.Agent):
    '''Deep Deterministic Policy Gradient.
    DDPG: https://arxiv.org/pdf/1509.02971.pdf
    '''

    def __init__(
        self, model=None, replay=None, exploration=None, actor_updater=None,
        critic_updater=None
    ):
        self.model = model or default_model()
        self.replay = replay or replays.Buffer()
        self.exploration = exploration or explorations.NormalActionNoise()
        self.actor_updater = actor_updater or \
            updaters.DeterministicPolicyGradient()
        self.critic_updater = critic_updater or \
            updaters.DeterministicQLearning()

    def initialize(self, observation_space, action_space, seed=None):
        super().initialize(seed=seed)
        self.model.initialize(observation_space, action_space)
        self.replay.initialize(seed)
        self.exploration.initialize(self._policy, action_space, seed)
        self.actor_updater.initialize(self.model)
        self.critic_updater.initialize(self.model)
        self.steps = 0
        self.training_counter = 0
        self.actor_critic_losses = None

    def step(self, observations, steps):
        # Get actions from the actor and exploration method.
        actions = self.exploration(observations, steps)

        # Keep some values for the next update.
        self.last_observations = observations.copy()
        self.last_actions = actions.copy()

        return actions

    def test_step(self, observations, steps):
        # Greedy actions for testing.
        return self._greedy_actions(observations).numpy()

    def update(self, observations, rewards, resets, terminations, steps):
        # Store the last transitions in the replay.
        self.replay.store(
            observations=self.last_observations, actions=self.last_actions,
            next_observations=observations, rewards=rewards, resets=resets,
            terminations=terminations)

        # Prepare to update the normalizers.
        if self.model.observation_normalizer:
            self.model.observation_normalizer.record(self.last_observations)
        if self.model.return_normalizer:
            self.model.return_normalizer.record(rewards)

        # Update the model if the replay is ready.
        replayed = False
        if self.replay.ready(steps):
            self.training_counter +=1
            replayed = True
            self.actor_critic_losses = self._update(steps)

        self.exploration.update(resets)

        return replayed, self.training_counter, self.actor_critic_losses

    @tf.function
    def _greedy_actions(self, observations):
        return self.model.actor(observations)

    def _policy(self, observations):
        return self._greedy_actions(observations).numpy()

    def _update(self, steps):
        keys = ('observations', 'actions', 'next_observations', 'rewards',
                'discounts')

        # Update both the actor and the critic multiple times.
        for batch in self.replay.get(*keys, steps=steps):
            infos = self._update_actor_critic(**batch)

            for key in infos:
                for k, v in infos[key].items():
                    logger.store(key + '/' + k, v.numpy())

        # Update the normalizers.
        if self.model.observation_normalizer:
            self.model.observation_normalizer.update()
        if self.model.return_normalizer:
            self.model.return_normalizer.update()

        return dict(actor_loss=infos['actor']['loss'], critic_loss=infos['critic']['loss'])

    @tf.function
    def _update_actor_critic(
        self, observations, actions, next_observations, rewards, discounts
    ):
        critic_infos = self.critic_updater(
            observations, actions, next_observations, rewards, discounts)
        actor_infos = self.actor_updater(observations)
        self.model.update_targets()
        return dict(critic=critic_infos, actor=actor_infos)
