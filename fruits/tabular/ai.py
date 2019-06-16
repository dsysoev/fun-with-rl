
import numpy as np


class SarsaAgent:
    def __init__(self, alpha, epsilon, gamma, legal_actions, strategy='mean'):
        """
        Sarsa Agent
        """
        self._qvalues = dict()
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.legal_actions = legal_actions
        self.strategy = strategy

    def get_qvalue(self, state, action):
        """ Returns Q(state,action) """
        sa = tuple(list(state) + [action])
        return self._qvalues.get(sa, 0)

    def set_qvalue(self, state, action, value):
        """ Sets the Qvalue for [state,action] to the given value """
        sa = tuple(list(state) + [action])
        self._qvalues[sa] = value

    def get_values(self, state):
        """ """
        state_values = [self.get_qvalue(state, a) for a in self.legal_actions]
        return state_values

    def get_action(self, state):
        """
        Compute the action to take in the current state, including exploration.
        With probability self.epsilon, we should take a random action.
            otherwise - the best policy action
        """

        is_random_action = np.random.random() <= self.epsilon

        if is_random_action:
            chosen_action = np.random.choice(self.legal_actions, size=1)[0]
        else:
            chosen_action = np.argmax(self.get_values(state))

        return chosen_action

    def update(self, state, action, reward, next_state):
        """
        Q(s,a) := (1 - alpha) * Q(s,a) + alpha * (r + gamma * V(s'))
        """
        if self.strategy == 'mean':
            q_next = np.mean(self.get_values(next_state))
        elif self.strategy == 'max':
            q_next = np.max(self.get_values(next_state))
        else:
            raise Exception('Error strategy name {}'.format(self.strategy))

        current_q = self.get_qvalue(state, action)

        delta = reward + self.gamma * q_next - current_q

        value = current_q + self.alpha * delta

        self.set_qvalue(state, action, value)
