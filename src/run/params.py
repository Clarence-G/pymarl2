import copy
import pickle


class AgentParam:
    def __init__(self):
        self.agent = None
        self.optimiser_dict = None
        # self.target_mac_dict = None
        # self.mixer_dict = None
        # self.target_mac_dict = None
        # self.learner = None

    def save_params(self, learner):
        self.agent = copy.deepcopy(learner.mac.agent)
        self.optimiser_dict = learner.optimiser.state_dict()
        # self.target_mac_dict = learner.target_mac.agent.state_dict()
        # if learner.mixer:
        #     self.mixer_dict = learner.mixer.state_dict()
        #     self.target_mixer_dict = learner.target_mixer.state_dict()

    def load_params(self, learner):
        learner.mac.agent.load_state_dict(self.agent.state_dict())
        learner.optimiser.load_state_dict(self.optimiser_dict)

    # save params to file path
    def save_params_to_file(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def clone(self):
        return copy.deepcopy(self)

    # load params from file path
    @classmethod
    def load_params_from_file(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)


