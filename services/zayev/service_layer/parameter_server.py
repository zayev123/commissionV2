# from services.zayev.service_layer.zayev import Zayev
import ray


# @ray.remote
class ParameterServer:
    def __init__(self, zayev):
        self.weights = None
        self.eval_weights = None
        self.Q = zayev.get_Q_network()

    def update_weights(self, new_parameters):
        self.weights = new_parameters
        return True
    
    def get_weights(self):
        return self.weights
    
    def get_eval_weights(self):
        return self.eval_weights
    
    def set_eval_weights(self):
        self.eval_weights = self.weights
        # print("eval_weights_1", self.eval_weights)
        return True
    
    def save_eval_weights(self,
        filename=
        'checkpoints/model_checkpoint'):
        # print("eval_weights_2", self.eval_weights)
        self.Q.set_weights(self.eval_weights)
        self.Q.save_weights(filename)
        print("Saved.")