from multiprocessing import Process, Pipe

from services.zayev.environment.market_simulator import MarketSimulator

class EnvProcess(Process):
    def __init__(self, env_idx, child_conn, env_config):
        super(EnvProcess, self).__init__()
        self.env = MarketSimulator(env_config=env_config)
        self.env_idx = env_idx
        self.child_conn = child_conn

    def run(self):
        super(EnvProcess, self).run()
        state = self.env.reset()
        state = self.space_shaper(state)
        self.child_conn.send(state)
        while True:
            action = self.child_conn.recv()

            state, reward, done, info = self.env.step(action)

            if done:
                state = self.env.reset()

            self.child_conn.send([state, reward, done, info])