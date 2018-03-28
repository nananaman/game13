import chainer
from chainer import Variable, optimizers, Chain
import chainer.functions as F
import chainer.links as L
import numpy as np
from collections import deque
from game13 import RandomPlayer, game
import copy


def huberloss(x, t, delta=1.):
    err = t - x
    cond = F.absolute(err).data < 1.0
    L2 = 0.5 * F.square(err)
    L1 = delta * (F.absolute(err) - 0.5)
    loss = F.where(cond, L2, L1)
    return F.mean(loss)


class MLP(Chain):
    def __init__(self, n_in, n_units, n_out):
        super().__init__(
            l1=L.Linear(n_in, n_units),
            l2=L.Linear(n_units, n_units),
            l3=L.Linear(n_units, n_out),
        )

    def __call__(self, x, t=None, train=False):
        #print(x)
        h = F.leaky_relu(self.l1(x))
        #print(h)
        h = F.leaky_relu(self.l2(h))
        #print(h)
        h = self.l3(h)
        #print(h)

        if train:
            # Normal Q func
            # return F.mean_squared_error(h, t)
            # Huber func
            return huberloss(h, t)
        else:
            return h


class Memory:
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, exp):
        self.buffer.append(exp)

    def sample(self, bs):
        idx = np.random.choice(
            np.arange(len(self.buffer)), size=bs, replace=False)
        return [self.buffer[ii] for ii in idx]

    def len(self):
        return len(self.buffer)


class DQN_player:
    def __init__(self, n=3, e=1, memory_size=300):
        self.n = n
        self.model = MLP(1, 128, n)
        self.model_ast = copy.deepcopy(self.model)
        self.optimizer = optimizers.RMSprop(lr=0.00015, alpha=0.95)
        self.optimizer.setup(self.model)
        self.e = e
        self.gamma = 0.95
        self.name = 'DQN'
        self.actMemory = Memory(memory_size)
        self.last_score = None
        self.last_act = None

    def act(self, score):
        x = np.array([[score]]).astype(np.float32)
        pred = self.model(x)
        act = np.argmax(pred.data, axis=1)

        if self.e > 0.2:
            self.e -= 1. / 20000
        if np.random.rand() < self.e:
            act = np.random.randint(0, 3)
        self.last_score = score
        self.last_act = act
        return act + 1

    def add_act_memory(self, reward, score):
        state = self.last_score
        act = self.last_act

        next_state = score
        self.actMemory.add((state, act, reward, next_state))

    def actLearn(self, bs):
        inputs = np.zeros((bs, 1))
        targets = np.zeros((bs, self.n))
        mini_batch = self.actMemory.sample(bs)
        #print(mini_batch)

        for i, (state, action, reward, next_state) in enumerate(mini_batch):
            inputs[i:i + 1] = state
            Q_value = self.model_ast(np.array([[next_state]]).astype(np.float32))
            target = reward + self.gamma * np.max(Q_value.data[0])
            targets[i][action] = target

        self.model.zerograds()
        loss = self.model(inputs.astype(np.float32),
                          targets.astype(np.float32), train=True)

        loss.backward()
        self.optimizer.update()

    def q_update(self):
        self.model_ast = copy.deepcopy(self.model)


class GameOrganizer:
    def __init__(self, game, p1, p2):
        self.game = game
        self.p1 = p1
        self.p2 = p2
        self.nplayed = 0
        self.nplay = 50000
        self.winCounter = [0, 0]

        self.learn_interval = 10
        self.batch_size = 64
        self.update_Q = 10

    def progress(self):
        while self.nplayed < self.nplay:
            reward = [0, 0]
            self.game.reset()
            result = 0
            turn = True
            for i in range(14):
                if turn:
                    action = self.p1.act(self.game.score)
                else:
                    action = self.p2.act(self.game.score)

                result = self.game.step(action)

                if result == 1:
                    if turn:
                        self.winCounter[1] += 1
                        reward = [-1, 1]
                    else:
                        self.winCounter[0] += 1
                        reward = [1, -1]
                if not turn :
                    self.p2.add_act_memory(reward[1], self.game.score)

                turn = not(turn)

                if self.p2.actMemory.len() > self.batch_size and self.nplayed % self.learn_interval == 0:
                    self.p2.actLearn(self.batch_size)

                if result == 1:
                    break
            self.nplayed += 1
            if self.nplayed % self.update_Q == 0:
                self.p2.q_update()

            if self.nplayed % 5000 == 0:
                print(self.winCounter)
                self.winCounter = [0, 0]


def main():
    n = 3
    p1 = RandomPlayer(n)
    p2 = DQN_player(n)
    Game = game(n)
    org = GameOrganizer(Game, p1, p2)
    org.progress()


if __name__ == '__main__':
    main()
