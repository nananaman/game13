import random

class game:
    def __init__(self, n=3):
        self.score = 0
        self.deadline = 3 * (n + 1) + 1
    
    def reset(self):
        self.score = 0

    def step(self, action):
        self.score += action
        if self.score >= self.deadline:
            result = 1
        else:
            result = 0
        return result

class player0:
    def __init__(self, n=3):
        self.nmax = n
    
    def act(self, score):
        action = self.decision(score)
        return action

class RandomPlayer(player0):
    def __init__(self, n=3):
        super().__init__(n)

    def decision(self, score):
        return random.randint(1, self.nmax)

class SuperPlayer(player0):
    def __init__(self, n=3):
        super().__init__(n)

    def decision(self, score):
        s = score % (self.nmax + 1)
        if s != 0:
            return self.nmax + 1 - s
        else:
            return random.randint(1, self.nmax)

class GameOrganizer:
    def __init__(self, game, p1, p2):
        self.game = game
        self.p1 = p1
        self.p2 = p2
        self.nplayed = 0
        self.nplay = 10000
        self.winCounter = [0, 0]

    def progress(self):
        while self.nplayed < self.nplay:
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
                    else:
                        self.winCounter[0] += 1
                    break
                turn = not(turn)
            self.nplayed += 1
        print(self.winCounter)

def main():
    n=3
    p1 = RandomPlayer(n)
    p2 = RandomPlayer(n)
    Game = game(n)
    org = GameOrganizer(Game, p1, p2)
    org.progress()

if __name__ == '__main__':
    main()

