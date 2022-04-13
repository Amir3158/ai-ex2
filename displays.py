import time


class SummaryDisplay(object):
    def __init__(self):
        super(SummaryDisplay, self).__init__()
        self.scores = []
        self.highest_tile = []
        self.game_durations = []
        self.game_start_time = None

    def initialize(self, initial_state):
        self.game_start_time = time.time()

    def update_state(self, new_state, action, opponent_action):
        if new_state.done:
            game_end_time = time.time()
            game_duration = round(game_end_time - self.game_start_time, 3)
            print("score: %s\nhighest tile: %s\ngame_duration: %s" % (new_state.score, new_state.board.max(),
                                                                      game_duration))
            self.scores.append(new_state.score)
            self.highest_tile.append(new_state.board.max())
            self.game_durations.append(game_duration)

    def mainloop_iteration(self):
        pass

    def print_stats(self):
        win_rate = len(list(filter(lambda x: x >= 2048, self.highest_tile))) / len(self.highest_tile)
        counts = dict()
        for i in self.highest_tile:
            counts[i] = counts.get(i, 0) + 1
        print("="*30)
        print("scores: %s" % self.scores)
        print("highest tile: %s" % self.highest_tile)
        print("game_durations: %s" % self.game_durations)
        print("avg score: %s" % round((sum(self.scores) / len(self.scores))))
        print("cnt highest tile: %s" % counts)
        print("avg game duration: %s" % round((sum(self.game_durations) / len(self.game_durations)), 3))
        print("win rate: %s" % win_rate)
        # print("=" * 30)

    def __repr__(self):
        win_rate = len(list(filter(lambda x: x >= 2048, self.highest_tile))) / len(self.highest_tile)
        counts = dict()
        for i in self.highest_tile:
            counts[i] = counts.get(i, 0) + 1

        return f"scores: {self.scores}" \
               f"highest tile: {self.highest_tile}" \
               f"game_durations: {self.game_durations}" \
               f"avg score: {round((sum(self.scores) / len(self.scores)))}" \
               f"cnt highest tile: {counts}" \
               f"avg game duration: {round((sum(self.game_durations) / len(self.game_durations)), 3)}" \
               f"win rate: {win_rate}" \
               "=" * 30

    def __str__(self):
        return self.__repr__()
