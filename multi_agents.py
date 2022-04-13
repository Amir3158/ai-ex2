import numpy as np
import abc
import util
from game import Agent, Action
from typing import Tuple, Union

from game_state import GameState


def snake_distance(origin_point: int, target_snake_idx: int, rows: int, cols: int) -> int:
    snake_indexes = np.arange(rows * cols).reshape(rows, cols)
    snake_indexes[-2::-2] = snake_indexes[-2::-2][:, ::-1]
    snake_distance = abs(snake_indexes.ravel()[origin_point] - snake_indexes.ravel()[target_snake_idx])
    return snake_distance


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        get_action takes a game_state and returns some Action.X for some X in the set {UP, DOWN, LEFT, RIGHT, STOP}
        """

        # Collect legal moves and successor states
        legal_moves = game_state.get_agent_legal_actions()

        # Choose one of the best actions
        # scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        action_to_score = {action: self.evaluation_function(game_state, action) for action in legal_moves}
        # best_score = max(scores)
        best_action = max(action_to_score, key=action_to_score.get)
        # best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        # chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        # print(action_to_score)
        return best_action
        # return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (GameState.py) and returns a number, where higher numbers are better.

        """
        # thresholds
        THRESH_FUSED_TILES = 0.1
        THRESH_UP_ACTION = 0.5
        THRESH_MAX_TILE = 1
        THRESH_MAX_TILE_DEGRADE = 0.5
        CALC_FUSED_TILES = lambda score, num_fused_tiles, max_tile: score * (1 + (THRESH_FUSED_TILES * num_fused_tiles)) - (max_tile * THRESH_MAX_TILE_DEGRADE)
        CALC_SCORE_SNAKE = lambda tile_val, snake_dist, target_snake_idx: (tile_val / (snake_dist + 1)) * target_snake_idx

        successor_game_state = current_game_state.generate_successor(action=action)
        board: np.ndarray = successor_game_state.board
        max_tile = successor_game_state.max_tile
        score = successor_game_state.score

        num_fused_tiles = np.count_nonzero(current_game_state.board) - np.count_nonzero(successor_game_state.board)

        all_tiles_idxes = np.argsort(board.ravel())
        snake_end = board.size - 1
        sum_score = 0

        for i, tile_idx in enumerate(reversed(all_tiles_idxes)):
            tile_val = board[np.unravel_index(tile_idx, board.shape)]
            if not tile_val:
                break
            target_snake_idx = snake_end - i
            snake_dist = snake_distance(tile_idx, target_snake_idx, *board.shape)
            score_for_tile = CALC_SCORE_SNAKE(tile_val, snake_dist, target_snake_idx)
            sum_score += score_for_tile

        if action is action.UP:
            sum_score *= THRESH_UP_ACTION

        sum_score += max_tile * THRESH_MAX_TILE

        if num_fused_tiles > 0:
            sum_score = CALC_FUSED_TILES(sum_score, num_fused_tiles, max_tile)

        return sum_score


def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.score


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinmaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=2):
        self.evaluation_function = util.lookup(evaluation_function, globals())
        self.depth = depth

    @abc.abstractmethod
    def get_action(self, game_state):
        return


class MinmaxAgent(MultiAgentSearchAgent):
    def get_action(self, game_state: GameState) -> Action:
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means our agent, the opponent is agent_index=1

        Action.STOP:
            The stop direction, which is always legal

        game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action
        """
        legal_moves = game_state.get_agent_legal_actions()
        action_to_score = {action: self.minimax(game_state.generate_successor(agent_index=0, action=action), self.depth, player=0)
                           for action in legal_moves}
        best_action = max(action_to_score, key=action_to_score.get)
        return best_action

    def minimax(self, state: GameState, depth: int, player: int) -> float:
        if depth == 0 or state.done:
            return self.evaluation_function(state)

        if player == 0:  # max player
            val = -np.inf
            for action in state.get_legal_actions(agent_index=player):
                val = max(val, self.minimax(state.generate_successor(agent_index=player, action=action), depth=depth - 1, player=1 - player))
            return val

        else:  # min player
            val = np.inf
            for action in state.get_legal_actions(agent_index=player):
                val = min(val, self.minimax(state.generate_successor(agent_index=player, action=action), depth=depth, player=1 - player))
            return val


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        legal_moves = game_state.get_agent_legal_actions()
        action_to_score = {action: self.alpha_beta(game_state.generate_successor(agent_index=0, action=action), self.depth, -np.inf, np.inf, player=0)
                           for action in legal_moves}
        best_action = max(action_to_score, key=action_to_score.get)
        return best_action

    def alpha_beta(self, state: GameState, depth: int, alpha: float, beta: float, player: int) -> float:
        if state.done or depth == 0:
            return self.evaluation_function(state)

        if player == 0:
            val = -np.inf

            for action in state.get_legal_actions(agent_index=player):
                val = max(val, self.alpha_beta(state.generate_successor(agent_index=player, action=action),
                                               depth - 1, alpha, beta, player=1 - player))
                if val >= beta:
                    break  # beta cutoff
                alpha = max(alpha, val)

            return val

        else:  # player == 1
            val = np.inf

            for action in state.get_legal_actions(agent_index=player):
                val = min(val, self.alpha_beta(state.generate_successor(agent_index=player, action=action),
                                               depth, alpha, beta, player=1 - player))
                if val <= alpha:
                    break  # alpha cutoff
                beta = min(beta, val)

            return val


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """
        legal_moves = game_state.get_agent_legal_actions()
        action_to_score = {action: self.expectimax(game_state.generate_successor(agent_index=0, action=action), self.depth, player=0)
                           for action in legal_moves}
        best_action = max(action_to_score, key=action_to_score.get)
        return best_action

    def expectimax(self, state: GameState, depth: int, player: int) -> float:
        if depth == 0 or state.done:
            return self.evaluation_function(state)

        if player == 0:  # max player
            val = -np.inf
            for action in state.get_legal_actions(agent_index=player):
                val = max(val, self.expectimax(state.generate_successor(agent_index=player, action=action), depth=depth - 1, player=1 - player))
            return val

        else:  # chance player
            vals = []
            for action in state.get_legal_actions(agent_index=player):
                vals.append(self.expectimax(state.generate_successor(agent_index=player, action=action), depth=depth, player=1 - player))

            return sum(vals) / len(vals)


def monotonicity(board: np.ndarray) -> Tuple[float, float, float]:
    best = -1

    for i in range(4):
        best = max(best, np.count_nonzero(np.diff(board) >= 0) + \
                   np.count_nonzero(np.diff(board, axis=0) >= 0))

        board = np.rot90(board)

    # return normalize(best, 0, 24)
    return best


def empty_board(board: np.ndarray) -> Tuple[float, float, float]:
    # return normalize(np.count_nonzero(board == 0), 0, board.size)
    return np.count_nonzero(board == 0)


def corner_heuristic(board: np.ndarray) -> Tuple[float, float, float]:
    weights = np.array([[0, 1, 2, 3],
                        [1, 2, 3, 4],
                        [2, 3, 4, 5],
                        [3, 4, 5, 6]])

    sorted_weights = np.sort(weights.ravel())
    sorted_board = np.sort(board.ravel())

    # return normalize((board * weights).sum(),
    #                  (sorted_weights * sorted_board[::-1]).sum(),
    #                  (sorted_weights * sorted_board).sum())

    return (board * weights).sum()


def in_range(board, row, col):
    return not ((row < 0 or row >= board.shape[0]) or (col < 0 or col >= board.shape[1]))


def clustering(board: np.ndarray) -> Tuple[float, float, float]:
    OFFSETS = [-1, 0, 1]
    clustering_score = 0

    for row in range(board.shape[0]):
        for col in range(board.shape[1]):
            if board[row][col] == 0:
                continue

            tmp_score = 0
            total_neighbours = 0

            for k in OFFSETS:
                for l in OFFSETS:
                    p = row + k
                    q = col + l

                    if in_range(board, p, q) and board[p][q] > 0:
                        total_neighbours += 1
                        tmp_score += abs(board[row][col] - board[p][q])

            if total_neighbours:
                clustering_score += (tmp_score / total_neighbours)

    # lower clustering_score means better board
    min_val = np.sum(np.diff(np.sort(board.ravel())))
    # return 1 - normalize(clustering_score , min_val, clustering_score)
    return clustering_score


def normalize(value: Union[float, np.ndarray], min_val: float, max_val: float):
    if max_val == min_val:
        return value

    if not isinstance(value, np.ndarray) and value - min_val == 0:
        normalized = 0
    else:
        normalized = (value - min_val) / (max_val - min_val)

    # # TODO remote asserts
    # if isinstance(value, np.ndarray):
    #     if not all(0 <= normalized) and all(normalized <= 1):
    #         assert False
    #
    # else:
    #     if not 0 <= normalized <= 1:
    #         assert False

    return normalized

def better_evaluation_function(current_game_state):
    """
    Your extreme 2048 evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    board: np.ndarray = current_game_state.board
    max_tile = current_game_state.max_tile
    score = current_game_state.score

    WEIGHTS = [2, 1, 0] # TODO can change coeffs

    weights = np.array(WEIGHTS)

    # coef = normalize(weights, min(weights), max(weights))
    coef = WEIGHTS

    scores = [monotonicity(board),
              empty_board(board),
              clustering(board)]


    # print(f'scores before coeffs = {scores}')
    return np.sum(np.multiply(scores, coef))


# Abbreviation
better = better_evaluation_function
