from typing import Tuple, List, Dict, Deque, Any, Optional
from math import *
import copy
from collections import deque
import random
import time

from snakepit.robot_snake import RobotSnake
from snakepit.datatypes import Position, Vector



class MyRobotSnake(RobotSnake):
    def __init__(self, game_settings, world, color):
        super().__init__(game_settings, world, color)
        self._players: List[Player] = []


    def next_direction(self, initial=False):
        """
        This method sends the next direction of the robot snake to the server.
        The direction is changed on the next frame load.

        The return value should be one of: `self.UP`, `self.DOWN`, `self.LEFT`, `self.RIGHT` or `None`.

        The snake's world (`self.world`) is a two-dimensional array that changes during each frame.
        Each point in the matrix is a two-item tuple `(char, color)`.

        Each snake has a different color represented by an integer.
        Your snake's color is available in `self.color`.

        More information can be found in the Snake documentation.
        """
        """ Play a sample game between two UCT players where each player gets a different number
            of UCT iterations (= simulations = tree nodes).
        """
        if initial:
            self._players = self._find_snake_heads()
            for index, player in enumerate(self._players):
                self._players[index] = self._find_snake_rest(player[0], is_player = index == 0)

        for player in self._players:
            old_head = player[0]
            _, color = self.world[old_head.x][old_head.y]
            head = self._update_snake_part(player[0], self.CH_HEAD, color)
            if head != player[0]:
                player.appendleft(head)
            tail = self._update_snake_part(player[-1], self.CH_TAIL, color)
            if tail != player[-1]:
                player.pop()
                assert tail == player[-1]

        state = GameState(self.world, self._players)
        m = UCT(rootstate = state, itermax = 50, maxdepth = 5, verbose = True)
        print('Sending', m)
        return m


    def _find_snake_heads(self):
        player_head: Position = None
        opponent_head: Position = None
        for x in range(self.world.SIZE_X):
            for y in range(self.world.SIZE_Y):
                char, color = self.world[x][y]
                if char == self.CH_HEAD:
                    if color == self.color:
                        player_head = Position(x, y)
                    else:
                        opponent_head = Position(x, y)
        if opponent_head:
            return [deque([player_head]), deque([opponent_head])]
        else:
            return [deque([player_head])]


    def _neighbours(self, position: Position):
        return [
            (position.x + 1, position.y),
            (position.x - 1, position.y),
            (position.x, position.y + 1),
            (position.x, position.y - 1),
        ]


    def _find_snake_rest(self, head: Position, is_player: bool):
        def next(position: Position, previous: Position):
            for x, y in self._neighbours(position):
                if x != previous.x or y != previous.y:
                    char, color = self.world[x][y]
                    if (char == self.CH_BODY or char == self.CH_TAIL) and (color == self.color) == is_player:
                        return Position(x, y)

        snake = deque([head])
        previous = head
        position = head
        while True:
            position, previous = next(position, previous), position
            if not position:
                break
            snake.append(position)
        return snake


    def _update_snake_part(self, old_position: Position, char_to_find: str, old_color: str):
        for x, y in self._neighbours(old_position) + [Position(old_position.x, old_position.y)]:
            char, color = self.world[x][y]
            if char == char_to_find and color == old_color:
                return Position(x, y)


# This is a very simple implementation of the UCT Monte Carlo Tree Search algorithm in Python 2.7.
# The function UCT(rootstate, itermax, verbose = False) is towards the bottom of the code.
# It aims to have the clearest and simplest possible code, and for the sake of clarity, the code
# is orders of magnitude less efficient than it could be made, particularly by using a
# state.GetRandomMove() or state.DoRandomRollout() function.
#
# Example GameState classes for Nim, OXO and Othello are included to give some idea of how you
# can write your own GameState use UCT in your 2-player game. Change the game to be played in
# the UCTPlayGame() function at the bottom of the code.
#
# Written by Peter Cowling, Ed Powley, Daniel Whitehouse (University of York, UK) September 2012.
#
# Licence is granted to freely use and distribute for any sensible/legal purpose so long as this comment
# remains in any distributed code.
#
# For more information about Monte Carlo Tree Search check out our web site at www.mcts.ai
Map = List[List[Tuple[str, str]]]
Player = Deque[Position]
Actions = {RobotSnake.UP, RobotSnake.DOWN, RobotSnake.LEFT, RobotSnake.RIGHT}


class GameState:
    """ A state of the game, i.e. the game board. These are the only functions which are
        absolutely necessary to implement UCT in any 2-player complete information deterministic
        zero-sum game, although they can be enhanced and made quicker, for example by using a
        GetRandomMove() function to generate a random move during rollout.
        By convention the players are numbered 0 and 1.
    """
    def __init__(self, terrain: Map, players: List[Player]) -> None:
        self.playerJustMoved = 1 # At the root pretend the player just moved is player 2 - player 1 has the first move
        self.terrain = terrain
        self.players = players
        self.last_action: List[Vector] = [None, None]
        self.scores = [0, 0]

    def clone(self) -> 'GameState':
        """ Clone this game state, don't deepcopy board.
        """
        st = GameState(self.terrain, copy.deepcopy(self.players))
        st.last_action = self.last_action[:]
        st.scores = self.scores[:]
        return st

    def do_move(self, move):
        """ Update a state by carrying out the given move.
            Must update playerJustMoved.
        """
        if len(self.players) > 1:
            self.playerJustMoved = 1 - self.playerJustMoved
            player_moving = self.playerJustMoved
        else:
            player_moving = 0
            self.playerJustMoved = 0
        self.last_action[player_moving] = move

        player = self.players[player_moving]
        head = player[0]
        new_head = Position(head.x + move.xdir, head.y + move.ydir)

        char, color = self.terrain[new_head.x][new_head.y]
        if self._collides_with_player(new_head):
            self.scores[player_moving] = -1000
        elif char.isnumeric():
            self.scores[player_moving] += int(char)
        elif char in {RobotSnake.CH_STONE, '-', '|', '+'}.union(RobotSnake.DEAD_BODY_CHARS):
            self.scores[player_moving] = -1000

        player.appendleft(new_head)
        player.pop()

    def get_moves(self):
        """ Get all possible moves from this state.
        """
        if len(self.players) > 1:
            player_moving = 1 - self.playerJustMoved
        else:
            player_moving = 0
        if self.last_action[player_moving] is not None:
            return list(Actions.difference({self._opposite_action(self.last_action[player_moving])}))
        else:
            return list(Actions)

    def get_result(self, playerjm):
        """ Get the game result from the viewpoint of playerjm.
        """
        # TODO: implement heuristics
        return self.scores[playerjm]

    def _collides_with_player(self, position: Position) -> bool:
        for player in self.players:
            if position in player:
                return True
        return False

    def _opposite_action(self, action: Vector) -> Vector:
        return Vector(-action.xdir, -action.ydir)


    def __repr__(self):
        """ Don't need this - but good style.
        """
        return f'moved: {self.playerJustMoved}, players: {self.players}, score: {self.scores}, last_action: {self.last_action}'


class OXOState:
    """ A state of the game, i.e. the game board.
        Squares in the board are in this arrangement
        012
        345
        678
        where 0 = empty, 1 = player 1 (X), 2 = player 2 (O)
    """
    def __init__(self):
        self.playerJustMoved = 2 # At the root pretend the player just moved is p2 - p1 has the first move
        self.board = [0,0,0,0,0,0,0,0,0] # 0 = empty, 1 = player 1, 2 = player 2

    def Clone(self):
        """ Create a deep clone of this game state.
        """
        st = OXOState()
        st.playerJustMoved = self.playerJustMoved
        st.board = self.board[:]
        return st

    def DoMove(self, move):
        """ Update a state by carrying out the given move.
            Must update playerToMove.
        """
        assert move >= 0 and move <= 8 and move == int(move) and self.board[move] == 0
        self.playerJustMoved = 3 - self.playerJustMoved
        self.board[move] = self.playerJustMoved

    def GetMoves(self):
        """ Get all possible moves from this state.
        """
        return [i for i in range(9) if self.board[i] == 0]

    def GetResult(self, playerjm):
        """ Get the game result from the viewpoint of playerjm.
        """
        for (x,y,z) in [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]:
            if self.board[x] == self.board[y] == self.board[z]:
                if self.board[x] == playerjm:
                    return 1.0
                else:
                    return 0.0
        if self.GetMoves() == []: return 0.5 # draw
        assert False # Should not be possible to get here

    def __repr__(self):
        s = ""
        for i in range(9):
            s += ".XO"[self.board[i]]
            if i % 3 == 2: s += "\n"
        return s


class Node:
    """ A node in the game tree. Note wins is always from the viewpoint of playerJustMoved.
        Crashes if state not specified.
    """
    def __init__(self, move = None, parent = None, state = None):
        self.move = move # the move that got us to this node - "None" for the root node
        self.parentNode = parent # "None" for the root node
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.untriedMoves = state.get_moves() # future child nodes
        self.playerJustMoved = state.playerJustMoved # the only part of the state that the Node needs later

    def uct_select_child(self):
        """ Use the UCB1 formula to select a child node. Often a constant UCTK is applied so we have
            lambda c: c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits to vary the amount of
            exploration versus exploitation.
        """
        max_value = float('-inf')
        max_node = 0
        for c in self.childNodes:
            value = c.wins/c.visits + sqrt(2*log(self.visits)/c.visits)
            if value > max_value:
                max_value = value
                max_node = c
        return max_node

    def add_child(self, m, s):
        """ Remove m from untriedMoves and add a new child node for this move.
            Return the added child node
        """
        n = Node(move = m, parent = self, state = s)
        self.untriedMoves.remove(m)
        self.childNodes.append(n)
        return n

    def update(self, result):
        """ Update this node - one additional visit and result additional wins. result must be from the viewpoint of playerJustmoved.
        """
        self.visits += 1
        self.wins += result

    def __repr__(self):
        return "[M:" + str(self.move) + " W/V:" + str(self.wins) + "/" + str(self.visits) + " U:" + str(self.untriedMoves) + "]"

    def tree_to_string(self, indent):
        s = self.indent_string(indent) + str(self)
        for c in self.childNodes:
             s += c.tree_to_string(indent + 1)
        return s

    def indent_string(self, indent):
        s = "\n"
        for i in range(1, indent+1):
            s += "| "
        return s

    def children_to_string(self):
        s = ""
        for c in self.childNodes:
             s += str(c) + "\n"
        return s


def UCT(rootstate, itermax, maxdepth, verbose = False):
    """ Conduct a UCT search for itermax iterations starting from rootstate.
        Return the best move from the rootstate.
        Assumes 2 alternating players (player 1 starts), with game results in the range [0.0, 1.0]."""
    # TODO: iterate until timeout, drop itermax
    root_node = Node(state = rootstate)
    start_time = time.monotonic()

    for i in range(itermax):
        node = root_node
        state = rootstate.clone()

        # Select
        while node.untriedMoves == [] and node.childNodes != []: # node is fully expanded and non-terminal
            node = node.uct_select_child()
            state.do_move(node.move)

        # Expand
        if node.untriedMoves: # if we can expand (i.e. state/node is non-terminal)
            m = random.choice(node.untriedMoves)
            state.do_move(m)
            node = node.add_child(m, state) # add child and descend tree

        # Rollout - this can often be made orders of magnitude quicker using a state.GetRandomMove() function
        for j in range(maxdepth):
            try:
                # TODO: use heuristics for selection?
                # TODO: cut bad branches
                state.do_move(random.choice(state.get_moves()))
            except ValueError:
                break # while state is non-terminal

        # Back-propagate
        while node is not None: # back-propagate from the expanded node and work back to the root node
            node.update(state.get_result(node.playerJustMoved)) # state is terminal. Update node with result from POV of node.playerJustMoved
            node = node.parentNode

    # Output some information about the tree - can be omitted
    if verbose:
        print(root_node.tree_to_string(0))
        print(f'Timing: {time.monotonic() - start_time}s')
    else:
        print(root_node.children_to_string())

    return sorted(root_node.childNodes, key = lambda c: c.visits)[-1].move # return the move that was most visited


if __name__ == '__main__':
    print('Ok')
