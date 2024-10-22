from easyAI import TwoPlayerGame

class GrundysGame(TwoPlayerGame):
    """
    Class representing the Grundy's game, where two players divide piles of objects.
    The game ends when a player cannot make a move.

    Attributes:
        players (list): List of players in the game (human and AI).
        stacks (list): List of stacks with the number of objects.
        current_player (int): Index of the current player.
    """

    def __init__(self, players, initial_pile=15):
        """
        Initializes a new instance of the Grundy's game.

        Args:
            players (list): List of players (human and AI).
            initial_pile (int): Number of objects in the initial stack.
        """
        self.players = players
        self.stacks = [initial_pile]
        self.current_player = 1

    def possible_moves(self):
        """
        Returns a list of possible moves for the current player.

        Returns:
            list: List of possible moves in the format "i j k".
        """
        moves = []
        for i, pile in enumerate(self.stacks):
            for j in range(1, pile):
                for k in range(1, pile - j + 1):
                    if j + k == pile and j != k:
                        moves.append(f'{i} {j} {k}')
        return moves

    def make_move(self, move):
        """
        Executes a move by splitting the selected stack into two new ones.

        Args:
            move (str): The move in the format "i j k", where i is the stack index,
                        j is the new number of objects in the first new stack,
                        and k is the new number of objects in the second new stack.
        """
        i, j, k = map(int, move.split())
        self.stacks[i] = j
        self.stacks.insert(i + 1, k)

    def win(self):
        """
        Checks if the player has won.

        Returns:
            bool: True if the player has won (all stacks have 2 or fewer), False otherwise.
        """
        return all(pile <= 2 for pile in self.stacks)

    def is_over(self):
        """
        Checks if the game is over.

        Returns:
            bool: True if the game is over, False otherwise.
        """
        return self.win()

    def show(self):
        """
        Displays the current state of stacks and their indices.
        """
        print(f"Stacks: {self.stacks}")
        print(f"------ {list(range(len(self.stacks)))}")

    def scoring(self):
        """
        Returns the score for the current player.

        Returns:
            int: 100 if the player has won, 0 otherwise.
        """
        return 100 if self.win() else 0
