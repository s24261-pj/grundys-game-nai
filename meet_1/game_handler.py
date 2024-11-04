from grundys_game import GrundysGame
from easyAI import AI_Player, Human_Player

class GameHandler:
    """
    Class managing the flow of Grundy's game with AI.

    Attributes:
        ai_algorithm (object): The AI algorithm for making decisions in the game.
        initial_pile (int): Number of objects in the initial stack.
    """

    def __init__(self, ai_algorithm, initial_pile=15):
        """
        Initializes a new instance of GameHandler.

        Args:
            ai_algorithm (object): The AI algorithm for making decisions in the game.
            initial_pile (int): Number of objects in the initial stack.
        """
        self.ai_algorithm = ai_algorithm
        self.initial_pile = initial_pile

    def start_game(self):
        """
        Starts a new game by creating a game instance and running the gameplay.
        After the game ends, it displays a message about the winner.
        """
        players = [Human_Player(), AI_Player(self.ai_algorithm)]
        game = GrundysGame(players, initial_pile=self.initial_pile)
        game.play()

        print(f"Player {game.opponent_index} wins! Congratulations!")
