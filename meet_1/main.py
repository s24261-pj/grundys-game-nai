from easyAI import Negamax
from game_handler import GameHandler

def main():
    """
    Main function to initialize the game and start the gameplay.

    This function sets up the AI algorithm and starts the game handler,
    allowing players to engage in Grundy's game once.
    """
    ai_algorithm = Negamax(5)

    game_handler = GameHandler(ai_algorithm)

    game_handler.start_game()

if __name__ == "__main__":
    main()
