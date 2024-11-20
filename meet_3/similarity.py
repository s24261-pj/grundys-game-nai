import numpy as np


class Similarity:
    """
    A class containing similarity metrics for comparing users' ratings.
    """
    @staticmethod
    def euclidean(dataset, user1, user2):
        """
        Compute the Euclidean similarity score.

        Args:
            dataset (dict): The dataset containing user ratings.
            user1 (str): The first user.
            user2 (str): The second user.

        Returns:
            float: The Euclidean similarity score.
        """
        common_movies = Similarity._find_common_movies(dataset, user1, user2)
        if not common_movies:
            return 0

        squared_diff = [
            np.square(dataset[user1][item] - dataset[user2][item])
            for item in common_movies
        ]

        return 1 / (1 + np.sqrt(np.sum(squared_diff)))

    @staticmethod
    def pearson(dataset, user1, user2):
        """
        Compute the Pearson similarity score.

        Args:
            dataset (dict): The dataset containing user ratings.
            user1 (str): The first user.
            user2 (str): The second user.

        Returns:
            float: The Pearson similarity score.
        """
        common_movies = Similarity._find_common_movies(dataset, user1, user2)
        if not common_movies:
            return 0

        user1_ratings = [dataset[user1][item] for item in common_movies]
        user2_ratings = [dataset[user2][item] for item in common_movies]

        user1_mean = np.mean(user1_ratings)
        user2_mean = np.mean(user2_ratings)

        numerator = np.sum(
            (dataset[user1][item] - user1_mean) * (dataset[user2][item] - user2_mean)
            for item in common_movies
        )
        denominator = np.sqrt(
            np.sum((dataset[user1][item] - user1_mean) ** 2 for item in common_movies)
            * np.sum((dataset[user2][item] - user2_mean) ** 2 for item in common_movies)
        )

        return numerator / denominator if denominator != 0 else 0

    @staticmethod
    def _find_common_movies(dataset, user1, user2):
        """
        Find movies rated by both users.

        Args:
            dataset (dict): The dataset containing user ratings.
            user1 (str): The first user.
            user2 (str): The second user.

        Returns:
            set: A set of movie keys rated by both users.
        """
        return set(dataset[user1].keys()).intersection(dataset[user2].keys())
