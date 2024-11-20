from similarity import Similarity


class Recommender:
    """
    A class to generate movie recommendations for users based on similarity metrics.
    """
    def __init__(self, dataset, similarity_function):
        """
        Initialize the recommender system.

        Args:
            dataset (dict): The dataset containing user ratings.
            similarity_function (function): The similarity function to use.
        """
        self.dataset = dataset
        self.similarity_function = similarity_function

    def find_similar_users(self, user, top_n=3):
        """
        Find the most similar users to the specified user.

        Args:
            user (str): The target user.
            top_n (int, optional): Number of similar users to return. Defaults to 3.

        Returns:
            list: A list of tuples (user, similarity_score), sorted by similarity.
        """
        scores = {
            other_user: self.similarity_function(self.dataset, user, other_user)
            for other_user in self.dataset if other_user != user
        }

        is_reverse = False if Similarity.euclidean == self.similarity_function else True

        return sorted(scores.items(), key=lambda x: x[1], reverse=is_reverse)[:top_n]

    def recommend_movies(self, user, similar_users, top_n=5):
        """
        Recommend movies for the target user based on similar users' ratings.

        Args:
            user (str): The target user.
            similar_users (list): A list of tuples (user, similarity_score).
            top_n (int, optional): Number of recommendations to return. Defaults to 5.

        Returns:
            list: A list of tuples (movie, score), sorted by score.
        """
        user_ratings = self.dataset[user]
        recommendations = {}

        for similar_user, similarity in similar_users:
            if Similarity.euclidean == self.similarity_function:
                similarity = 1 / (1 + similarity)

            for movie, rating in self.dataset[similar_user].items():
                if movie not in user_ratings:
                    if movie not in recommendations:
                        recommendations[movie] = 0
                    recommendations[movie] += similarity * rating

        for movie in recommendations:
            total_similarity = sum(
                1 / (1 + sim[1]) if Similarity.euclidean == self.similarity_function else sim[1]
                for sim in similar_users
            )
            recommendations[movie] /= total_similarity

        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return sorted_recommendations[:top_n]
