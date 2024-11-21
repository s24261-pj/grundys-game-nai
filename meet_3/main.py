import argparse
from dataset import Dataset
from similarity import Similarity
from recommender import Recommender


def build_arg_parser():
    """
    Build a command-line argument parser.

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    parser = argparse.ArgumentParser(description='Compute similarity and recommend movies.')
    parser.add_argument('--user', required=True, help='User for whom to recommend movies.')
    parser.add_argument('--score-type', required=True, choices=['Euclidean', 'Pearson'], help='Similarity metric.')
    return parser


if __name__ == '__main__':
    """
    Main entry point for the recommender system script. Parses arguments,
    computes similarity scores, and generates recommendations.
    """
    args = build_arg_parser().parse_args()
    user = args.user
    score_type = args.score_type

    data = Dataset.load_from_file('data.json')

    similarity_function = Similarity.euclidean if score_type == 'Euclidean' else Similarity.pearson

    recommender = Recommender(data, similarity_function)

    try:
        similar_users = recommender.find_similar_users(user)
        print(f"\nTop similar users to {user} using {score_type} metric:")
        for similar_user, score in similar_users:
            print(f"{similar_user}: {score:.2f}")

        recommendations = recommender.recommend_movies(user, similar_users)
        print(f"\nTop {len(recommendations)} recommended movies for {user}:")
        for movie, score in recommendations:
            print(f"{movie}: {score:.2f}")

        anti_recommendations = recommender.anti_recommend_movies(user, similar_users)
        print(f"\nTop {len(anti_recommendations)} anti-recommended movies for {user}:")
        for movie, score in anti_recommendations:
            print(f"{movie}: {score:.2f}")

    except KeyError as e:
        print(f"Error: {e}")
