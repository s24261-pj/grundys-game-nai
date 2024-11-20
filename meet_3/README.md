# Movie Recommendation System

This project implements a movie recommendation system based on similarity metrics between users. The system calculates the similarity between users based on their movie ratings and recommends movies to a target user based on the ratings of similar users. It uses two main similarity metrics: **Euclidean Distance** and **Pearson Correlation**.

## Authors

- **Mateusz Kopczyński (s24261)**
- **Artur Szulc (s24260)**

## Environment Setup

To run the Movie Recommendation System locally, follow the setup steps below:

### Prerequisites

- Python 3.12+
- `numpy` for numerical computations
- `json` for handling the dataset

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/s24261-pj/grundys-game-nai.git
    cd grundys-game-nai
    ```

2. Move to 3-nd meet

   ```bash
   cd meet_3
   ```

3. Install Required Dependencies

    Install the dependencies listed in `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```
    The `requirements.txt` file should contain:

    ```text
    argparse
   json
   numpy
    ```

4. Run the System

    After installation, you can start the program with the following command:

    ```bash
    python main.py --user="<target_user>" --score-type="<similarity_metric>"
    ```

    - Replace `<target_user>` with the username of the user you want to get recommendations for.
    - Replace `<similarity_metric>` with either `Euclidean` or `Pearson` to select the similarity metric to use.

### Example:

```bash
python main.py --user="John Doe" --score-type="Pearson"
```

## How the System Works

### Inputs:
- **user**: The target user for whom we want to recommend movies.
- **score-type**: The similarity metric used to calculate user similarity. Choose between:
  - **Euclidean**: Based on the Euclidean distance between users' ratings.
  - **Pearson**: Based on the Pearson correlation coefficient between users' ratings.

### Steps:
1. **Compute Similarity**: The system calculates the similarity between the target user and all other users in the dataset using the selected metric (Euclidean or Pearson).
2. **Find Similar Users**: Based on the similarity scores, the system identifies the most similar users to the target user.
3. **Recommend Movies**: The system recommends movies that the similar users have rated highly but that the target user has not yet rated.

### Example Similarity Calculation
If **User1** and **User2** have rated the same movies similarly, they will be considered similar, and the system will recommend movies that **User2** has rated highly but **User1** hasn't watched yet.

## Similarity Metrics

### Euclidean Distance:
Measures the straight-line distance between two users' rating vectors.

### Pearson Correlation:
Measures the linear relationship between two users' ratings.

## Screenshots