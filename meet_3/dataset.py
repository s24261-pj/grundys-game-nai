import json


class Dataset:
    """
    A utility class to handle loading datasets from files.
    """
    @staticmethod
    def load_from_file(file_path):
        """
        Load a dataset from a JSON file.

        Args:
            file_path (str): The path to the JSON file.

        Returns:
            dict: The loaded dataset as a dictionary.
        """
        with open(file_path, 'r') as f:
            return json.load(f)
