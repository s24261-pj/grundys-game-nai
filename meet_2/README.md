# Air Conditioning Power Generator

This project implements a fuzzy logic-based system for controlling air conditioning power. The system calculates the ideal cooling power based on three main inputs: indoor temperature, outdoor temperature, and the number of people in the room. By using fuzzy logic, this system can simulate realistic adjustments to cooling power to maintain comfort effectively.

## System Rules and Description

This air conditioning power generator uses fuzzy logic to determine the cooling power output dynamically. Based on inputs for indoor and outdoor temperatures and the number of people, the system generates appropriate responses for cooling needs, ranging from disabled to very high power. 

## Authors

- **Mateusz Kopczyński (s24261)**
- **Artur Szulc (s24260)**

## Environment Setup

To run this fuzzy logic-based air conditioning power generator locally, set up the environment as follows:

### Prerequisites

- Python 3.12+
- `numpy` and `scikit-fuzzy` libraries for implementing fuzzy logic rules and computations

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/s24261-pj/grundys-game-nai.git
    cd grundys-game-nai
    ```

2. Move to 2-nd meet

   ```bash
   cd meet_2
   ```

3. Install Required Dependencies

    Install the dependencies listed in `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

    The `requirements.txt` file should contain:

    ```text
    numpy
    scikit-fuzzy
    scipy
    packaging
    networkx
    ```

4. Run the System

    After installation, you can start the program with:

    ```bash
    python main.py
    ```

## How the System Works

The system uses fuzzy logic to handle the following inputs:

- **Temperature Inside**: Defines the comfort level of the indoor temperature.
- **Temperature Outside**: Provides context for external conditions that may affect indoor climate.
- **Number of People**: Accounts for the room occupancy level, impacting cooling needs.

### Fuzzy Logic Rules

The system defines fuzzy sets and membership functions for each input, which are then combined to determine the `Cooling Power` output. Based on specific combinations of inputs, the fuzzy rules generate the most suitable cooling power level.

### Example Fuzzy Rule

For instance, if the indoor temperature is "very hot" and the room is "full," the system will set the cooling power to "very high."

## Membership functions universe visualization
![image](https://github.com/user-attachments/assets/f4e39f55-14a7-429b-873e-21bb297273f5)
![image](https://github.com/user-attachments/assets/0c2e19eb-9dab-47ed-b4ef-286b739f290a)
![image](https://github.com/user-attachments/assets/6a267876-7b72-4eb4-8bfd-2c764943763e)

## Screenshots
![image](https://github.com/user-attachments/assets/a39ca860-17f3-4340-a0f7-ca2b56b9a54d)
![image](https://github.com/user-attachments/assets/99ba40f6-2e95-49af-a174-e9a17d82a044)
![image](https://github.com/user-attachments/assets/115cdbd5-6aeb-488e-86ba-115e572589cc)
