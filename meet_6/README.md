# Flag Detection

This project implements a system for real-time flag detection using computer vision and machine learning techniques. The application utilizes the ultralytics and opencv-python libraries to analyze video input from a camera and identify which flag is currently visible. The training data for the model was created using the Roboflow platform.
## Authors

- **Mateusz Kopczyński (s24261)**
- **Artur Szulc (s24260)**

---

## Environment Setup

### Prerequisites

- Python 3.12+
- Required Libraries:
    - `ultralytics`
    - `opencv-python`

### Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/s24261-pj/grundys-game-nai.git
    cd grundys-game-nai
    ```

2. **Navigate to the Meet 6 directory**:
    ```bash
    cd meet_6
    ```

3. **Install Dependencies**:
   Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

   Example `requirements.txt` content:
    ```text
    ultralytics
    opencv-python
    ```
4. **Run The Application**:
    ```bash
    python train_yolo_model.py
    python main.py
    ```
---

## Video

