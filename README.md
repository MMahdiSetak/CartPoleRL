# REINFORCE CartPole Project
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/MMahdiSetak/CartPoleRL/blob/main/LICENSE)

Harness the power of reinforcement learning to master the classic CartPole balancing challenge.

## Description
The project applies the REINFORCE algorithm, enabling an agent to adeptly balance a pole on a cart. Underpinned by a neural network, the agent evolves its capabilities, with all training advancements, including hyperparameter fine-tuning, elegantly visualized and logged via TensorBoard.

## Prerequisites
- Python 3.x (tested on 3.10)
- pip

## Installation

1. Clone this repository to your local machine.
2. Navigate to the project directory and install the required libraries using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the main training script:
    ```bash
    python main.py
    ```

2. To visualize training progress, use TensorBoard:
    ```bash
    tensorboard --logdir=cartpole_tensorboard
    ```

## Features

- **Hyperparameter Grid Search**: Comprehensive exploration of parameter space to optimize performance.
- **Visualization**: The `utils.py` offers robust tools for plotting learning curves and visual demonstrations of the
  agent's expertise.
- **Policy Network**: A lean and efficient neural network, as detailed in `model.py`, underpins the agent's policy.
- **Trajectory Collection and Training**: The `policy.py` equips users with functions for trajectory collection and
  REINFORCE algorithm-based training.

## Results

Visual insights into the project's performance:

- **Hyperparameter Grid Search**:<br>
  ![Grid Search Results](./assets/images/grid_search.png)
- **Learning Curve** (Average of 10 runs):<br>
  ![Average Learning Curve](./assets/images/10runs.png)
- **Learning Process Visualization**:<br>
  ![Learning Process GIF](./assets/gifs/learning_process.gif)
- **Sample Learning Curve**:<br>
  ![Sample Performance GIF](./assets/images/sample.png)
- **Sample Performance**:<br>
  ![Sample Performance GIF](./assets/gifs/sample.gif)

## Models
All trained models are diligently saved in the `assets/saved_models` directory. As an example, the model trained with a
hidden layer of size 4 is available [here](./assets/saved_models/model_4.pth).

## Contributing
Your contributions are welcomed:

1. Fork the Project.
2. Establish your Feature Branch (`git checkout -b feature/AmazingFeature`).
3. Commit the Changes (`git commit -m 'Introduce AmazingFeature'`).
4. Push to the Branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

## License
This project is licensed under the MIT License. Refer to [`LICENSE`](./LICENSE) for detailed information.