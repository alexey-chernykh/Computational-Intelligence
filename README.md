
# Computational Intelligence Portfolio

**Author:** Oleksii Chernykh
**Context:** 5th-year Master's course projects in Informatics, Kharkiv National University of Radio Electronics (NURE).

## Overview

This repository contains a collection of projects demonstrating the practical application of computational intelligence methods. The portfolio covers the development of Convolutional Neural Networks (CNNs), Fuzzy Logic systems, the application of evolutionary algorithms (Genetic Algorithms, Swarm Intelligence), and the creation of hybrid neuro-fuzzy systems to solve complex optimization and classification tasks.

---

## Projects Included

### 1. Custom Image Classification using CNNs

Developed a Convolutional Neural Network to classify nature images (Kaggle Natural Scenes Dataset).

* **Data Processing:** Resized images to a fixed format (128x128), normalized pixels, and split the data into training and testing sets.
* **Model Architecture:** Built a Sequential CNN using Keras, incorporating `Conv2D` layers (32, 64, and 128 filters), `MaxPooling2D`, and `Dropout` to prevent overfitting.
* **Evaluation:** Analyzed accuracy and loss during training and validation stages; visualized the Confusion Matrix and the distribution of predictions across classes (forest, mountain, sea, etc.).

### 2. Fuzzy Logic Decision System for Autonomous Vehicles

Created a decision-making system for an autonomous vehicle in uncertain conditions using the `skfuzzy` library.

* **Input Linguistic Variables:** "Distance to obstacle" (distance), "Speed" (speed), "Road condition" (road_condition).
* **Output Variable:** "Action" (action) — ranging from hard braking to accelerating.
* **Rule Base:** Configured fuzzy inference rules (e.g., if distance is short and speed is high, the action is hard braking).
* **Simulation:** Applied defuzzification (centroid method) to obtain a specific numerical value for acceleration/braking based on various simulated scenarios.

### 3. Neural Network Hyperparameter Optimization (Genetic Algorithm)

Utilized a Genetic Algorithm to find the optimal Artificial Neural Network (ANN) architecture on the Digits dataset.

* **Genetic Representation:** Encoded hyperparameters (number of hidden layers, number of neurons, activation function type like `relu`/`tanh`) as chromosomes using the `DEAP` library.
* **Evolutionary Operators:** Applied tournament selection, two-point crossover, and mutations to generate new generations.
* **Fitness Function:** Evaluated each configuration by rapidly training the model and using validation accuracy as the survival criterion.
* **Result:** Automatically discovered the best network configuration that provides the highest classification accuracy for handwritten digits.

### 4. Logistics Routing Optimization (Particle Swarm Optimization - PSO)

Solved the Vehicle Routing Problem using the Particle Swarm Optimization algorithm.

* **Search Space:** Represented point visitation routes (depot and clients) as particle coordinates in a multidimensional space.
* **Cost Function:** Minimized the total Euclidean distance (`distance.cdist`) for the complete delivery route.
* **Mathematical Model:** Updated the velocity and position of each particle based on its personal best known position (pBest) and the swarm's global best known position (gBest).
* **Visualization:** Plotted the algorithm's convergence graph and displayed the optimal route on a 2D plane.

### 5. Hybrid System: Neural Network + Fuzzy Logic

Predicted medical disease risks by combining classical Deep Learning with Fuzzy Logic expert rules.

* **Neural Network Component:** Trained a model for primary classification to obtain the probability of heart disease based on numerical medical indicators of patients.
* **Fuzzy Component:** Used the neural network's probability output as one of the input parameters, combining it with cholesterol levels.
* **Hybrid Inference:** Applied Mamdani fuzzy inference rules to adjust the prediction, resulting in a more interpretable and accurate "Risk Level" percentage.

---

## Technologies & Libraries

* **Programming Language:** Python
* **Deep Learning & ANNs:** TensorFlow, Keras
* **Fuzzy Logic:** scikit-fuzzy (`skfuzzy`)
* **Evolutionary Algorithms:** DEAP (Distributed Evolutionary Algorithms in Python)
* **Machine Learning:** Scikit-learn
* **Data Analysis & Visualization:** NumPy, Pandas, Matplotlib, SciPy

---

## Installation & Setup

1. Clone the repository:
```bash
git clone https://github.com/alexey-chernykh/[YOUR_REPOSITORY_NAME].git

```


2. Install the required dependencies:
```bash
pip install -r requirements.txt

```


3. Navigate to the respective laboratory folder and open the `.ipynb` files in Jupyter Notebook or run the Python scripts.