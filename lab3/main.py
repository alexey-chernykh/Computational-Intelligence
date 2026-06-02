import numpy as np

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from tensorflow import keras
from keras import layers

import random
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt


data = load_digits()
X = data.data
y = data.target

encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


def create_model(n_layers, n_neurons, activation, input_shape, n_classes):
    model = keras.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))

    for _ in range(n_layers):
        model.add(layers.Dense(n_neurons, activation=activation))
        
    model.add(layers.Dense(n_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("attr_layers", random.randint, 1, 3)
toolbox.register("attr_neurons", random.choice, [16, 32, 64, 128])
toolbox.register("attr_act", random.randint, 0, 1)

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_layers, toolbox.attr_neurons, toolbox.attr_act), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate_model(individual):
    n_layers = individual[0]
    n_neurons = individual[1]
    activation = 'relu' if individual[2] == 0 else 'tanh'
    
    model = create_model(n_layers, n_neurons, activation, (64,), 10)

    model.fit(X_train, y_train, epochs=5, verbose=0) 
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return (accuracy,)

toolbox.register("evaluate", evaluate_model)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=[1, 16, 0], up=[3, 128, 1], indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    pop = toolbox.population(n=10)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, 
                                   stats=stats, halloffame=hof, verbose=True)
    
    print("Найкращі параметри:", hof[0])

    best_n_layers = hof[0][0]
    best_n_neurons = hof[0][1]
    if hof[0][2] == 1:
        best_activation = 'tanh'
    else:
        best_activation = 'relu'

    print(f"Тренування фінальної моделі з параметрами: "
        f"Layers={best_n_layers}, Neurons={best_n_neurons}, Act={best_activation}")

    final_model = create_model(
        n_layers=best_n_layers, 
        n_neurons=best_n_neurons, 
        activation=best_activation, 
        input_shape=(64,),  # 64 ознаки для load_digits
        n_classes=10        # 10 цифр
    )

    history = final_model.fit(
        X_train, y_train, 
        epochs=20,          # Даємо більше часу на навчання
        batch_size=32, 
        verbose=1,
        validation_split=0.1
    )

    final_loss, final_acc = final_model.evaluate(X_test, y_test, verbose=0)

    print("-" * 30)
    print(f"Фінальна точність (Accuracy): {final_acc:.4f}")
    print("-" * 30)

    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Навчання найкращої моделі')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()


main()