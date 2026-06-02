import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import skfuzzy as fuzz
from skfuzzy import control as ctrl

print("Генерація даних...")
np.random.seed(42)
n_samples = 1000

age = np.random.randint(20, 80, n_samples)    # років
chol = np.random.randint(150, 400, n_samples) # мг/дл
bps = np.random.randint(90, 180, n_samples)   # мм рт.ст.

risk_factor = (age / 80) + (chol / 400) + (bps / 180)
labels = (risk_factor + np.random.normal(0, 0.2, n_samples)) > 1.8
labels = labels.astype(int)

df = pd.DataFrame({'age': age, 'chol': chol, 'bps': bps, 'target': labels})

X = df[['age', 'chol', 'bps']].values
y = df['target'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("Навчання нейронної мережі...")

model = keras.Sequential([
    layers.InputLayer(input_shape=(3,)),    # Age, Chol, BPS
    layers.Dense(16, activation='relu'),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')   # Вихід 0-1
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0, validation_split=0.1)
nn_loss, nn_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Точність базової нейромережі: {nn_acc:.4f}")

nn_predictions = model.predict(X_test).flatten()

print("Налаштування нечіткої логіки...")

nn_prob = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'nn_prob')
chol_lvl = ctrl.Antecedent(np.arange(100, 401, 1), 'chol')

final_risk = ctrl.Consequent(np.arange(0, 101, 1), 'risk')

nn_prob['low'] = fuzz.trimf(nn_prob.universe, [0, 0, 0.5])
nn_prob['medium'] = fuzz.trimf(nn_prob.universe, [0.3, 0.5, 0.7])
nn_prob['high'] = fuzz.trimf(nn_prob.universe, [0.5, 1.0, 1.0])

chol_lvl['normal'] = fuzz.trimf(chol_lvl.universe, [100, 100, 240])
chol_lvl['high'] = fuzz.trapmf(chol_lvl.universe, [200, 240, 400, 400])

final_risk['low'] = fuzz.trimf(final_risk.universe, [0, 0, 50])
final_risk['medium'] = fuzz.trimf(final_risk.universe, [25, 50, 75])
final_risk['high'] = fuzz.trimf(final_risk.universe, [50, 100, 100])

rule1 = ctrl.Rule(nn_prob['low'] & chol_lvl['normal'], final_risk['low'])
rule2 = ctrl.Rule(nn_prob['low'] & chol_lvl['high'], final_risk['medium'])
rule3 = ctrl.Rule(nn_prob['medium'], final_risk['medium'])
rule4 = ctrl.Rule(nn_prob['high'], final_risk['high'])
rule5 = ctrl.Rule(nn_prob['high'] & chol_lvl['high'], final_risk['high'])

risk_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
risk_sim = ctrl.ControlSystemSimulation(risk_ctrl)

print("Запуск гібридної системи...")

hybrid_results = []
test_samples_indices = range(20) 

for i in test_samples_indices:
    patient_chol = X_test[i][1] * scaler.scale_[1] + scaler.mean_[1]
    patient_nn_prob = nn_predictions[i]

    risk_sim.input['nn_prob'] = patient_nn_prob
    risk_sim.input['chol'] = patient_chol

    try:
        risk_sim.compute()
        result = risk_sim.output['risk']
    except:
        result = 50 # Fallback
        
    hybrid_results.append(result)



nn_prob.view()
final_risk.view()

plt.figure(figsize=(12, 6))
plt.plot(nn_predictions[:20] * 100, label='NN Prediction (%)', marker='o', linestyle='--')
plt.plot(hybrid_results, label='Hybrid Fuzzy Risk (%)', marker='x', linewidth=2)
plt.axhline(y=50, color='r', linestyle=':', alpha=0.5, label='Threshold')
plt.title('Порівняння: Чиста Нейромережа vs Гібридна Система')
plt.ylabel('Рівень ризику (0-100)')
plt.xlabel('Індекс пацієнта')
plt.legend()
plt.grid(True)
plt.show()

print("\nПриклад роботи системи:")
print(f"{'NN Prob':<10} | {'Cholesterol':<12} | {'Hybrid Risk':<12} | {'Diagnosis'}")
print("-" * 50)
for i in range(5):
    p_chol = X_test[i][1] * scaler.scale_[1] + scaler.mean_[1]
    diag = "SICK" if hybrid_results[i] > 50 else "HEALTHY"
    print(f"{nn_predictions[i]:.4f}     | {p_chol:.1f}        | {hybrid_results[i]:.2f}       | {diag}")