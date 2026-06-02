import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import pandas as pd
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

speed_universe_max = 36 

try:
    data = pd.read_csv('archive/Self-driving Car Dataset/driving_log.csv')
    speed_desc = data['speed'].describe()
    print("--- Аналіз вхідних даних (speed) ---")
    print(speed_desc)

    max_speed_from_data = speed_desc['max']
    
    speed_universe_max = int(np.ceil(max_speed_from_data)) + 5
    
    print(f"\nДинамічно визначена верхня межа універсуму швидкості: {speed_universe_max}")
    print("------------------------------------------\n")
    
except Exception as e:
    print(f"Помилка завантаження даних (файл відсутній або пошкоджений): {e}")
    print(f"Використання універсуму за замовчуванням: 0-{speed_universe_max-1}\n")

#вхідні змінні
distance = ctrl.Antecedent(np.arange(0, 101, 1), 'distance')
speed = ctrl.Antecedent(np.arange(0, speed_universe_max, 1), 'speed')
road = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'road_condition')

#вихідна змінна
action = ctrl.Consequent(np.arange(-10, 11, 1), 'action', defuzzify_method='centroid')

distance['Close'] = fuzz.trapmf(distance.universe, [0, 0, 10, 25])
distance['Medium'] = fuzz.trimf(distance.universe, [15, 35, 55])
distance['Far'] = fuzz.trapmf(distance.universe, [45, 70, 100, 100])

speed['Low'] = fuzz.trapmf(speed.universe, [0, 0, 5, 15])
speed['Medium'] = fuzz.trimf(speed.universe, [10, 20, 30])
speed['High'] = fuzz.trapmf(speed.universe, [25, 30, 35, 35])

road['Icy'] = fuzz.trapmf(road.universe, [0, 0, 0.1, 0.3])
road['Wet'] = fuzz.trimf(road.universe, [0.2, 0.5, 0.8])
road['Good'] = fuzz.trapmf(road.universe, [0.7, 0.9, 1.0, 1.0])

action['Hard_Brake'] = fuzz.trimf(action.universe, [-10, -8, -6])
action['Light_Brake'] = fuzz.trimf(action.universe, [-7, -4, 0])
action['Maintain_Speed'] = fuzz.trimf(action.universe, [-2, 0, 2])
action['Light_Accelerate'] = fuzz.trimf(action.universe, [0, 3, 6])
action['Hard_Accelerate'] = fuzz.trimf(action.universe, [5, 8, 10])

rule1 = ctrl.Rule(distance['Far'] & speed['High'] & road['Good'], action['Maintain_Speed'])
rule2 = ctrl.Rule(distance['Far'] & speed['Medium'] & road['Good'], action['Light_Accelerate'])
rule3 = ctrl.Rule(distance['Far'] & speed['Low'] & road['Good'], action['Hard_Accelerate'])

rule4 = ctrl.Rule(distance['Medium'] & speed['High'] & road['Good'], action['Light_Brake'])
rule5 = ctrl.Rule(distance['Medium'] & speed['Medium'] & road['Good'], action['Maintain_Speed'])
rule6 = ctrl.Rule(distance['Medium'] & speed['Low'] & road['Good'], action['Light_Accelerate'])

rule7 = ctrl.Rule(distance['Close'] & speed['High'] & road['Good'], action['Hard_Brake'])
rule8 = ctrl.Rule(distance['Close'] & speed['Medium'] & road['Good'], action['Light_Brake'])
rule9 = ctrl.Rule(distance['Close'] & speed['Low'] & road['Good'], action['Maintain_Speed'])

rule10 = ctrl.Rule(distance['Far'] & speed['High'] & road['Wet'], action['Light_Brake'])
rule11 = ctrl.Rule(distance['Far'] & speed['Medium'] & road['Wet'], action['Maintain_Speed'])
rule12 = ctrl.Rule(distance['Far'] & speed['Low'] & road['Wet'], action['Light_Accelerate'])

rule13 = ctrl.Rule(distance['Medium'] & speed['High'] & road['Wet'], action['Hard_Brake'])
rule14 = ctrl.Rule(distance['Medium'] & speed['Medium'] & road['Wet'], action['Light_Brake'])
rule15 = ctrl.Rule(distance['Medium'] & speed['Low'] & road['Wet'], action['Maintain_Speed'])

rule16 = ctrl.Rule(distance['Close'] & speed['High'] & road['Wet'], action['Hard_Brake'])
rule17 = ctrl.Rule(distance['Close'] & speed['Medium'] & road['Wet'], action['Hard_Brake'])
rule18 = ctrl.Rule(distance['Close'] & speed['Low'] & road['Wet'], action['Light_Brake'])

rule19 = ctrl.Rule(distance['Far'] & speed['High'] & road['Icy'], action['Hard_Brake']) 
rule20 = ctrl.Rule(distance['Far'] & speed['Medium'] & road['Icy'], action['Light_Brake'])
rule21 = ctrl.Rule(distance['Far'] & speed['Low'] & road['Icy'], action['Maintain_Speed']) 

rule22 = ctrl.Rule(distance['Medium'] & speed['High'] & road['Icy'], action['Hard_Brake'])
rule23 = ctrl.Rule(distance['Medium'] & speed['Medium'] & road['Icy'], action['Hard_Brake'])
rule24 = ctrl.Rule(distance['Medium'] & speed['Low'] & road['Icy'], action['Light_Brake'])

rule25 = ctrl.Rule(distance['Close'] & speed['High'] & road['Icy'], action['Hard_Brake'])
rule26 = ctrl.Rule(distance['Close'] & speed['Medium'] & road['Icy'], action['Hard_Brake'])
rule27 = ctrl.Rule(distance['Close'] & speed['Low'] & road['Icy'], action['Hard_Brake'])

fis_control_system = ctrl.ControlSystem([
    rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9,
    rule10, rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18,
    rule19, rule20, rule21, rule22, rule23, rule24, rule25, rule26, rule27
])

fis_simulator = ctrl.ControlSystemSimulation(fis_control_system)

fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, figsize=(10, 16))

ax0.plot(distance.universe, fuzz.trapmf(distance.universe, [0, 0, 10, 25]), 'b', linewidth=1.5, label='Close')
ax0.plot(distance.universe, fuzz.trimf(distance.universe, [15, 35, 55]), 'g', linewidth=1.5, label='Medium')
ax0.plot(distance.universe, fuzz.trapmf(distance.universe, [45, 70, 100, 100]), 'r', linewidth=1.5, label='Far')
ax0.set_title('Distance to Obstacle (meters)')
ax0.legend()

ax1.plot(speed.universe, fuzz.trapmf(speed.universe, [0, 0, 5, 15]), 'b', linewidth=1.5, label='Low')
ax1.plot(speed.universe, fuzz.trimf(speed.universe, [10, 20, 30]), 'g', linewidth=1.5, label='Medium')
ax1.plot(speed.universe, fuzz.trapmf(speed.universe, [25, 30, 35, 35]), 'r', linewidth=1.5, label='High')
ax1.set_title('Car Speed (from dataset)')
ax1.legend()

ax2.plot(road.universe, fuzz.trapmf(road.universe, [0, 0, 0.1, 0.3]), 'b', linewidth=1.5, label='Icy')
ax2.plot(road.universe, fuzz.trimf(road.universe, [0.2, 0.5, 0.8]), 'g', linewidth=1.5, label='Wet')
ax2.plot(road.universe, fuzz.trapmf(road.universe, [0.7, 0.9, 1.0, 1.0]), 'r', linewidth=1.5, label='Good')
ax2.set_title('Road Condition Index (RCI)')
ax2.legend()

ax3.plot(action.universe, fuzz.trimf(action.universe, [-10, -8, -6]), 'b', linewidth=1.5, label='Hard Brake')
ax3.plot(action.universe, fuzz.trimf(action.universe, [-7, -4, 0]), 'g', linewidth=1.5, label='Light Brake')
ax3.plot(action.universe, fuzz.trimf(action.universe, [-2, 0, 2]), 'k', linewidth=1.5, label='Maintain Speed')
ax3.plot(action.universe, fuzz.trimf(action.universe, [0, 3, 6]), 'm', linewidth=1.5, label='Light Accelerate')
ax3.plot(action.universe, fuzz.trimf(action.universe, [5, 8, 10]), 'r', linewidth=1.5, label='Hard Accelerate')
ax3.set_title('Control Action (Brake / Maintain / Accelerate)')
ax3.legend()

for ax in [ax0, ax1, ax2, ax3]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

print("\n--- Тестування Симулятора ---")

print("Сценарій 1 (Небезпека):")
print(f"distance = {8}")
print(f"speed = {25}")
print(f"road_condition = {0.1}")
fis_simulator.input['distance'] = 8
fis_simulator.input['speed'] = 25
fis_simulator.input['road_condition'] = 0.1
fis_simulator.compute()
print(f"Вихідна дія = {fis_simulator.output['action']:.2f}")

print("Сценарій 2 (Круїз):")
print(f"distance = {80}")
print(f"speed = {18}")
print(f"road_condition = {1.0}")
fis_simulator.input['distance'] = 80
fis_simulator.input['speed'] = 18
fis_simulator.input['road_condition'] = 1.0
fis_simulator.compute()
print(f"Вихідна дія = {fis_simulator.output['action']:.2f}")

print("Сценарій 3 (Розгін):")
print(f"distance = {90}")
print(f"speed = {5}")
print(f"road_condition = {0.9}")
fis_simulator.input['distance'] = 90
fis_simulator.input['speed'] = 5
fis_simulator.input['road_condition'] = 0.9
fis_simulator.compute()
print(f"Вихідна дія = {fis_simulator.output['action']:.2f}")

x_dist = np.arange(distance.universe.min(), distance.universe.max(), 5)
y_speed = np.arange(speed.universe.min(), speed.universe.max(), 2)
X, Y = np.meshgrid(x_dist, y_speed)

road_conditions = [1.0, 0.5, 0.1]
titles = ['Road: Good (1.0)', 'Road: Wet (0.5)', 'Road: Icy (0.1)']
results_Z = []

for road_val in road_conditions:
    Z = np.zeros_like(X)
    for i, dist_val in enumerate(x_dist):
        for j, speed_val in enumerate(y_speed):
            fis_simulator.input['distance'] = dist_val
            fis_simulator.input['speed'] = speed_val
            fis_simulator.input['road_condition'] = road_val
            
            try:
                fis_simulator.compute()
                Z[j, i] = fis_simulator.output['action']
            except:
                Z[j, i] = 0
    results_Z.append(Z)

fig = plt.figure(figsize=(18, 8))

for i in range(3):
    ax = fig.add_subplot(1, 3, i + 1, projection='3d')
    Z = results_Z[i]
    
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis',
                           linewidth=0.4, antialiased=True)
    
    ax.set_title(titles[i])
    ax.set_xlabel('Distance (meters)')
    ax.set_ylabel('Speed')
    ax.set_zlabel('Action (Brake/Accel)')
    ax.view_init(30, 200)

plt.tight_layout()
plt.show()