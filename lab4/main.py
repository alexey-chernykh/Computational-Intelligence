import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance

class LogisticPSO:
    def __init__(self, num_particles, num_points, dist_matrix, w=0.5, c1=1.5, c2=1.5):
        self.num_particles = num_particles
        self.num_points = num_points
        self.dist_matrix = dist_matrix

        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.positions = np.random.uniform(-1, 1, (num_particles, num_points))
        self.velocities = np.zeros((num_particles, num_points))

        self.pbest_pos = self.positions.copy()
        self.pbest_scores = np.full(num_particles, float('inf'))

        self.gbest_pos = None
        self.gbest_score = float('inf')
        
        self.history = []

    def _get_route(self, particle_position):
        clients_order = np.argsort(particle_position[1:]) + 1 

        return np.concatenate(([0], clients_order, [0]))

    def _calculate_cost(self, position):
        """Фітнес-функція: рахує загальну довжину маршруту"""
        route = self._get_route(position)
        total_dist = 0
        for i in range(len(route) - 1):
            total_dist += self.dist_matrix[route[i], route[i+1]]
        return total_dist

    def optimize(self, iterations=100):
        print(f"Починаємо оптимізацію на {iterations} ітерацій...")
            
        for it in range(iterations):
            for i in range(self.num_particles):
                cost = self._calculate_cost(self.positions[i])

                if cost < self.pbest_scores[i]:
                    self.pbest_scores[i] = cost
                    self.pbest_pos[i] = self.positions[i].copy()

                    if cost < self.gbest_score:
                        self.gbest_score = cost
                        self.gbest_pos = self.positions[i].copy()

            self.history.append(self.gbest_score)

            r1 = np.random.rand(self.num_particles, self.num_points)
            r2 = np.random.rand(self.num_particles, self.num_points)

            self.velocities = (self.w * self.velocities + 
                            self.c1 * r1 * (self.pbest_pos - self.positions) + 
                            self.c2 * r2 * (self.gbest_pos - self.positions))
                
            self.positions = self.positions + self.velocities

            if (it + 1) % 20 == 0:
                print(f"Ітерація {it+1}/{iterations}, Поточний мінімум: {self.gbest_score:.2f} км")
                
        return self._get_route(self.gbest_pos), self.gbest_score


np.random.seed(42) 
NUM_POINTS = 20   
MAP_SIZE = 100

print("Генеруємо дані...")
coordinates = np.random.rand(NUM_POINTS, 2) * MAP_SIZE

df = pd.DataFrame(coordinates, columns=['x', 'y'])
df['type'] = ['Depot'] + ['Client'] * (NUM_POINTS - 1)

dist_matrix = distance.cdist(coordinates, coordinates, 'euclidean')

pso = LogisticPSO(num_particles=100, num_points=NUM_POINTS, dist_matrix=dist_matrix)
best_route, best_cost = pso.optimize(iterations=150)

print("-" * 30)
print(f"Оптимізацію завершено.")
print(f"Найкращий маршрут (порядок точок): {best_route}")
print(f"Загальна відстань: {best_cost:.2f} км")
print("-" * 30)

plt.figure(figsize=(10, 5))
plt.plot(pso.history, label='Загальна відстань маршруту', color='blue')
plt.title('Конвергенція алгоритму PSO')
plt.xlabel('Ітерація')
plt.ylabel('Відстань')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10, 8))

for i in range(len(best_route) - 1):
    start_node = best_route[i]
    end_node = best_route[i+1]
    p1 = coordinates[start_node]
    p2 = coordinates[end_node]
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', alpha=0.6, linewidth=1) 

plt.scatter(df.iloc[1:]['x'], df.iloc[1:]['y'], c='red', s=50, label='Клієнти')
plt.scatter(df.iloc[0]['x'], df.iloc[0]['y'], c='green', s=200, marker='*', label='Склад')

for i, node_idx in enumerate(best_route[:-1]):
    plt.text(coordinates[node_idx][0] + 1, coordinates[node_idx][1] + 1, 
             str(node_idx), fontsize=9, color='darkblue')

plt.title(f'Оптимізований логістичний маршрут\nДовжина: {best_cost:.2f} км')
plt.legend()
plt.grid(True)
plt.show()