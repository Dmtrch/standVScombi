import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time  # Добавлен импорт модуля time
import random

# Определение простой модели для MNIST
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Функция для оценки модели на тестовом наборе
def evaluate(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += len(target)
    return correct / total

# 1. Обучение с использованием только градиентного спуска
def train_pure_gd(model, device, train_loader, test_loader, threshold=0.92, max_epochs=20):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    start_time = time.time()

    # Считаем общее количество обработанных батчей, для периодической оценки
    batch_counter = 0

    for epoch in range(max_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            batch_counter += 1

            # Периодическая оценка модели каждые 100 батчей
            if batch_counter % 100 == 0:
                acc = evaluate(model, device, test_loader)
                elapsed_time = time.time() - start_time
                print(f"Pure GD: время {elapsed_time:.2f} сек, точность {acc:.4f}")
                if acc >= threshold:
                    print(f"Pure GD достиг пороговой точности за {elapsed_time:.2f} сек")
                    return elapsed_time
    elapsed_time = time.time() - start_time
    print("Pure GD не достиг пороговой точности за заданное число эпох.")
    return elapsed_time

# Операции генетического алгоритма: кроссовер

def crossover(parent1, parent2, noise_factor=0.01):
    child = SimpleNet()
    with torch.no_grad():
        for param_c, param_p1, param_p2 in zip(child.parameters(), parent1.parameters(), parent2.parameters()):
            # Простой кроссовер – среднее значений параметров с добавлением адаптивного шума
            param_c.copy_((param_p1 + param_p2) / 2 + noise_factor * torch.randn_like(param_p1))
    return child

def mutate(model, mutation_rate=0.1):
    with torch.no_grad():
        for param in model.parameters():
            if torch.rand(1).item() < mutation_rate:
                param.add_(0.01 * torch.randn_like(param))
    return model

def tournament_selection(population, fitness, tournament_size=2):
    selected_indices = random.sample(range(len(population)), tournament_size)
    best_index = max(selected_indices, key=lambda i: fitness[i])
    return population[best_index]

# 2. Обучение с комбинированным подходом (градиентный спуск + генетические алгоритмы)
def train_combined(device, train_loader, test_loader, population_size=4, threshold=0.92, max_generations=25, gd_steps=50, max_time=float('inf')):
    # Инициализация популяции моделей
    population = [SimpleNet().to(device) for _ in range(population_size)]
    start_time = time.time()
    generation = 0
    best_acc = 0

    while generation < max_generations and best_acc < threshold:
        print(f"\nПоколение {generation + 1}")
        
        current_mutation_rate = max(0.01, 0.1 * (0.95 ** generation))
        current_noise_factor = max(0.005, 0.01 * (0.95 ** generation))

        # Для каждой модели выполняем несколько итераций градиентного спуска
        for candidate in population:
            candidate.train()
            optimizer = optim.Adam(candidate.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            steps = 0
            for data, target in train_loader:
                if steps >= gd_steps:
                    break
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = candidate(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                steps += 1

                elapsed_time = time.time() - start_time
                if elapsed_time >= max_time:
                    print("Комбинированный подход превысил максимальное время, заданное для чистого градиентного спуска.")
                    return elapsed_time

        # Оценка всех кандидатов на тестовом наборе
        fitness = []
        for idx, candidate in enumerate(population):
            acc = evaluate(candidate, device, test_loader)
            fitness.append(acc)
            print(f"Кандидат {idx + 1}: точность {acc:.4f}")
        best_acc = max(fitness)
        if best_acc >= threshold:
            elapsed_time = time.time() - start_time
            print(f"Комбинированный подход достиг пороговой точности на поколении {generation + 1} (время: {elapsed_time:.2f} сек)")
            return elapsed_time

        # Отбор – турнирный отбор двух родителей
        parents = []
        while len(parents) < 2:
            candidate = tournament_selection(population, fitness, tournament_size=2)
            if candidate not in parents:
                parents.append(candidate)

        # Формирование нового поколения с использованием адаптивных параметров кроссовера и мутации
        new_population = parents.copy()  # сохраняем родителей
        while len(new_population) < population_size:
            child = crossover(parents[0], parents[1], noise_factor=current_noise_factor)
            child = mutate(child, mutation_rate=current_mutation_rate)
            child = child.to(device)
            new_population.append(child)
        population = new_population
        generation += 1

    elapsed_time = time.time() - start_time
    print("Комбинированный подход не достиг пороговой точности за заданное число поколений.")
    return elapsed_time

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor()])
    print(device)

    # Загрузка MNIST (обучающая и тестовая выборки)
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    print("Начало обучения с использованием чистого градиентного спуска...")
    model = SimpleNet().to(device)
    gd_time = train_pure_gd(model, device, train_loader, test_loader)

    print("\nНачало обучения с комбинированным подходом (градиентный спуск + генетические алгоритмы)...")
    hybrid_time = train_combined(device, train_loader, test_loader, max_time=gd_time)

    print(f"\nВремя обучения для градиентного спуска: {gd_time:.2f} сек")
    print(f"Время обучения для комбинированного подхода: {hybrid_time:.2f} сек")

if __name__ == "__main__":
    main()