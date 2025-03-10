import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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
def train_pure_gd(model, device, train_loader, test_loader, threshold=0.95, max_epochs=20):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    total_iters = 0

    for epoch in range(max_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_iters += 1

            # Периодическая оценка модели
            if total_iters % 100 == 0:
                acc = evaluate(model, device, test_loader)
                print(f"Pure GD: итерация {total_iters}, точность {acc:.4f}")
                if acc >= threshold:
                    print(f"Pure GD достиг пороговой точности на итерации {total_iters}")
                    return total_iters
    print("Pure GD не достиг пороговой точности за заданное число эпох.")
    return total_iters

# Операции генетического алгоритма: кроссовер и мутация
def crossover(parent1, parent2):
    child = SimpleNet()
    with torch.no_grad():
        for param_c, param_p1, param_p2 in zip(child.parameters(), parent1.parameters(), parent2.parameters()):
            # Простой кроссовер – среднее значений параметров с добавлением небольшого шума
            param_c.copy_((param_p1 + param_p2) / 2 + 0.01 * torch.randn_like(param_p1))
    return child

def mutate(model, mutation_rate=0.1):
    with torch.no_grad():
        for param in model.parameters():
            if torch.rand(1).item() < mutation_rate:
                param.add_(0.01 * torch.randn_like(param))
    return model

# 2. Обучение с комбинированным подходом (градиентный спуск + генетические алгоритмы)
def train_combined(device, train_loader, test_loader, population_size=4, threshold=0.95, max_generations=25, gd_steps=50, max_iters=float('inf')):
    # Инициализация популяции моделей
    population = [SimpleNet().to(device) for _ in range(population_size)]
    total_iters = 0
    generation = 0
    best_acc = 0

    while generation < max_generations and best_acc < threshold:
        print(f"\nПоколение {generation + 1}")
        # Для каждой модели выполняем несколько итераций градиентного спуска
        for candidate in population:
            candidate.train()
            optimizer = optim.SGD(candidate.parameters(), lr=0.01)
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
                total_iters += 1

                if total_iters >= max_iters:
                    print("Комбинированный подход превысил максимальное число итераций, заданное для чистого градиентного спуска.")
                    return total_iters

        # Оценка всех кандидатов на тестовом наборе
        fitness = []
        for idx, candidate in enumerate(population):
            acc = evaluate(candidate, device, test_loader)
            fitness.append(acc)
            print(f"Кандидат {idx + 1}: точность {acc:.4f}")
        best_acc = max(fitness)
        if best_acc >= threshold:
            print(f"Комбинированный подход достиг пороговой точности на поколении {generation + 1} (общие итерации: {total_iters})")
            return total_iters

        # Отбор – выбор двух лучших моделей
        sorted_population = [model for _, model in sorted(zip(fitness, population), key=lambda pair: pair[0], reverse=True)]
        parents = sorted_population[:2]

        # Формирование нового поколения
        new_population = parents.copy()  # сохраняем родителей
        while len(new_population) < population_size:
            child = crossover(parents[0], parents[1])
            child = mutate(child)
            child = child.to(device)
            new_population.append(child)
        population = new_population
        generation += 1

    print("Комбинированный подход не достиг пороговой точности за заданное число поколений.")
    return total_iters

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
    iters_gd = train_pure_gd(model, device, train_loader, test_loader)

    print("\nНачало обучения с комбинированным подходом (градиентный спуск + генетические алгоритмы)...")
    iters_hybrid = train_combined(device, train_loader, test_loader, max_iters=iters_gd)

    print(f"\nОбщее число итераций для градиентного спуска: {iters_gd}")
    print(f"Общее число итераций для комбинированного подхода: {iters_hybrid}")

if __name__ == "__main__":
    main()