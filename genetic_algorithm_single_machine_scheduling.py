import random
import numpy as np

# ============================== 基础函数 ==============================

# 新适应度函数（基于任务延迟）
def fitness_function(schedule, processing_times, due_dates):
    current_time = 0
    total_delay = 0
    for task in schedule:
        current_time += processing_times[task]
        delay = max(0, current_time - due_dates[task])
        total_delay += delay
    return 1 / (total_delay + 1)  # +1 防止除0

# 获取总延迟
def get_total_delay(schedule, processing_times, due_dates):
    current_time = 0
    total_delay = 0
    for task in schedule:
        current_time += processing_times[task]
        delay = max(0, current_time - due_dates[task])
        total_delay += delay
    return total_delay

# 初始化种群
def initialize_population(tasks, pop_size):
    population = []
    for _ in range(pop_size):
        population.append(random.sample(tasks, len(tasks)))
    return population

# 选择操作
def selection(population, fitness_values):
    total_fitness = sum(fitness_values)
    probabilities = [fitness / total_fitness for fitness in fitness_values]
    selected_indices = np.random.choice(len(population), size=2, p=probabilities)
    return [population[i] for i in selected_indices], selected_indices
# 交叉操作
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + [item for item in parent2 if item not in parent1[:point]]
    child2 = parent2[:point] + [item for item in parent1 if item not in parent2[:point]]
    return child1, child2, point
# 变异操作
def mutation(child, mutation_rate):
    mutated = False
    child_copy = child.copy()
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(child)), 2)
        child_copy[idx1], child_copy[idx2] = child_copy[idx2], child_copy[idx1]
        mutated = True
    return child_copy, mutated, (idx1, idx2) if mutated else None

# 打印最优解的演化过程
def track_evolution(best_schedule_history, processing_times, due_dates):
    print("\n============= 最优解演化历程 =============")
    prev_fitness = 0

    for i, entry in enumerate(best_schedule_history):
        generation = entry["generation"]
        schedule = entry["schedule"]
        fitness = entry["fitness"]
        total_delay = get_total_delay(schedule, processing_times, due_dates)

        improvement = ""
        if i > 0:
            improvement = f"改进率: {((fitness - prev_fitness) / prev_fitness * 100):.2f}%"

        prev_fitness = fitness

        print(f"\n第 {generation} 代:")
        print(f"调度顺序: {schedule}")
        print(f"总延迟时间: {total_delay}")
        print(f"适应度: {fitness:.8f}")
        if improvement:
            print(improvement)
        if "origin" in entry:
            print(f"产生方式: {entry['origin']}")
        if "parents" in entry:
            print(f"父代索引: {entry['parents']}")
        if "crossover_point" in entry:
            print(f"交叉点: {entry['crossover_point']}")
        if "mutation" in entry and entry["mutation"]:
            print(f"变异位置: {entry['mutation_indices']}")


# ============================== 主算法 ==============================

def genetic_algorithm(tasks, processing_times, due_dates, pop_size, generations, mutation_rate):
    population = initialize_population(tasks, pop_size)
    print(f"初始种群: {population}")

    best_schedule_history = []
    global_best_schedule = None
    global_best_fitness = 0

    for generation in range(generations):
        fitness_values = [fitness_function(s, processing_times, due_dates) for s in population]

        current_best_idx = np.argmax(fitness_values)
        current_best_schedule = population[current_best_idx]
        current_best_fitness = fitness_values[current_best_idx]

        if global_best_schedule is None or current_best_fitness > global_best_fitness:
            global_best_schedule = current_best_schedule.copy()
            global_best_fitness = current_best_fitness

            history_entry = {
                "generation": generation + 1,
                "schedule": global_best_schedule.copy(),
                "fitness": global_best_fitness,
                "origin": "初始种群" if generation == 0 else "当代最优"
            }
            best_schedule_history.append(history_entry)

        if generation % 5 == 0:
            print(f"Generation {generation + 1}: Current best = {current_best_schedule}, Fitness = {current_best_fitness:.8f}")
            print(f"Global best so far: {global_best_schedule}, Global best fitness: {global_best_fitness:.8f}")

        new_population = []
        while len(new_population) < pop_size:
            parents, parent_indices = selection(population, fitness_values)
            child1, child2, crossover_point = crossover(*parents)

            child1_after_mutation, child1_mutated, child1_mutation_indices = mutation(child1, mutation_rate)
            child2_after_mutation, child2_mutated, child2_mutation_indices = mutation(child2, mutation_rate)

            child1_fitness = fitness_function(child1_after_mutation, processing_times, due_dates)
            child2_fitness = fitness_function(child2_after_mutation, processing_times, due_dates)

            for child, fitness, mutated, mutation_indices in [
                (child1_after_mutation, child1_fitness, child1_mutated, child1_mutation_indices),
                (child2_after_mutation, child2_fitness, child2_mutated, child2_mutation_indices)
            ]:
                if fitness > global_best_fitness:
                    global_best_schedule = child.copy()
                    global_best_fitness = fitness
                    entry = {
                        "generation": generation + 1,
                        "schedule": global_best_schedule.copy(),
                        "fitness": global_best_fitness,
                        "origin": "交叉+变异" if mutated else "交叉",
                        "parents": parent_indices,
                        "crossover_point": crossover_point
                    }
                    if mutated:
                        entry["mutation"] = True
                        entry["mutation_indices"] = mutation_indices
                    best_schedule_history.append(entry)

            new_population.extend([child1_after_mutation, child2_after_mutation])

        population = new_population[:pop_size]

    track_evolution(best_schedule_history, processing_times, due_dates)

    return global_best_schedule, global_best_fitness


# ============================== 简单示例运行 ==============================

tasks = list(range(5))
processing_times = [3, 5, 2, 4, 1]
due_dates = [5, 7, 4, 10, 6]

pop_size = 20
generations = 100
mutation_rate = 0.1

random.seed(42)
np.random.seed(42)

best_schedule, max_fitness = genetic_algorithm(tasks, processing_times, due_dates,
                                               pop_size, generations, mutation_rate)

print("\nFinal Results:")
print(f"Best Schedule Found Overall: {best_schedule}, Max Fitness: {max_fitness:.8f}")


# 显示最终调度的细节
def display_schedule_details(schedule, processing_times, due_dates):
    current_time = 0
    print("\n详细调度信息:")
    print("Order | Task | Processing | Due | Completion | Delay")
    print("-" * 60)
    for i, task in enumerate(schedule):
        current_time += processing_times[task]
        delay = max(0, current_time - due_dates[task])
        print(f"{i+1:5d} | {task:4d} | {processing_times[task]:10d} | {due_dates[task]:3d} | {current_time:10d} | {delay:5d}")
    total_delay = get_total_delay(schedule, processing_times, due_dates)
    print("-" * 60)
    print(f"总延迟时间: {total_delay}")

display_schedule_details(best_schedule, processing_times, due_dates)
