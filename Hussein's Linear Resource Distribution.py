import psutil
import numpy as np
from typing import List, Tuple
from scipy.optimize import linprog
#author @HusseinAkugizibwe
#2019/BSE/004/PS

def get_resource_limits() -> Tuple[float, int, int]:
    """Get resource limits based on available system resources."""
    target_time = float(input("Enter target time in seconds: "))
    available_memory = psutil.virtual_memory().available
    available_cpu = psutil.cpu_count()
    target_memory = int(float(input(f"Enter target memory in GB (available memory: {available_memory / (1024 ** 3):.2f} GB): ")) * 1024 ** 3)

    return target_time, target_memory, available_cpu

def generate_tasks(num_tasks: int, num_vars: int, num_constraints: int) -> List[Tuple[List[float], List[List[float]], List[float]]]:
    """Generate tasks with random coefficients and constraints."""
    tasks = []
    for i in range(num_tasks):
        # Generate random coefficients and constraints
        c = np.random.rand(num_vars)
        A_ub = np.random.rand(num_constraints, num_vars)
        b_ub = np.random.rand(num_constraints)

        tasks.append((c, A_ub, b_ub))

    return tasks

def optimize_resources(tasks: List[Tuple[List[float], List[List[float]], List[float]]], target_time: float, target_memory: int, available_cpu: int) -> List[Tuple[int, int, float]]:
    """Optimize resources for multiple tasks."""
    resource_allocations = []
    for i, task in enumerate(tasks):
        c = task[0]
        A_ub = task[1]
        b_ub = task[2]

        # Set bounds for variables
        bounds = [(0, available_cpu), (0, target_memory), (0, target_time)]

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
        t, m, c = res.x

        resource_allocations.append((int(c), int(m), t))

    return resource_allocations

if __name__ == '__main__':
    # Get resource limits
    T, M, C = get_resource_limits()

    # Generate random tasks
    num_tasks = int(input("Enter number of tasks: "))
    num_vars = int(input("Enter number of variables: "))
    num_constraints = int(input("Enter number of constraints: "))
    tasks = generate_tasks(num_tasks, num_vars, num_constraints)

    # Optimize resources for tasks
    resource_allocations = optimize_resources(tasks, T, M, C)

    # Output optimal resource allocations for each task
    for i, task in enumerate(resource_allocations):
        c, m, t = task
        print(f"\nTask {i+1}: Allocated {c} CPU cores and {m / (1024 ** 3):.2f} GB of memory, with an execution time of {t:.3f} seconds.")
