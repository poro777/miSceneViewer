import time
# Dictionary to store total execution time for each function
total_execution_time = {}
total_execution_times = {}

def measure_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        #print(f"Execution time of {func.__name__}: {execution_time} seconds")

        # Update total execution time for the function
        if func.__name__ in total_execution_time:
            total_execution_time[func.__name__] += execution_time
            total_execution_times[func.__name__] += 1
        else:
            total_execution_time[func.__name__] = execution_time
            total_execution_times[func.__name__] = 1
        return result
    return wrapper

def clear_total_execution_time():
    print("Clear execution time...")
    total_execution_time.clear()
    total_execution_times.clear()

# Function to print total execution time for each function
def print_total_execution_time():
    print("\nExecution time:")
    total = 0
    for func_name, total_time in total_execution_time.items():
        times = total_execution_times[func_name]
        total += total_time
        print(f"{func_name}: average({times})= {round(total_time / times,6) },	total= {round(total_time,6)} seconds")

    print("\ntotal:", total)
