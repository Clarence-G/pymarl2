import matplotlib.pyplot as plt

# Read data from file
with open("mutation_train.txt", "r") as file:
    lines = file.readlines()

state_set_values = []
avg_reward_values = []

# Extract "states" and "reward" values from each line
for line in lines:
    parts = line.strip().split(", ")
    state_set = int(parts[2].split(": ")[1])
    avg_reward = float(parts[1].split(": ")[1])
    state_set_values.append(state_set)
    avg_reward_values.append(avg_reward)

# Plot state_set vs avg_reward as a scatter plot
plt.scatter(state_set_values, avg_reward_values, s=12, c='b')
plt.xlabel("States")
plt.ylabel("Reward")
plt.title("States vs Reward")

if __name__ == '__main__':
    plt.show()

