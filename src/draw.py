import oapackage

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    with open('eval_result.txt', 'r') as file:
        lines = file.readlines()

    state_set_values = []
    avg_reward_values = []

    pareto = oapackage.ParetoDoubleLong()

    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('len(state_set)'):
            parts = line.split(', ')
            state_set = int(parts[0].split(': ')[1])
            avg_reward = float(parts[1].split(': ')[1])
            winning_rate = float(parts[2].split(': ')[1])
            state_set_values.append(state_set)
            avg_reward_values.append(avg_reward)

            w = oapackage.doubleVector((state_set, avg_reward))
            pareto.addvalue(w, i)

    pareto.show(verbose=1)
    for x in pareto.allvalues():
        print(x)
    lst = pareto.allindices()
    states_opt = []
    rewards_opt = []
    for x in lst:
        states_opt.append(state_set_values[x])
        rewards_opt.append(avg_reward_values[x])



    # Plot state_set vs avg_reward as a scatter plot
    plt.figure()

    # label with  label='Non Pareto-optimal'
    plt.scatter(state_set_values, avg_reward_values, s=12, c='b', label='Non Pareto-optimal')

    # label with  label='Pareto optimal'
    plt.scatter(states_opt, rewards_opt, s=12, c='r', label='Pareto optimal')
    plt.xlabel("States")
    plt.ylabel('Reward')
    plt.title('States vs Reward')
    plt.show()
