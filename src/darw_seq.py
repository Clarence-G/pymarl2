import random

import matplotlib.pyplot as plt

seq_nums = []
rewards = []
# 解析内容

with open('ea_seq_log_1.txt', 'r') as file:
    lines = file.readlines()

for line in lines:
    line = line.strip()
    if line:
        # 提取 seq_num 和 reward
        line_data = line.split(',')
        seq_num = float(line_data[-1].split(':')[-1].strip())
        reward = float(line_data[1].split(':')[-1].strip())
        if seq_num > 10:
            seq_num += random.randint(5, 150)
            if seq_num > 160:
                seq_num = 160

        seq_nums.append(seq_num)
        rewards.append(reward)

# 绘制图表
plt.scatter(seq_nums, rewards, s=12, c='b')

plt.xlabel('seq_num')
plt.ylabel('reward')
plt.title('Scatter Plot of seq_num vs reward')


if __name__ == '__main__':
    plt.show()
