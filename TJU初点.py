import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, ConnectionPatch
import numpy as np

# 创建图形和坐标轴
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.set_aspect('equal')

# 设置背景色
fig.patch.set_facecolor('#f0f0f5')
ax.set_facecolor('#f0f0f5')
ax.axis('off')

# 颜色定义
light_blue = '#a6cee3'
dark_gray = '#333333'

# 1. 绘制左侧输入模块（运行拓扑和功率注入）
input_rect = FancyBboxPatch((0.5, 3), 1.2, 2, boxstyle="round,pad=0.1",
                           facecolor=light_blue, edgecolor=dark_gray, linewidth=1)
ax.add_patch(input_rect)

# 输入模块内的标签
ax.text(1.1, 4.7, '运行拓扑', ha='center', va='center', fontsize=10, color=dark_gray)
ax.text(1.1, 3.7, '功率注入', ha='center', va='center', fontsize=10, color=dark_gray)

# 分割线
ax.plot([0.5, 1.7], [4.2, 4.2], color=dark_gray, linewidth=0.5)

# 2. 绘制神经网络结构（多层感知机）
# 定义层参数
layers = [
    {'x': 2.5, 'neurons': 6, 'y_start': 2.5},  # 输入层
    {'x': 4.0, 'neurons': 5, 'y_start': 2.8},  # 隐藏层1
    {'x': 5.5, 'neurons': 4, 'y_start': 3.0},  # 隐藏层2
    {'x': 7.0, 'neurons': 3, 'y_start': 3.2},  # 隐藏层3
    {'x': 8.5, 'neurons': 2, 'y_start': 3.4}   # 输出层
]

# 绘制神经元和连接线
prev_neurons = []
for i, layer in enumerate(layers):
    neurons = []
    y_positions = np.linspace(layer['y_start'], layer['y_start'] + 2, layer['neurons'])
    
    for j, y in enumerate(y_positions):
        # 绘制神经元圆点
        neuron = Circle((layer['x'], y), radius=0.15, facecolor=light_blue, 
                       edgecolor=dark_gray, linewidth=1)
        ax.add_patch(neuron)
        neurons.append((layer['x'], y))
        
        # 绘制层间连接线
        if i > 0:
            for prev_x, prev_y in prev_neurons:
                ax.plot([prev_x, layer['x']], [prev_y, y], color=dark_gray, 
                       linewidth=0.3, alpha=0.7)
    
    prev_neurons = neurons

# 3. 绘制输出模块
output_rect = FancyBboxPatch((9.2, 3.7), 1.2, 0.6, boxstyle="round,pad=0.1",
                            facecolor=light_blue, edgecolor=dark_gray, linewidth=1)
ax.add_patch(output_rect)
ax.text(9.8, 4.0, '状态变量', ha='center', va='center', fontsize=10, color=dark_gray)

# 4. 绘制输入到神经网络的连接线
input_points = [(1.7, 4.7), (1.7, 3.7)]  # 运行拓扑和功率注入的输出点
first_layer_neurons = [(layers[0]['x'], y) for y in np.linspace(layers[0]['y_start'], 
                                                               layers[0]['y_start'] + 2, 
                                                               layers[0]['neurons'])]

for input_point in input_points:
    for neuron in first_layer_neurons:
        ax.plot([input_point[0], neuron[0]], [input_point[1], neuron[1]], 
               color=dark_gray, linewidth=0.3, alpha=0.7)

# 5. 绘制神经网络到输出的连接线
last_layer_neurons = [(layers[-1]['x'], y) for y in np.linspace(layers[-1]['y_start'], 
                                                               layers[-1]['y_start'] + 2, 
                                                               layers[-1]['neurons'])]
output_point = (9.2, 4.0)  # 状态变量模块的输入点

for neuron in last_layer_neurons:
    ax.plot([neuron[0], output_point[0]], [neuron[1], output_point[1]], 
           color=dark_gray, linewidth=0.3, alpha=0.7)

# 6. 添加标题
ax.text(5, 7.5, '2.构造式可行解生成器', ha='center', va='center', 
        fontsize=16, fontweight='bold', color=dark_gray)

# 7. 添加说明文本
description_text = """网络以功率注入与运行拓扑等条件为输入，输出潮流状态变量
在前向计算中显式嵌入潮流方程残差与各类安全约束函数
通过内点牵引、渐进式混合损失等策略确保严格可行解"""
ax.text(5, 1, description_text, ha='center', va='center', 
        fontsize=9, color=dark_gray, style='italic')

plt.tight_layout()
plt.savefig('constructive_feasible_solution_generator.png', dpi=300, bbox_inches='tight')
plt.show()