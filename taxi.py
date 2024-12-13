import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

# Criação do ambiente Taxi-v3
env = gym.make('Taxi-v3', render_mode="human")  # Usar render_mode="human" para visualização gráfica

# Inicialização da tabela Q
state_size = env.observation_space.n
action_size = env.action_space.n
q_table = np.zeros((state_size, action_size))

# Hiperparâmetros
alpha = 0.1  # Taxa de aprendizado
gamma = 0.99  # Fator de desconto
epsilon = 1.0  # Probabilidade de exploração
epsilon_min = 0.01
epsilon_decay = 0.995
num_episodes = 5000
max_steps = 100

# Armazenar recompensas por episódio
rewards_all_episodes = []

# Configuração do modo interativo do matplotlib
plt.ion()
fig, ax = plt.subplots()
ax.set_xlabel('Episódios')
ax.set_ylabel('Recompensa Acumulada')
ax.set_title('Recompensas ao longo dos episódios')
line, = ax.plot([], [], color='blue')

# Função de escolha de ação (ε-greedy)
def choose_action(state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Explorar
    else:
        return np.argmax(q_table[state, :])  # Explorar a melhor ação

# Treinamento com Q-Learning
for episode in range(num_episodes):
    state, _ = env.reset()  # Reset retorna uma tupla na versão mais recente
    total_rewards = 0
    done = False

    for _ in range(max_steps):
        action = choose_action(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)

        # Atualização da tabela Q
        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action]
        )

        state = next_state
        total_rewards += reward

        if done:
            break

    # Reduzir epsilon (exploração decresce com o tempo)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    rewards_all_episodes.append(total_rewards)

    # Atualizar gráfico em tempo real
    line.set_xdata(range(len(rewards_all_episodes)))
    line.set_ydata(rewards_all_episodes)
    ax.relim()
    ax.autoscale_view()
    plt.draw()
    plt.pause(0.001)

print("Treinamento concluído!")
print("Tabela Q final:")
print(q_table)

# Finalizar gráfico interativo
plt.ioff()
plt.show()

# Teste do agente treinado
state, _ = env.reset()
done = False

print("\nTestando o agente treinado...\n")
for step in range(max_steps):
    action = np.argmax(q_table[state, :])  # Ação ótima
    next_state, reward, done, _, _ = env.step(action)
    
    time.sleep(0.5)  # Pausa para visualização mais lenta
    state = next_state

    if done:
        print("\nPassageiro entregue com sucesso!")
        break

env.close()