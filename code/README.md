# Simulação de Difusão de Desinformação

Este código implementa o modelo de difusão de desinformação em redes sociais descrito no paper, com visualizações interativas.

## Instalação

```bash
pip install -r requirements.txt
```

## Uso Básico

Execute a simulação com os parâmetros padrão:

```bash
python3 simulation.py
```

Isso irá:
1. Criar uma rede social com 20 agentes
2. Executar 10 rounds de simulação
3. Gerar visualizações da rede e métricas

## Modificando Parâmetros

### Opção 1: Editar diretamente no código

Abra `simulation.py` e modifique o dicionário `config` na função `main()`:

```python
config = {
    'num_agents': 20,           # Número de agentes (comece pequeno!)
    'prob_in': 0.6,             # Probabilidade de conexão dentro do grupo
    'prob_out': 0.1,            # Probabilidade de conexão entre grupos
    'num_groups': 2,            # Número de grupos na rede
    'message_left_bias': 2.0,   # Parâmetro alpha para bias de mensagens
    'message_right_bias': 2.0,  # Parâmetro beta para bias de mensagens
    'prob_truth': 0.5,          # Probabilidade de mensagem ser verdadeira
    'agent_left_bias': 2.0,     # Parâmetro alpha para bias dos agentes
    'agent_right_bias': 2.0,    # Parâmetro beta para bias dos agentes
    'ave_reputation': 0.0,      # Média da reputação inicial
    'variance_reputation': 1.0,  # Variância da reputação inicial
    'bias_strength': 0.3,       # Peso do alinhamento ideológico (omega)
    'reputation_reward_strength': 0.5,  # Ganho de reputação por verdade (gamma)
    'reputation_penalty_strength': 0.5,  # Penalidade por desinformação (delta)
    'forwarding_cost': 0.1,     # Custo de encaminhar (k)
    'num_rounds': 10,           # Número de rounds
    'initial_senders': [0],     # IDs dos agentes que recebem mensagem inicial
    'seed': 42                  # Seed para reprodutibilidade
}
```

### Opção 2: Usar arquivo JSON

Edite `config.json` e execute:

```python
from simulation import Simulation
sim = Simulation(config_file='config.json')
sim.run()
sim.visualize_network(round_result=sim.run_round(0))
sim.plot_metrics()
```

## Parâmetros Importantes

### Rede
- **num_agents**: Número de agentes na rede (recomendado: 10-50 para começar)
- **prob_in**: Probabilidade de conexão dentro do mesmo grupo (0-1)
- **prob_out**: Probabilidade de conexão entre grupos diferentes (0-1, deve ser < prob_in)
- **num_groups**: Número de grupos na rede (recomendado: 2-4)

### Mensagens
- **prob_truth**: Probabilidade de uma mensagem ser verdadeira (0-1)
- **message_left_bias / message_right_bias**: Parâmetros da distribuição Beta para o viés ideológico das mensagens

### Agentes
- **bias_strength (ω)**: Quanto os agentes valorizam alinhamento ideológico
- **reputation_reward_strength (γ)**: Ganho de reputação ao encaminhar mensagens verdadeiras
- **reputation_penalty_strength (δ)**: Penalidade de reputação ao encaminhar mensagens falsas
- **forwarding_cost (k)**: Custo fixo de encaminhar uma mensagem

### Simulação
- **num_rounds**: Número de rounds a executar
- **initial_senders**: Lista de IDs dos agentes que recebem a mensagem inicialmente

## Visualizações

A simulação gera dois arquivos:

1. **network_diffusion.png**: Mostra a estrutura da rede e a difusão da mensagem no último round
   - Cores diferentes indicam grupos
   - Vermelho = encaminhou a mensagem
   - Laranja = recebeu mas não encaminhou
   - Cinza = não recebeu

2. **metrics.png**: Gráficos das métricas ao longo do tempo
   - Alcance da mensagem
   - Taxa de encaminhamento
   - Contaminação por desinformação
   - Reputação média

## Estrutura do Código

- `simulation.py`: Arquivo principal com a classe Simulation
- `snlearn/agent.py`: Implementação dos agentes
- `snlearn/message.py`: Implementação das mensagens
- `snlearn/socialnetwork.py`: Geração e gerenciamento da rede social
- `config.json`: Arquivo de configuração opcional

## Exemplo: Rede Pequena e Simples

Para começar com uma rede muito simples:

```python
config = {
    'num_agents': 10,
    'prob_in': 0.8,
    'prob_out': 0.2,
    'num_groups': 2,
    'num_rounds': 5,
    'initial_senders': [0, 1],
    'seed': 42
}
```

Isso cria uma rede pequena com 10 agentes, 2 grupos bem conectados, e executa apenas 5 rounds começando com 2 agentes iniciais.

