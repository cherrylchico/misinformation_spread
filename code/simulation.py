"""
Simulação de difusão de desinformação em redes sociais
Baseado no modelo descrito nos arquivos .tex
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
from snlearn.socialnetwork import SocialNetwork
from snlearn.agent import Agent
from snlearn.message import Message
from typing import List, Dict
import json

class Simulation:
    def __init__(self, config_file=None, **kwargs):
        """
        Inicializa a simulação
        
        Args:
            config_file: caminho para arquivo JSON com configurações
            **kwargs: parâmetros de configuração (sobrescreve config_file)
        """
        if config_file:
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = kwargs if kwargs else self._default_config()
        
        # Parâmetros da rede
        self.num_agents = self.config.get('num_agents', 20)
        self.prob_in = self.config.get('prob_in', 0.6)
        self.prob_out = self.config.get('prob_out', 0.1)
        self.num_groups = self.config.get('num_groups', 2)
        
        # Parâmetros de mensagens
        self.message_left_bias = self.config.get('message_left_bias', 2.0)
        self.message_right_bias = self.config.get('message_right_bias', 2.0)
        self.prob_truth = self.config.get('prob_truth', 0.5)
        
        # Parâmetros dos agentes
        self.agent_left_bias = self.config.get('agent_left_bias', 2.0)
        self.agent_right_bias = self.config.get('agent_right_bias', 2.0)
        self.ave_reputation = self.config.get('ave_reputation', 0.0)
        self.variance_reputation = self.config.get('variance_reputation', 1.0)
        self.bias_strength = self.config.get('bias_strength', 0.3)
        self.reputation_reward_strength = self.config.get('reputation_reward_strength', 0.5)
        self.reputation_penalty_strength = self.config.get('reputation_penalty_strength', 0.5)
        self.forwarding_cost = self.config.get('forwarding_cost', 0.1)
        
        # Parâmetros da simulação
        self.num_rounds = self.config.get('num_rounds', 10)
        self.initial_senders = self.config.get('initial_senders', [0])  # IDs dos agentes que recebem a mensagem inicialmente
        
        # Inicializa componentes
        self.network = SocialNetwork(
            self.num_agents, 
            self.prob_in, 
            self.prob_out, 
            self.num_groups,
            seed=self.config.get('seed', None)
        )
        
        self.agents = self._create_agents()
        self.messages = []
        self.history = {
            'reach': [],
            'forwarding_rate': [],
            'misinformation_contamination': [],
            'reputation_by_type': []
        }
        
    def _default_config(self):
        """Configuração padrão para uma rede pequena e fácil"""
        return {
            'num_agents': 20,
            'prob_in': 0.6,
            'prob_out': 0.1,
            'num_groups': 2,
            'message_left_bias': 2.0,
            'message_right_bias': 2.0,
            'prob_truth': 0.5,
            'agent_left_bias': 2.0,
            'agent_right_bias': 2.0,
            'ave_reputation': 0.0,
            'variance_reputation': 1.0,
            'bias_strength': 0.3,
            'reputation_reward_strength': 0.5,
            'reputation_penalty_strength': 0.5,
            'forwarding_cost': 0.1,
            'num_rounds': 10,
            'initial_senders': [0],
            'seed': 42
        }
    
    def _create_agents(self):
        """Cria todos os agentes da simulação"""
        agents = []
        for i in range(self.num_agents):
            agent = Agent(
                left_bias=self.agent_left_bias,
                right_bias=self.agent_right_bias,
                ave_reputation=self.ave_reputation,
                variance_reputation=self.variance_reputation,
                bias_strength=self.bias_strength,
                reputation_reward_strength=self.reputation_reward_strength,
                reputation_penalty_strength=self.reputation_penalty_strength,
                forwarding_cost=self.forwarding_cost
            )
            agents.append(agent)
        return agents
    
    def run_round(self, round_num):
        """Executa um round da simulação"""
        # 1. Gera uma nova mensagem
        message = Message(
            left_bias=self.message_left_bias,
            right_bias=self.message_right_bias,
            prob_truth=self.prob_truth
        )
        self.messages.append(message)
        
        # 2. Inicializa quem recebeu a mensagem
        received = set(self.initial_senders)
        received_from = {agent_id: [] for agent_id in self.initial_senders}
        
        # 3. Processa difusão em camadas (BFS)
        # Primeiro, os agentes iniciais decidem se vão encaminhar
        for sender_id in self.initial_senders:
            sender_reputations = [self.agents[sender_id].reputation]
            self.agents[sender_id].average_utility(message, sender_reputations, store=True)
            self.agents[sender_id].decide_action(store=True)
        
        current_layer = [sid for sid in self.initial_senders 
                        if self.agents[sid].current_action == 1]
        processed = set(self.initial_senders)
        
        while current_layer:
            next_layer = []
            
            for sender_id in current_layer:
                # Se decidiu encaminhar, envia para vizinhos
                neighbors = self.network.get_neighbors(sender_id)
                for neighbor_id in neighbors:
                    if neighbor_id not in received:
                        received.add(neighbor_id)
                        received_from[neighbor_id] = []
                    if sender_id not in received_from[neighbor_id]:
                        received_from[neighbor_id].append(sender_id)
                    if neighbor_id not in processed:
                        next_layer.append(neighbor_id)
                        processed.add(neighbor_id)
            
            # Agentes na próxima camada decidem se vão encaminhar
            current_layer = []
            for agent_id in next_layer:
                sender_reps = [self.agents[sid].reputation for sid in received_from[agent_id]]
                self.agents[agent_id].average_utility(message, sender_reps, store=True)
                self.agents[agent_id].decide_action(store=True)
                if self.agents[agent_id].current_action == 1:
                    current_layer.append(agent_id)
        
        # 4. (Já processado durante a difusão BFS)
        
        # 5. Revela verdade e atualiza reputações
        message.reveal_truth()
        for agent_id in received:
            self.agents[agent_id].update_reputation(message, store=True)
        
        # 6. Calcula métricas
        reach = len(received) / self.num_agents
        forwarding_count = sum(1 for agent_id in received if self.agents[agent_id].current_action == 1)
        forwarding_rate = forwarding_count / len(received) if received else 0
        
        false_forward_count = sum(
            1 for agent_id in received 
            if self.agents[agent_id].current_action == 1 and message.truth == 0
        )
        misinformation_contamination = false_forward_count / len(received) if received else 0
        
        self.history['reach'].append(reach)
        self.history['forwarding_rate'].append(forwarding_rate)
        self.history['misinformation_contamination'].append(misinformation_contamination)
        
        return {
            'message': message,
            'received': received,
            'forwarded': [aid for aid in received if self.agents[aid].current_action == 1],
            'reach': reach,
            'forwarding_rate': forwarding_rate,
            'misinformation_contamination': misinformation_contamination
        }
    
    def run(self):
        """Executa toda a simulação"""
        print(f"Iniciando simulação com {self.num_agents} agentes e {self.num_rounds} rounds...")
        results = []
        for round_num in range(self.num_rounds):
            result = self.run_round(round_num)
            results.append(result)
            print(f"Round {round_num + 1}: Reach={result['reach']:.2%}, "
                  f"Forwarding={result['forwarding_rate']:.2%}, "
                  f"Misinfo={result['misinformation_contamination']:.2%}, "
                  f"Truth={result['message'].truth}")
        return results
    
    def visualize_network(self, round_result=None, save_path=None):
        """Visualiza a rede e a difusão de mensagens"""
        G = self.network.graph
        pos = nx.spring_layout(G, seed=42, k=1, iterations=50)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Subplot 1: Rede com cores por grupo
        ax1 = axes[0]
        colors = plt.cm.Set3(np.linspace(0, 1, self.num_groups))
        node_colors = [colors[self.network.group_assignments[i] % len(colors)] 
                      for i in range(self.num_agents)]
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=300, ax=ax1, alpha=0.8)
        nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True, 
                              arrowsize=10, ax=ax1, edge_color='gray')
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax1)
        ax1.set_title('Estrutura da Rede Social\n(Cores indicam grupos)', fontsize=12)
        ax1.axis('off')
        
        # Subplot 2: Difusão de mensagem (se round_result fornecido)
        ax2 = axes[1]
        if round_result:
            message = round_result['message']
            received = round_result['received']
            forwarded = round_result['forwarded']
            
            # Cores dos nós
            node_colors_diff = []
            for i in range(self.num_agents):
                if i in forwarded:
                    node_colors_diff.append('red')  # Encaminhou
                elif i in received:
                    node_colors_diff.append('orange')  # Recebeu mas não encaminhou
                else:
                    node_colors_diff.append('lightgray')  # Não recebeu
            
            nx.draw_networkx_nodes(G, pos, node_color=node_colors_diff, 
                                  node_size=300, ax=ax2, alpha=0.8)
            nx.draw_networkx_edges(G, pos, alpha=0.2, arrows=True, 
                                  arrowsize=10, ax=ax2, edge_color='gray')
            
            # Destaca arestas de difusão
            if forwarded:
                diffusion_edges = []
                for sender in forwarded:
                    neighbors = self.network.get_neighbors(sender)
                    for neighbor in neighbors:
                        if neighbor in received:
                            diffusion_edges.append((sender, neighbor))
                
                if diffusion_edges:
                    nx.draw_networkx_edges(G, pos, edgelist=diffusion_edges,
                                          edge_color='red', width=2, alpha=0.6,
                                          arrows=True, arrowsize=15, ax=ax2)
            
            nx.draw_networkx_labels(G, pos, font_size=8, ax=ax2)
            
            truth_str = "Verdadeira" if message.truth == 1 else "Falsa"
            ax2.set_title(f'Difusão da Mensagem (Round)\n'
                         f'Verdade: {truth_str}, '
                         f'Alcance: {round_result["reach"]:.1%}, '
                         f'Taxa de Encaminhamento: {round_result["forwarding_rate"]:.1%}',
                         fontsize=12)
        else:
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                  node_size=300, ax=ax2, alpha=0.8)
            nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True, 
                                  arrowsize=10, ax=ax2, edge_color='gray')
            nx.draw_networkx_labels(G, pos, font_size=8, ax=ax2)
            ax2.set_title('Rede Social', fontsize=12)
        
        ax2.axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_metrics(self, save_path=None):
        """Plota métricas ao longo do tempo"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        rounds = range(1, len(self.history['reach']) + 1)
        
        # Alcance
        axes[0, 0].plot(rounds, self.history['reach'], marker='o', linewidth=2)
        axes[0, 0].set_title('Alcance da Mensagem', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Round')
        axes[0, 0].set_ylabel('Fração de Agentes que Receberam')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0, 1.1])
        
        # Taxa de encaminhamento
        axes[0, 1].plot(rounds, self.history['forwarding_rate'], marker='s', 
                       color='orange', linewidth=2)
        axes[0, 1].set_title('Taxa de Encaminhamento', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Round')
        axes[0, 1].set_ylabel('Fração que Encaminhou')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 1.1])
        
        # Contaminação por desinformação
        axes[1, 0].plot(rounds, self.history['misinformation_contamination'], 
                       marker='^', color='red', linewidth=2)
        axes[1, 0].set_title('Contaminação por Desinformação', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Round')
        axes[1, 0].set_ylabel('Fração que Encaminhou Mensagens Falsas')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1.1])
        
        # Reputação média
        avg_reputation = [np.mean([agent.reputation for agent in self.agents]) 
                         for _ in rounds]
        axes[1, 1].plot(rounds, avg_reputation, marker='d', color='green', linewidth=2)
        axes[1, 1].set_title('Reputação Média dos Agentes', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Round')
        axes[1, 1].set_ylabel('Reputação Média')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


def main():
    """Função principal para executar a simulação"""
    # Configuração fácil de modificar
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
        'variance_reputation': 1.0, # Variância da reputação inicial
        'bias_strength': 0.3,       # Peso do alinhamento ideológico (omega)
        'reputation_reward_strength': 0.5,  # Ganho de reputação por verdade (gamma)
        'reputation_penalty_strength': 0.5,  # Penalidade por desinformação (delta)
        'forwarding_cost': 0.1,     # Custo de encaminhar (k)
        'num_rounds': 10,           # Número de rounds
        'initial_senders': [0],     # IDs dos agentes que recebem mensagem inicial
        'seed': 42                  # Seed para reprodutibilidade
    }
    
    # Cria e executa simulação
    sim = Simulation(**config)
    results = sim.run()
    
    # Visualizações
    print("\nGerando visualizações...")
    
    # Mostra difusão do último round
    if results:
        sim.visualize_network(round_result=results[-1], 
                             save_path='network_diffusion.png')
    
    # Mostra métricas ao longo do tempo
    sim.plot_metrics(save_path='metrics.png')
    
    print("\nSimulação concluída!")
    print(f"Alcance médio: {np.mean(sim.history['reach']):.2%}")
    print(f"Taxa de encaminhamento média: {np.mean(sim.history['forwarding_rate']):.2%}")
    print(f"Contaminação média: {np.mean(sim.history['misinformation_contamination']):.2%}")


if __name__ == '__main__':
    main()

