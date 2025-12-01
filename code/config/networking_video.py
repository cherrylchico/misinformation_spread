import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import numpy as np

# --- CONFIGURATION ---
NUM_NODES = 20
SEED_NODES = 2
FRAMES = 60  # Extended for Round 2

# Initialize Graph
G = nx.barabasi_albert_graph(NUM_NODES, 2)
pos = nx.spring_layout(G, seed=42)

# Node States: 0=Unaware, 1=Received, 2=Shared, 3=Ignored
node_states = {n: 0 for n in G.nodes()}
reputation = {n: 50 for n in G.nodes()} # Start with 50 rep
penalized_nodes = set() # Track who got penalized in Round 1
colors = []

fig, ax = plt.subplots(figsize=(8, 6))

def update(frame):
    ax.clear()
    ax.set_title(f"Simulation Step: {frame}", fontsize=15)
    
    # LOGIC STEPS BASED ON FRAME NUMBER
    
    # --- ROUND 1 ---
    
    # 1. Message Generation & Seeding (Frames 0-5)
    if frame < 5:
        ax.set_title("Step 1: Seeding Message (Bias: Right, Truth: ?)")
        # Highlight hubs
        seeds = [n for n, d in G.degree()] 
        seeds.sort(reverse=True)
        active_seeds = seeds[:SEED_NODES]
        
        color_map = []
        for n in G.nodes():
            if n in active_seeds:
                color_map.append('orange') # Seeds
                node_states[n] = 2 # They share
            else:
                color_map.append('lightgrey')
        
        nx.draw(G, pos, node_color=color_map, with_labels=True, node_size=400, ax=ax, edge_color="gray")

    # 2. Diffusion / Spreading (Frames 6-20)
    elif frame < 20:
        ax.set_title("Step 2: Diffusion Waves (Utility Calc)")
        
        # Simple diffusion simulation for visual effect
        current_sharers = [n for n, s in node_states.items() if s == 2]
        
        for sender in current_sharers:
            for neighbor in G.neighbors(sender):
                if node_states[neighbor] == 0:
                    # Random visual decision: 50% chance to share
                    if random.random() > 0.5:
                        node_states[neighbor] = 2 # Share
                    else:
                        node_states[neighbor] = 3 # Ignore
        
        color_map = []
        for n in G.nodes():
            if node_states[n] == 2: color_map.append('green') # Sharer
            elif node_states[n] == 3: color_map.append('red') # Ignorer
            elif node_states[n] == 0: color_map.append('lightgrey') # Unaware
            
        nx.draw(G, pos, node_color=color_map, with_labels=True, node_size=400, ax=ax, edge_color="gray")

    # 3. Truth Revelation (Frames 20-25)
    elif frame < 25:
        ax.set_title("Step 3: Truth Revealed: IT WAS A LIE!")
        color_map = ['green' if node_states[n]==2 else 'lightgrey' for n in G.nodes()]
        nx.draw(G, pos, node_color=color_map, with_labels=True, node_size=400, ax=ax, edge_color="gray")
        
    # 4. Reputation Update (Frames 25-30)
    elif frame < 30:
        ax.set_title("Step 4: Reputation Penalty Applied")
        color_map = []
        sizes = []
        for n in G.nodes():
            if node_states[n] == 2: 
                color_map.append('darkred') # Penalized
                sizes.append(100) # Shrink size visually
                penalized_nodes.add(n)
            else:
                color_map.append('lightgrey')
                sizes.append(300)
                
        nx.draw(G, pos, node_color=color_map, with_labels=True, node_size=sizes, ax=ax)

    # --- ROUND 2 ---
    
    # 5. Round 2 Seeding (Frames 30-35)
    elif frame < 35:
        ax.set_title("Step 5: Round 2 - New Message (Bias: Left)")
        
        # Reset states for Round 2, but keep penalized nodes memory
        if frame == 30:
            for n in G.nodes():
                node_states[n] = 0
        
        # New seeds (maybe different ones or same)
        seeds = [n for n, d in G.degree()] 
        seeds.sort(reverse=True)
        active_seeds = seeds[:SEED_NODES]
        
        color_map = []
        sizes = []
        for n in G.nodes():
            # Size depends on reputation (penalized nodes stay small)
            size = 100 if n in penalized_nodes else 300
            sizes.append(size)
            
            if n in active_seeds:
                color_map.append('orange') # Seeds
                node_states[n] = 2 # They share
            else:
                color_map.append('lightgrey')
                
        nx.draw(G, pos, node_color=color_map, with_labels=True, node_size=sizes, ax=ax, edge_color="gray")

    # 6. Round 2 Diffusion (Frames 35-50)
    elif frame < 50:
        ax.set_title("Step 6: Round 2 Diffusion (Penalized nodes have less influence)")
        
        current_sharers = [n for n, s in node_states.items() if s == 2]
        
        for sender in current_sharers:
            for neighbor in G.neighbors(sender):
                if node_states[neighbor] == 0:
                    # Logic: If sender was penalized, neighbor is LESS likely to share
                    prob_share = 0.2 if sender in penalized_nodes else 0.6
                    
                    if random.random() < prob_share:
                        node_states[neighbor] = 2 # Share
                    else:
                        node_states[neighbor] = 3 # Ignore
        
        color_map = []
        sizes = []
        for n in G.nodes():
            size = 100 if n in penalized_nodes else 300
            sizes.append(size)
            
            if node_states[n] == 2: color_map.append('green')
            elif node_states[n] == 3: color_map.append('red')
            else: color_map.append('lightgrey')
            
        nx.draw(G, pos, node_color=color_map, with_labels=True, node_size=sizes, ax=ax, edge_color="gray")

    # 7. Round 2 Outcome (Frames 50-60)
    else:
        ax.set_title("Step 7: Round 2 Truth Revealed (TRUE)")
        color_map = []
        sizes = []
        for n in G.nodes():
            size = 100 if n in penalized_nodes else 300 # Keep size from R1
            sizes.append(size)
            
            # If shared true message, maybe recover size/color?
            # For simplicity, just show green for correct sharers
            if node_states[n] == 2:
                color_map.append('gold') # Reward!
            else:
                color_map.append('lightgrey')
                
        nx.draw(G, pos, node_color=color_map, with_labels=True, node_size=sizes, ax=ax, edge_color="gray")

# Create Animation
ani = animation.FuncAnimation(fig, update, frames=FRAMES, interval=200, repeat=False)

# To save (remove comment below if you have ffmpeg installed)
ani.save('simulation_round.gif', writer='pillow')
plt.show()