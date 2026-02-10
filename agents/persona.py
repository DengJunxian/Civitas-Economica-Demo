# file: agents/persona.py
"""
Agent Persona System
Defines the personality traits and generation logic for Civitas agents.
"""

from dataclasses import dataclass, field
from enum import Enum
import random
from typing import List, Dict, Optional

class RiskAppetite(str, Enum):
    CONSERVATIVE = "Conservative" # Low risk tolerance, prefers cash/bonds
    BALANCED = "Balanced"         # Moderate risk
    AGGRESSIVE = "Aggressive"     # High risk, seeks alpha
    GAMBLER = "Gambler"           # Very high risk, likes volatility

class InvestmentHorizon(str, Enum):
    SHORT_TERM = "Short-term"     # Day trader / Scalper
    MEDIUM_TERM = "Medium-term"   # Swing trader
    LONG_TERM = "Long-term"       # Value investor

@dataclass
class Persona:
    """
    Defines the psychological and strategic profile of a TraderAgent.
    """
    name: str = "Anonymous"
    risk_appetite: RiskAppetite = RiskAppetite.BALANCED
    investment_horizon: InvestmentHorizon = InvestmentHorizon.MEDIUM_TERM
    
    # Social Psychology Traits
    conformity: float = 0.5       # 0.0 (Contrarian) -> 1.0 (Herd Follower)
    influence: float = 0.5        # 0.0 (Invisible) -> 1.0 (Opinion Leader)
    patience: float = 0.5         # 0.0 (Impulsive) -> 1.0 (Patient)
    
    # Cognitive Biases (0.0 to 1.0 strength)
    loss_aversion: float = 2.25   # Kahneman's Lambda (Standard ~2.25)
    overconfidence: float = 0.0
    
    def __str__(self):
        return (f"Persona({self.name} | {self.risk_appetite.value} | "
                f"{self.investment_horizon.value} | Conformity: {self.conformity:.2f})")

class PersonaGenerator:
    """
    Generates diverse personas for agents.
    For now, uses template-based generation with random variations.
    Can be upgraded to use LLM for rich narratives.
    """
    
    FIRST_NAMES = ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Heidi", 
                   "Ivan", "Judy", "Kevin", "Lily", "Mike", "Nina", "Oscar", "Peggy"]
    
    ROLES = ["Retail Investor", "Day Trader", "Fund Manager", "Quant", "Speculator", "Pension Fund"]

    @staticmethod
    def generate_random_persona() -> Persona:
        """Generate a completely random persona."""
        name = f"{random.choice(PersonaGenerator.FIRST_NAMES)}_{random.randint(100,999)}"
        
        risk = random.choice(list(RiskAppetite))
        horizon = random.choice(list(InvestmentHorizon))
        
        # Correlate traits slightly
        conformity = random.betavariate(2, 2) # Bell curve around 0.5
        influence = random.paretovariate(3)   # Heavy tail (few influencers)
        influence = min(0.95, influence / 5.0) # Normalize
        
        # Adjust based on risk/horizon
        if risk == RiskAppetite.CONSERVATIVE:
            loss_aversion = random.uniform(2.5, 3.5)
            patience = random.uniform(0.6, 0.9)
        elif risk == RiskAppetite.GAMBLER:
            loss_aversion = random.uniform(1.0, 1.5)
            patience = random.uniform(0.1, 0.4)
            conformity = random.uniform(0.6, 0.9) # Gamblers often follow crowds
        else:
            loss_aversion = random.uniform(2.0, 2.5)
            patience = random.uniform(0.3, 0.7)
            
        return Persona(
            name=name,
            risk_appetite=risk,
            investment_horizon=horizon,
            conformity=conformity,
            influence=influence,
            patience=patience,
            loss_aversion=loss_aversion
        )

    @staticmethod
    def generate_distribution(n: int) -> List[Persona]:
        """
        Generate a distribution of personas representing a market ecosystem.
        e.g., 70% Retail, 20% Institutions, 10% Gamblers
        """
        personas = []
        for _ in range(n):
            r = random.random()
            if r < 0.7:
                # Retail (Mixed, leaning herding)
                p = PersonaGenerator.generate_random_persona()
                p.influence = min(0.3, p.influence)
            elif r < 0.9:
                # Institution (Lower conformity, higher influence)
                p = PersonaGenerator.generate_random_persona()
                p.risk_appetite = RiskAppetite.BALANCED
                p.investment_horizon = InvestmentHorizon.LONG_TERM
                p.conformity = random.uniform(0.1, 0.3) 
                p.influence = random.uniform(0.5, 0.9)
                p.name = f"Institution_{random.randint(1, 99)}"
            else:
                # Whales / Influencers
                p = PersonaGenerator.generate_random_persona()
                p.influence = random.uniform(0.8, 1.0)
                p.name = f"Whale_{random.randint(1, 99)}"
                
            personas.append(p)
        return personas
