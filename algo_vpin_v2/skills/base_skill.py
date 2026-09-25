"""
Base Trading Skill Interface
Foundation for all modular skills in Chinmay's Option Scalping Architecture.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class SkillResult:
    """Standardized output returned by any Trading Skill"""
    is_favorable: bool
    confidence: float = 0.50
    signal: Optional[str] = None
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseTradingSkill(ABC):
    """
    Abstract Base Class for modular Trading Skills.
    Each skill encapsulates a specific domain of market expertise
    (e.g., Price Action, Strike Selection, R:R calculation, Position Guardian).
    """
    def __init__(self, name: str):
        self.name = name
        self.is_active: bool = True

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> SkillResult:
        """Execute the skill's domain evaluation logic"""
        pass

    def __repr__(self) -> str:
        return f"<TradingSkill: {self.name} (Active={self.is_active})>"
