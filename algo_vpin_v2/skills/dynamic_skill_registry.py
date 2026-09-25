"""
Dynamic Skill Registry & Hot-Loader for Algo VPIN v2.0
Enables zero-downtime hot-reloading of new trading skills and AI-generated pattern detectors.
"""

import importlib
import inspect
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_skill import BaseTradingSkill, SkillResult

logger = logging.getLogger("algo_vpin_v2.skills.dynamic_registry")


class DynamicSkillRegistry:
    """
    Central Registry for modular trading skills with runtime hot-reloading.
    """
    def __init__(self, skills_dir: Optional[Path] = None):
        self.skills_dir = skills_dir or Path(__file__).resolve().parent
        self.registered_skills: Dict[str, BaseTradingSkill] = {}
        self.discover_and_load_skills()

    def discover_and_load_skills(self) -> int:
        """
        Scans `algo_vpin_v2/skills/` for all classes implementing `BaseTradingSkill`
        and registers them dynamically into memory.
        """
        loaded_count = 0
        for py_file in self.skills_dir.glob("*.py"):
            if py_file.name in ("__init__.py", "base_skill.py", "dynamic_skill_registry.py"):
                continue

            module_name = f"algo_vpin_v2.skills.{py_file.stem}"
            try:
                # Dynamic import or reload
                if module_name in importlib.sys.modules:
                    mod = importlib.reload(importlib.sys.modules[module_name])
                else:
                    mod = importlib.import_module(module_name)

                for attr_name in dir(mod):
                    cls = getattr(mod, attr_name)
                    if (
                        inspect.isclass(cls)
                        and issubclass(cls, BaseTradingSkill)
                        and cls is not BaseTradingSkill
                    ):
                        try:
                            instance = cls()
                            self.registered_skills[instance.name] = instance
                            loaded_count += 1
                        except Exception as e:
                            logger.debug(f"[SkillRegistry] Instantiation notice for {cls.__name__}: {e}")

            except Exception as e:
                logger.error(f"[SkillRegistry] Error loading skill module {py_file.name}: {e}")

        logger.info(f"[SkillRegistry] Dynamic Skill Registry loaded {len(self.registered_skills)} active skills: {list(self.registered_skills.keys())}")
        return len(self.registered_skills)

    def hot_reload_skills(self) -> int:
        """Zero-downtime hot-reload triggered when AI Agent creates or improves a skill"""
        logger.info("[SkillRegistry] Initiating live Hot-Reload of all trading skills...")
        return self.discover_and_load_skills()

    def get_skill(self, name: str) -> Optional[BaseTradingSkill]:
        return self.registered_skills.get(name)

    def get_active_skills(self) -> Dict[str, BaseTradingSkill]:
        """Returns dictionary of all active registered skills"""
        return {k: v for k, v in self.registered_skills.items() if getattr(v, "is_active", True)}

    def evaluate_all(self, *args, **kwargs) -> Dict[str, SkillResult]:
        """Runs evaluation across all dynamically active skills"""
        results = {}
        for name, skill in self.registered_skills.items():
            if getattr(skill, "is_active", True):
                try:
                    res = skill.evaluate(*args, **kwargs)
                    results[name] = res
                except Exception as e:
                    logger.debug(f"[SkillRegistry] Error evaluating skill {name}: {e}")
        return results
