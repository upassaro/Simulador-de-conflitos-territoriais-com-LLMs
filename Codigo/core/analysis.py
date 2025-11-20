import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field
from typing import Literal

from .models import Decision 
from .llm_builder import build_llm
from config.llm_config import LLM_CONFIG

class RoundAnalysis(BaseModel):
    """Modelo de dados para a análise de uma rodada."""
    impact_summary: str = Field(
        ...,
        description="Um resumo narrativo de 2-3 frases sobre as consequências diretas das ações da rodada (ex: 'A ação X aumentou a tensão, enquanto a ação Y abriu um canal diplomático. Nenhum dano económico foi reportado.')"
    )
    escalation_level: Literal[0, 1, 2, 3, 4, 5] = Field(
        ...,
        description="O número inteiro (de 0 a 5) que melhor representa o nível de escalada atual do conflito, com base nas ações da rodada."
    )

ANALYSIS_PROMPT = ChatPromptTemplate.from_template(
    """Você é um analista de inteligência neutro e objetivo. Sua tarefa é analisar as ações estratégicas tomadas por vários Estados-Nação em uma rodada de simulação e fornecer um resumo conciso do seu impacto.

# ESCALA DE ESCALADA DE REFERÊNCIA:
- Nível 0: Disputa retórica/sanções
- Nível 1: Escaramuças/zonas cinzentas
- Nível 2: Proxy intensificado/ataques cibernéticos significativos
- Nível 3: Mísseis convencionais limitados
- Nível 4: Ameaças e demonstrações nucleares
- Nível 5: Crise máxima multilateral

# AÇÕES TOMADAS NA ÚLTIMA RODADA:
{round_actions}

# SUA MISSÃO:
Com base nas ações acima, forneça uma análise estruturada.
1.  **impact_summary:** Escreva um resumo narrativo (máximo 3 frases) sobre as consequências geopolíticas e materiais das ações tomadas.
2.  **escalation_level:** Com base nas ações, escolha o número do nível de escalada que melhor representa o estado atual do conflito.

# SCHEMA JSON OBRIGATÓRIO:
{schema}
"""
)

class AnalysisModule:
    """Módulo que usa um LLM para analisar o resultado de cada rodada."""
    def __init__(self):
        analyst_config = LLM_CONFIG.get("juiz") 

        self.llm = build_llm(
            provider=analyst_config["provider"],
            model=analyst_config["model"],
            structured_output_model=RoundAnalysis
        )
        self.chain = ANALYSIS_PROMPT | self.llm

    def analyze_round(self, round_decisions: dict) -> RoundAnalysis:
        """
        Analisa as decisões de uma rodada e retorna um objeto RoundAnalysis.

        Args:
            round_decisions: Um dicionário com {nome_do_ator: ação_tomada}.
        """
        actions_text = "\n".join([f"- {actor}: '{action}'" for actor, action in round_decisions.items()])

        response = self.chain.invoke({
            "round_actions": actions_text,
            "schema": json.dumps(RoundAnalysis.model_json_schema(), indent=2, ensure_ascii=False)
        })
        return response