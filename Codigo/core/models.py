from pydantic import BaseModel, Field
from typing import Literal, Optional, List

# Lista de possíveis ações para facilitar a manutenção

MILITARY_ACTIONS = Literal[
    "convocação parcial de reservistas", "realização de exercícios militares dissuasórios",
    "reforço de guarnições de fronteira", "operações de reconhecimento persistente com drones",
    "interdição limitada de alvos com drones", "ativação de defesas antidrone",
    "intrusão em redes civis críticas", "derrubada temporária de redes de comunicação",
    "endurecimento de defesas cibernéticas", "lançamento de salvas de mísseis convencionais de precisão",
    "interdição de bases militares inimigas (alto nível)", "elevação do nível de prontidão nuclear",
    "realização de teste nuclear simbólico", "ameaça de emprego limitado de arma nuclear (sem detalhes)"
]

DIPLOMATIC_ACTIONS = Literal[
    "propor cessar-fogo monitorado bilateralmente", "convocar comissões de limites territoriais",
    "oferecer troca de garantias de segurança", "impor sanções econômicas unilaterais ou com aliados",
    "reconhecer formalmente uma entidade separatista", "expulsar missões diplomáticas",
    "estabelecer canais de comunicação militar de emergência", "propor mediação por terceiros fora do Conselho Global"
]

COUNCIL_ACTIONS = Literal[
    "propor resolucao condenatoria contra um ator", "iniciar investigacao independente sobre um incidente",
    "apoiar sancoes multilaterais aprovadas pelo conselho", "propor novo tratado para criacao de normas internacionais",
    "financiar fundo de ajuda humanitaria supervisionado pelo conselho", "vetar resolucao do conselho",
    "abster-se de votacao no conselho"
]

class Decision(BaseModel):
    """Representa uma decisão completa de um ator em uma rodada."""
    action_primary: Literal[MILITARY_ACTIONS, DIPLOMATIC_ACTIONS] = Field(
        ...,
        description="A principal ação estratégica militar ou diplomática escolhida pelo ator."
    )

    justification_text: str = Field(
        ...,
        min_length=20,
        description="O raciocínio detalhado por trás da escolha da ação principal."
    )

    council_participation: Optional[Literal["participar", "abster-se"]] = Field(
        None,
        description="A decisão do ator de participar ou não da sessão do Conselho Global nesta rodada."
    )

    council_action: Optional[COUNCIL_ACTIONS] = Field(
        None,
        description="Se participou do conselho, qual ação institucional foi escolhida. Deve ser nulo se a participação for 'abster-se'."
    )

class Verdict(BaseModel):
    """Representa a avaliação do Juiz sobre uma decisão."""
    verdict: Literal["Neorrealismo", "Neoliberalismo", "Construtivismo"] = Field(
        ...,
        description="A classificação teórica da decisão."
    )

    rationale: str = Field(
        ...,
        min_length=20,
        description="A justificativa do Juiz para a classificação, possivelmente citando o manual de RI."
    )

class DecisionValidationError(Exception):
    """Exceção personalizada para quando a saída do LLM falha na validação Pydantic para a classe Decision."""
    def __init__(self, message, raw_output):
        super().__init__(message)
        self.raw_output = raw_output 



    