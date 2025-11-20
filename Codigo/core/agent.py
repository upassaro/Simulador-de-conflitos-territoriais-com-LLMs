import json

from langchain.memory import VectorStoreRetrieverMemory
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import ValidationError

# Importa todos os modelos de dados e o nosso erro personalizado
from .models import Decision, DecisionValidationError, FinalResolution

# --- PROMPT PARA A DECISÃO DA RODADA ---
AGENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Você é o principal tomador de decisões estratégicas do Estado-Nação fictício: {actor_name}.
Sua resposta DEVE ser um único e válido objeto JSON, estritamente conforme o schema fornecido.
Atenção: Garanta que a sua saída JSON usa a codificação UTF-8 correta para todos os caracteres especiais (ex: 'ç', 'ã').
Não adicione nenhum texto ou comentário fora do JSON.""",
        ),
        (
            "user",
            """# SEUS OBJETIVOS ESTRATÉGICOS PRINCIPAIS:
{objectives}

# CONTEXTO GLOBAL E DO SEU ESTADO
- **Cenário Atual:** {synopsis}
- **Seu Papel Neste Cenário:** {actor_role}
- **Seu Perfil Ideológico:** {ideological_profile}
- **Contexto Histórico do seu Estado:** {historical_context}
- **Situação Interna Atual:** {internal_context}
- **Suas Capacidades Militares e Tecnológicas:** {capabilities}
- **Suas Linhas Vermelhas (não cruzar):** {red_lines}

# MEMÓRIAS RELEVANTES DE AÇÕES PASSADAS (Recuperadas para si):
{history}

# BRIEFING DE INTELIGÊNCIA DA ÚLTIMA RODADA
- **Eventos:** {situation_summary}
- **Análise de Impacto:** {impact_analysis}
- **Nível de Escalada Atual:** Nível {escalation_level} (de 0 a 5).

# SITUAÇÃO ATUAL
- **Rodada Atual:** {round_number} de 20

# SUA MISSÃO
- Sua ação na última rodada foi: '{last_action}'. Repetir a mesma estratégia pode ser ineficaz.
- Avalie se a sua próxima ação o aproxima dos seus OBJETIVOS ESTRATÉGICOS.

Com base em TODO o contexto, decida a próxima ação estratégica do seu Estado.
1.  **Escolha UMA Ação Principal (`action_primary`):** ...
2.  **Justifique sua Decisão (`justification_text`):** ...
3.  **Decida sobre o Conselho Global:** ...

# REGRAS RÍGIDAS PARA A SAÍDA:
- Você DEVE escolher os valores para os campos 'action_primary' e 'council_action' COPIANDO EXATAMENTE as strings da lista de opções fornecida no schema JSON abaixo.
- NÃO modifique, abrevie ou reescreva as opções. A sua resposta deve ser uma correspondência exata.

# SCHEMA JSON OBRIGATÓRIO
{schema}
""",
        ),
    ]
)

CORRECTION_PROMPT = ChatPromptTemplate.from_template(
    """Sua tarefa é corrigir um objeto JSON inválido. Um agente de IA tentou gerar uma resposta JSON, mas falhou na validação.

# SCHEMA JSON OBRIGATÓRIO (A RESPOSTA DEVE SEGUIR ESTA ESTRUTURA):
{schema}

# ERRO DE VALIDAÇÃO DETECTADO:
{validation_error}

# SAÍDA JSON DEFEITUOSA (GERADA PELO AGENTE):
```json
{faulty_output}

SUA MISSÃO:
Analise o erro, a saída defeituosa e o schema. Corrija o JSON para que ele se conforme estritamente com o schema e o erro apontado.
Sua resposta deve conter APENAS o objeto JSON corrigido e válido, sem nenhum outro texto, comentário ou explicação.
"""
)

class StateAgent:
    """Representa um único ator com memória vetorial e capacidade de autocorreção."""
    def __init__(
    self, llm: Runnable, actor_data: dict, role: str, embedding_model: HuggingFaceEmbeddings
    ):
        """
        Inicializa o agente de estado.

        Args:
            llm (Runnable): O modelo de linguagem a ser usado.
            actor_data (dict): Dados que definem o ator.
            role (str): O papel do ator no cenário.
            embedding_model (HuggingFaceEmbeddings): O modelo para criar embeddings de texto.
        """
        self.llm = llm
        self.actor_data = actor_data
        self.name = actor_data.get("name", "Nome Desconhecido")
        self.role = role
        self.llm_config = {}

        # Tenta extrair a configuração do objeto LLM para uso posterior
        if hasattr(llm, "bound") and hasattr(llm.bound, "model"):
            self.llm_config = {"provider": "desconhecido", "model": llm.bound.model}
        elif hasattr(llm, "model_name"):
            self.llm_config = {"provider": "desconhecido", "model": llm.model_name}

        # Configura a memória vetorial para o agente
        vectorstore = FAISS.from_texts(
            texts=["Início do registo de memória."], embedding=embedding_model
        )
        retriever = vectorstore.as_retriever(search_kwargs=dict(k=3))

        self.memory = VectorStoreRetrieverMemory(
            retriever=retriever,
            input_key="situation_summary",
            memory_key="history",
        )

    def decide(
        self,
        synopsis: str,
        situation_summary: str,
        round_number: int,
        last_action: str | None,
        impact_analysis: str,
        escalation_level: int,
    ) -> Decision:
        """
        Processa o contexto e invoca o LLM para decidir, com um loop de autocorreção.

        Args:
            synopsis (str): Sinopse geral do cenário.
            situation_summary (str): Resumo dos eventos da última rodada.
            round_number (int): O número da rodada atual.
            last_action (str | None): A última ação tomada por este agente.
            impact_analysis (str): Análise do impacto da última rodada.
            escalation_level (int): O nível de escalada atual.

        Returns:
            Decision: Um objeto de decisão validado.

        Raises:
            DecisionValidationError: Se o agente não conseguir produzir uma saída válida.
        """
        print(f"\n🤖 Invocando agente: {self.name} (Papel: {self.role})")

        max_attempts = 2
        response_data = None

        for attempt in range(max_attempts):
            try:
                # Na primeira tentativa, usa o prompt normal
                if attempt == 0:
                    chain = (
                        RunnablePassthrough.assign(
                            history=self.memory.load_memory_variables
                        )
                        | AGENT_PROMPT
                        | self.llm
                    )

                    objectives = self.actor_data.get(
                        "objectives", "Agir conforme o perfil ideológico."
                    )
                    red_lines = self.actor_data.get("alliances", {}).get(
                        "red_lines", "Nenhuma definida."
                    )

                    response_data = chain.invoke(
                        {
                            "objectives": objectives,
                            "actor_name": self.name,
                            "synopsis": synopsis,
                            "actor_role": self.role,
                            "ideological_profile": self.actor_data.get("ideological_profile"),
                            "historical_context": json.dumps(self.actor_data.get("historical_context", {})),
                            "internal_context": json.dumps(self.actor_data.get("internal_context", {})),
                            "capabilities": json.dumps(self.actor_data.get("capabilities", {})),
                            "red_lines": red_lines,
                            "round_number": round_number,
                            "situation_summary": situation_summary or "Nenhuma ação foi tomada ainda.",
                            "last_action": last_action or "Nenhuma (esta é a primeira rodada)",
                            "impact_analysis": impact_analysis,
                            "escalation_level": escalation_level,
                            "schema": json.dumps(Decision.model_json_schema(), ensure_ascii=False, indent=2),
                        }
                    )

                # Valida a resposta com o modelo Pydantic
                decision = (
                    Decision(**response_data)
                    if isinstance(response_data, dict)
                    else Decision.model_validate(response_data)
                )

                # Salva a decisão na memória
                memoria_para_guardar = (
                    f"Na rodada {round_number}, meus objetivos eram '{objectives}'. "
                    f"A situação era: '{situation_summary}'. "
                    f"Minha decisão foi '{decision.action_primary}' porque '{decision.justification_text}'."
                )
                self.memory.save_context(
                    {"situation_summary": situation_summary or "Início da simulação."},
                    {"output": memoria_para_guardar},
                )
                print(f"  -> Memória de '{self.name}' foi atualizada.")
                return decision

            except ValidationError as e:
                print(f"   ⚠️ Erro de validação Pydantic para '{self.name}' na tentativa {attempt + 1}.")

                if attempt < max_attempts - 1:
                    print("      A tentar autocorreção...")
                    correction_chain = CORRECTION_PROMPT | self.llm

                    faulty_output_str = (
                        json.dumps(response_data, ensure_ascii=False)
                        if isinstance(response_data, dict)
                        else str(response_data)
                    )

                    # Usa a saída defeituosa para tentar a correção
                    response_data = correction_chain.invoke(
                        {
                            "validation_error": str(e),
                            "faulty_output": faulty_output_str,
                            "schema": json.dumps(Decision.model_json_schema(), ensure_ascii=False, indent=2),
                        }
                    )
                else:
                    print(f"   ❌ Autocorreção falhou para '{self.name}'. A registar a falha definitiva.")
                    raise DecisionValidationError(message=str(e), raw_output=response_data)

        raise DecisionValidationError(
            message="O agente não conseguiu produzir uma decisão válida após as tentativas de correção.",
            raw_output=response_data,
        )