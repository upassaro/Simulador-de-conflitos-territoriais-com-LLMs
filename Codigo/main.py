import json
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
from core.analysis import AnalysisModule
import os 

# Importando as nossas ferramentas
from config.llm_config import LLM_CONFIG
from core.llm_builder import build_llm
from core.models import Decision, FinalResolution, DecisionValidationError
from core.agent import StateAgent
from core.judge import Judge
from langchain_huggingface import HuggingFaceEmbeddings

def run_full_simulation():
    """
    Função principal que carrega todos os dados e orquestra a execução completa
    de todos os cenários, incluindo a fase de resolução final.
    """
    load_dotenv()
    print("1. Variáveis de ambiente carregadas.")

    try:
        with open('data/cenarios.json', 'r', encoding='utf-8') as f:
            dados_cenarios = json.load(f)
        scenarios = dados_cenarios.get("scenarios", [])
        if not scenarios:
            print("❌ Erro: Nenhum cenário encontrado no arquivo JSON.")
            return
        print(f"2. Arquivo de cenários carregado. {len(scenarios)} cenários encontrados.")
    except Exception as e:
        print(f"❌ Erro ao carregar cenários: {e}")
        return

    try:
        juiz = Judge(pdf_path="data/manual_ri.pdf")
    except Exception as e:
        print(f"❌ Falha ao inicializar o Juiz. Erro: {e}")
        return
    
    try:
        analyst = AnalysisModule()
        print("4. Módulo de Análise de Inteligência inicializado.")
    except Exception as e:
        print(f"❌ Falha ao inicializar o Módulo de Análise. Erro: {e}")
        return

    print("3. Criando o modelo de embedding compartilhado (pode demorar na primeira vez)...")
    try:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("   ✅ Modelo de embedding carregado.")
    except Exception as e:
        print(f"   ❌ Erro ao carregar modelo de embedding: {e}")
        return

    agent_llm_configs = {k: v for k, v in LLM_CONFIG.items() if k != "juiz"}
    llm_keys_ordered = list(agent_llm_configs.keys())
    
    
    for scenario_index, current_scenario in enumerate(scenarios):
        scenario_id = current_scenario.get("id", f"SCN-{scenario_index+1}")

        output_filename = f"outputs/resultados_{scenario_id}.csv"
        if os.path.exists(output_filename):
            print(f"\n✅ Cenário {scenario_id} já foi concluído (arquivo encontrado). A saltar.")
            continue

        print(f"\n\n{'='*20} INICIANDO SIMULAÇÃO PARA O CENÁRIO: {scenario_id} - {current_scenario['title']} {'='*20}")

        scenario_results = []

        agents = {}
        actor_names = [actor['name'] for actor in current_scenario['actors']]
        
        current_llm_assignment = {actor_names[i]: llm_keys_ordered[(i + scenario_index) % len(llm_keys_ordered)] for i in range(len(actor_names))}

        for actor_data in current_scenario['actors']:
            actor_name = actor_data['name']
            llm_key = current_llm_assignment[actor_name]
            config_llm = agent_llm_configs[llm_key]
            
            print(f"  - Preparando ator '{actor_name}' com o LLM '{llm_key}' ({config_llm['model']})")
            
            agent_llm = build_llm(
                provider=config_llm["provider"],
                model=config_llm["model"],
                structured_output_model=Decision
            )
            
            agents[actor_name] = StateAgent(
                llm=agent_llm,
                actor_data=actor_data,
                role=current_scenario["role_assignment"][actor_name],
                embedding_model=embedding_model
            )

        last_actions = {} 
        situation_summary = "Esta é a primeira rodada. O cenário acaba de começar."
        impact_analysis = "Nenhuma, esta é a primeira rodada."
        escalation_level = 0
        total_rounds = 20
        for round_num in range(1, total_rounds + 1):
            print(f"\n--- Cenário {scenario_id} | Rodada {round_num}/{total_rounds} ---")
            
            round_decisions = {}
            for actor_name, agent in agents.items():
                try:
                    last_action_for_agent = last_actions.get(actor_name)

                    decision = agent.decide(
                        synopsis=current_scenario["synopsis"],
                        situation_summary=situation_summary,
                        round_number=round_num,
                        last_action=last_action_for_agent,
                        impact_analysis=impact_analysis,
                        escalation_level=escalation_level
                    )
                    
                    verdict = juiz.evaluate(decision=decision)
                    
                    round_decisions[actor_name] = decision.action_primary
                    last_actions[actor_name] = decision.action_primary
                    
                    config_llm = agent_llm_configs[current_llm_assignment[actor_name]]
                    llm_key = current_llm_assignment[actor_name]
                    result_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "scenario_id": scenario_id,
                        "scenario_type": current_scenario.get("scenario_type"),
                        "round_number": round_num,
                        "actor_name": actor_name,
                        "actor_role": agent.role,
                        "llm_config_key": llm_key,
                        "llm_provider": config_llm["provider"],
                        "llm_model": config_llm["model"],
                        "action_primary": decision.action_primary,
                        "council_participation": decision.council_participation,
                        "council_action": decision.council_action,
                        "judge_verdict": verdict.verdict,
                        "justification_text": decision.justification_text,
                        "judge_rationale": verdict.rationale,
                    }
                    scenario_results.append(result_entry)
                    
                    print(f"  -> Decisão de '{actor_name}': {decision.action_primary} | Veredito do Juiz: {verdict.verdict}")

                except Exception as e:
                    print(f"  ❌ Erro ao processar a decisão para '{actor_name}'. Erro: {e}")
                    scenario_results.append({"scenario_id": scenario_id, "round_number": round_num, "actor_name": actor_name, "error": str(e)})

            if round_decisions:
                try:
                    print("   -- Analisando o impacto da rodada...")
                    analysis_result = analyst.analyze_round(round_decisions)
    
                    situation_summary = "; ".join([f"{name} escolheu '{action}'" for name, action in round_decisions.items()])
                    impact_analysis = analysis_result.impact_summary
                    escalation_level = analysis_result.escalation_level
                    print(f"   -- Análise: Impacto: '{impact_analysis}'. Novo Nível de Escalada: {escalation_level}")
                except Exception as e:
                    print(f"   ⚠️ Erro na análise da rodada: {e}. A usar dados da rodada anterior.")
            else:
                print("   -- Nenhuma decisão bem-sucedida na rodada para analisar.")

        if scenario_results:
            df_results = pd.DataFrame(scenario_results)
            df_results.to_csv(output_filename, index=False, encoding='utf-8-sig')
            print(f"\n✅ Resultados do Cenário {scenario_id} salvos em: '{output_filename}'")
        else:
            print(f"⚠️ Nenhuma decisão foi registada para o cenário {scenario_id}.")

    print(f"\n\n{'='*20} TODAS AS SIMULAÇÕES DISPONÍVEIS FORAM CONCLUÍDAS {'='*20}")

if __name__ == "__main__":
    run_full_simulation()