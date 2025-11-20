# --------------------------------------------------------------------------------
# CONFIGURAÇÃO DOS MODELOS DE LINGUAGEM PARA A SIMULAÇÃO
# --------------------------------------------------------------------------------
# Este arquivo centraliza a definição de todos os LLMs usados no projeto.
# Os campos "provider" e "model" são usados pela função `build_llm` para instanciar 
# o modelo correto.
# --------------------------------------------------------------------------------

LLM_CONFIG = {
    # Agente 1: OpenAI (GPT-5-MINI)
    "agente_openai_gpt": {
        "provider": "openai",
        "model": "gpt-5-mini-2025-08-07"
    },

   # Agente 2: Llama 3 (70B) via Groq
    "agente_groq_llama": {
        "provider": "groq",
        "model": "llama-3.3-70b-versatile"
    },

    # Agente 3: DeepSeek (v.3)
    "agente_deepseek": {
        "provider": "deepseek",  
        "model": "deepseek-reasoner" 
    },

    # Agente 4: MaritacaAI (Sabiá-3.1)
    "agente_maritaca_sabia": {
        "provider": "maritaca", 
        "model": "sabia-3.1"
    },
    
    # Agente 5: Grok (modelo da xAI)
    "agente_xai_grok": {
        "provider": "xai", 
        "model": "grok-4-fast-non-reasoning"
    },

    # O Juiz: OpenAI.
    "juiz": {
        "provider": "openai",
        "model": "gpt-5-2025-08-07"
    }
}