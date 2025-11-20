🌍 Simulação de Relações Internacionais com Agentes Cognitivos (LLMs)

Este repositório contém um framework de simulação multi-agente projetado para modelar cenários de crise geopolítica e tomada de decisão estratégica. O sistema utiliza Grandes Modelos de Linguagem (LLMs) para personificar Chefes de Estado, um Analista de Inteligência e um Juiz Acadêmico, integrando conceitos de Teoria das Relações Internacionais com engenharia de prompt avançada.

🚀 Visão Geral

O projeto orquestra uma simulação baseada em turnos onde diferentes Estados-Nação (agentes) interagem diante de um cenário de crise. Diferente de chatbots comuns, este sistema implementa:

    Agentes de Estado: Atores com memória persistente (RAG), perfil ideológico, objetivos estratégicos e "linhas vermelhas".

    Juiz Especialista (RAG): Um agente avaliador que consulta uma base de conhecimento vetorial (um manual acadêmico de RI) para classificar as ações dos jogadores conforme teorias clássicas (Neorrealismo, Neoliberalismo, Construtivismo).

    Analista de Impacto: Um módulo que resume as consequências de cada rodada e determina o nível de escalada do conflito (DEFCON/Escalada).

    Saída Estruturada: Uso rigoroso de Pydantic para garantir que as decisões dos LLMs sigam schemas JSON validados e ações pré-definidas.

🛠️ Arquitetura Técnica

O sistema é construído em Python utilizando o ecossistema LangChain.

    Orquestrador (main.py): Gerencia o loop de simulação, carregamento de cenários e persistência de dados (CSV).

    Core (core/):

        agent.py: Implementa a memória vetorial (FAISS) e o loop de decisão com autocorreção (retry parser) para garantir JSONs válidos.

        judge.py: Pipeline RAG que fragmenta o manual de RI (manual_ri.pdf), cria embeddings e avalia a coerência teórica das jogadas.

        llm_builder.py: Factory para instanciar modelos de múltiplos provedores (OpenAI, Groq/Llama, DeepSeek, Maritaca, xAI).

    Configuração: Sistema modular para definir quais modelos controlam quais agentes.

📋 Pré-requisitos

    Python 3.10+

    Bibliotecas listadas em requirements.txt

    Chaves de API para os provedores que deseja utilizar (OpenAI, Groq, etc.)

⚙️ Instalação e Configuração

    Clone o repositório:
    Bash

git clone https://github.com/upassaro/Simulador-de-conflitos-territoriais-com-LLMs
cd seu-projeto

Instale as dependências: Recomenda-se o uso de um ambiente virtual (venv).
Bash

pip install -r requirements.txt

Configure as Variáveis de Ambiente: Crie um arquivo .env na raiz do projeto e adicione suas chaves de API:
Snippet de código

    OPENAI_API_KEY=sk-...
    GROQ_API_KEY=gsk_...
    MARITACA_API_KEY=...
    DEEPSEEK_API_KEY=...
    # Adicione apenas as chaves dos modelos que pretende usar

    Arquivos de Dados Necessários: Certifique-se de que a pasta data/ contenha:

        cenarios.json: Arquivo com a definição dos atores, contexto e sinopse da crise.

        manual_ri.pdf: O livro-texto ou artigo acadêmico que servirá de base para o Juiz (RAG).

▶️ Como Executar

Para iniciar a simulação completa (executando todos os cenários definidos em data/cenarios.json):
Bash

python main.py

O sistema verificará se o cenário já foi simulado. Caso contrário, iniciará as rodadas, exibindo no console as decisões dos agentes, os vereditos do juiz e a análise de impacto.

📂 Estrutura do Projeto

Plaintext

.
├── config/
│   └── llm_config.py      # Mapeamento de modelos (GPT-4, Llama-3, etc.)
├── core/
│   ├── agent.py           # Lógica do Agente de Estado (Memória + Decisão)
│   ├── analysis.py        # Agente Analista de Inteligência
│   ├── judge.py           # Agente Juiz (RAG + Avaliação Teórica)
│   ├── llm_builder.py     # Construtor de LLMs e Parsers
│   └── models.py          # Schemas Pydantic (Decision, Verdict)
├── data/
│   ├── cenarios.json      # Definição dos cenários de simulação
│   └── manual_ri.pdf      # Base de conhecimento para o Juiz
├── outputs/               # Resultados gerados (CSV)
├── main.py                # Entry point da aplicação
├── requirements.txt       # Dependências do projeto
└── .env                   # Variáveis de ambiente (não versionado)

🧪 Modelos Suportados

A arquitetura é agnóstica ao modelo, suportando atualmente via llm_config.py:

    OpenAI: GPT-4o, GPT-5-mini (simulado/beta)

    Groq: Llama 3 (70B)

    DeepSeek: DeepSeek-V3/Reasoner

    Maritaca AI: Sabiá-3

    xAI: Grok

📊 Resultados

Os resultados de cada simulação são salvos automaticamente na pasta outputs/ em formato CSV, contendo:

    Rodada e Timestamp

    Ação escolhida e justificativa

    Participação e voto no Conselho Global

    Veredito do Juiz (Realismo/Liberalismo/Construtivismo) e racional

    Modelo de LLM utilizado pelo agente

📝 Licença

Este projeto está sob a licença MIT. Consulte o arquivo LICENSE para mais detalhes.
