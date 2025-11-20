import os
import json
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Ferramentas específicas para o RAG
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Ferramentas do nosso projeto
from .models import Decision, Verdict
from .llm_builder import build_llm
from config.llm_config import LLM_CONFIG
from .agent import AGENT_PROMPT

JUDGE_PROMPT = ChatPromptTemplate.from_template(
    """Você é um Juiz académico, especialista em Teorias das Relações Internacionais. Sua tarefa é analisar a decisão de um ator e classificá-la de acordo com as teorias clássicas, usando os trechos do manual académico fornecido como sua principal referência.

# MANUAL DE REFERÊNCIA:
{retrieved_context}

# DECISÃO DO ATOR PARA ANÁLISE:
- Ação Principal: {action}
- Justificativa do Ator: {justification}
- Ação no Conselho Global: {council_action}

# SUA MISSÃO:
Com base EXCLUSIVAMENTE nos trechos do manual e na decisão do ator, forneça um veredito.
1.  **Verdict:** Classifique a decisão como "Neorrealismo", "Neoliberalismo" ou "Construtivismo".
    -   **Neorrealismo:** Foco no poder relativo, segurança, autoajuda, desconfiança em instituições.
    -   **Neoliberalismo:** Foco na cooperação, ganhos mútuos, instituições para reduzir incerteza e custos.
    -   **Construtivismo:** Foco em normas, identidade, legitimidade, reputação, e o que é considerado "apropriado".
2.  **Rationale:** Escreva uma justificativa clara para o seu veredito, explicando como a ação e a justificativa do ator se alinham com os conceitos apresentados no manual de referência.

# SCHEMA JSON OBRIGATÓRIO:
{schema}
"""
)

class Judge:
    """
    O Juiz da simulação. Carrega um manual de RI, cria uma base de conhecimento (RAG)
    e avalia as decisões dos agentes.
    """
    def __init__(self, pdf_path: str):
        print("\n⚖️  Inicializando o Juiz...")
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Arquivo do manual não encontrado em: {pdf_path}")

        # Configura o LLM do Juiz
        judge_config = LLM_CONFIG["juiz"]
        self.llm = build_llm(
            provider=judge_config["provider"],
            model=judge_config["model"],
            structured_output_model=Verdict
        )

        # Configura e constrói o pipeline RAG
        self._setup_rag_pipeline(pdf_path)
        print("✅ Juiz pronto e base de conhecimento carregada.")

    def _setup_rag_pipeline(self, pdf_path: str):
        """Carrega o PDF, divide em pedaços, cria embeddings e armazena em um vector store."""
        print("   - Carregando o manual de RI...")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        print(f"   - Dividindo o manual em {len(docs)} páginas/pedaços...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        print("   - Criando embeddings (isso pode levar um momento)...")
        # Usando um modelo de embedding local e gratuito para evitar custos de API
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        print("   - Criando o banco de dados vetorial (FAISS)...")
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        self.retriever = vectorstore.as_retriever()

        # Define a cadeia RAG final
        self.rag_chain = (
            {
                "retrieved_context": self.retriever,
                "action": RunnablePassthrough(),
                "justification": RunnablePassthrough(),
                "council_action": RunnablePassthrough(),
                "schema": RunnablePassthrough()
             }
            | AGENT_PROMPT
            | self.llm
        )

        def format_docs(docs):
            """Função auxiliar para juntar o conteúdo dos documentos recuperados em um único texto."""
            return "\n\n---\n\n".join(doc.page_content for doc in docs)

        self.rag_chain = (
            {
                # O 'retriever' agora é alimentado especificamente pela 'justification'.
                # A sua saída (lista de documentos) é então formatada para texto pela 'format_docs'.
                "retrieved_context": itemgetter("justification") | self.retriever | format_docs,
                # Usamos 'itemgetter' para passar os outros valores diretamente.
                "action": itemgetter("action"),
                "justification": itemgetter("justification"),
                "council_action": itemgetter("council_action"),
                "schema": itemgetter("schema")
             }
            | JUDGE_PROMPT
            | self.llm
        )

    def evaluate(self, decision: Decision) -> Verdict:
        print(f"⚖️  Juiz avaliando a decisão...")

        # O dicionário de invocação permanece o mesmo
        response = self.rag_chain.invoke({
            "action": decision.action_primary,
            "justification": decision.justification_text,
            "council_action": decision.council_action or "Nenhuma",
            "schema": json.dumps(Verdict.model_json_schema(), ensure_ascii=False, indent=2)
        })
        return response


   