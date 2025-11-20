import os
import json
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel
from typing import Optional, Type, Any

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None

try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None

try:
    from langchain_community.chat_models import ChatMaritalk
except ImportError:
    ChatMaritalk = None

try:
    from langchain_deepseek import ChatDeepSeek
except ImportError:
    ChatDeepSeek = None

try:
    from langchain_xai import ChatXAI
except ImportError:
    ChatXAI = None

class FixEncodingJsonOutputParser(JsonOutputParser):
    """
    Um parser que tenta corrigir problemas comuns de codificação (mojibake)
    antes de analisar a string JSON.
    """
    def parse(self, text: str) -> Any:
        try:
            fixed_text = text.encode('latin-1').decode('utf-8')
            return super().parse(fixed_text)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return super().parse(text)
        except Exception as e:
            raise OutputParserException(f"Falha ao analisar a saída JSON após a tentativa de correção. Erro: {e}")

def build_llm(
    provider: str,
    model: str,
    temperature: float = 0.1,
    structured_output_model: Optional[Type[BaseModel]] = None
) -> Runnable:
    """
    Constrói e retorna um objeto de LLM da LangChain com base no provedor.
    Tenta usar with_structured_output e, se falhar, usa um parser com correção de codificação.
    """
    provider = provider.lower()
    llm = None
    
    force_parser_fallback = ["xai", "maritaca"] 
    
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"): raise ValueError("Chave de API OPENAI_API_KEY não encontrada no arquivo .env")
        if not ChatOpenAI: raise ImportError("langchain-openai não está instalado.")
        llm = ChatOpenAI(model=model, temperature=temperature)
    elif provider == "groq":
        if not os.getenv("GROQ_API_KEY"): raise ValueError("Chave de API GROQ_API_KEY não encontrada no arquivo .env")
        if not ChatGroq: raise ImportError("langchain-groq não está instalado.")
        llm = ChatGroq(model=model, temperature=temperature)
    elif provider == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"): raise ValueError("Chave de API ANTHROPIC_API_KEY não encontrada no .env")
        if not ChatAnthropic: raise ImportError("langchain-anthropic não está instalado.")
        llm = ChatAnthropic(model=model, temperature=temperature)
    elif provider == "maritaca":
        if not os.getenv("MARITACA_API_KEY"): raise ValueError("Chave de API MARITACA_API_KEY não encontrada no .env")
        if not ChatMaritalk: raise ImportError("langchain-community e maritalk não estão instalados.")
        llm = ChatMaritalk(model=model, api_key=os.getenv("MARITACA_API_KEY"))
    elif provider == "deepseek":
        if not os.getenv("DEEPSEEK_API_KEY"): raise ValueError("Chave de API DEEPSEEK_API_KEY não encontrada no .env")
        if not ChatDeepSeek: raise ImportError("langchain-deepseek não está instalado.")
        llm = ChatDeepSeek(model=model, temperature=temperature, api_key=os.getenv("DEEPSEEK_API_KEY"))
    elif provider == "xai":
        if not os.getenv("XAI_API_KEY"): raise ValueError("Chave de API XAI_API_KEY não encontrada no .env")
        if not ChatXAI: raise ImportError("langchain-xai não está instalado.")
        llm = ChatXAI(model=model, temperature=temperature, api_key=os.getenv("XAI_API_KEY"))
    else:
        raise ValueError(f"Provedor '{provider}' não é suportado.")


    if structured_output_model and llm:
        if provider in force_parser_fallback:
            print(f"   ⚠️  Aviso: Forçando o uso do parser de correção para o provedor '{provider}'.")
            parser = FixEncodingJsonOutputParser(pydantic_object=structured_output_model)
            return llm | parser

        try:
            return llm.with_structured_output(structured_output_model)
        except NotImplementedError:
            print(f"   ⚠️  Aviso: O provedor '{provider}' não suporta 'with_structured_output'. A usar o parser de correção como alternativa.")
            parser = FixEncodingJsonOutputParser(pydantic_object=structured_output_model)
            return llm | parser
    
    if llm:
        return llm
    
    raise RuntimeError("Falha fatal ao construir o LLM.")