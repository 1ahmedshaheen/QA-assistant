from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda, RunnableAssign

from app.utils.llm import get_llm
from loguru import logger

class DocumentSummaryBase(BaseModel):
    running_summary: str = Field("", description="Running description of the document. Do not override; only update!")
    main_ideas: List[str] = Field([], description="Most important information from the document (max 3)")
    loose_ends: List[str] = Field([], description="Open questions that would be good to incorporate into summary (max 3)")

def build_extraction_chain():
    """Returns a chain that populates a DocumentSummaryBase via slot-filling."""
    llm = get_llm()
    parser = PydanticOutputParser(pydantic_object=DocumentSummaryBase)
    
    prompt = ChatPromptTemplate.from_template(
        "You are generating a running summary of a document portion. Make it readable by a technical user.\n"
        "Keep it short, but as dense and useful as possible!\n"
        "Don't repeat the same response.\n"
        "Current Fact Sheet: {info_base}\n\n"
        "{format_instructions}\n"
        "Update the fact sheet with the following new portion: {input}"
    )
    
    def preparse(text: str):
        # Local LLMs sometimes wrap JSON in code blocks or omit braces
        text = text.strip()
        if "```json" in text:
            text = text.split("```json")[-1].split("```")[0]
        if "{" not in text: text = "{" + text
        if "}" not in text: text = text + "}"
        return text

    chain = (
        RunnableAssign({"format_instructions": lambda _: parser.get_format_instructions()})
        | prompt 
        | llm 
        | RunnableLambda(lambda x: preparse(x.content) if hasattr(x, "content") else preparse(str(x)))
        | parser
    )
    return chain

def summarize_documents(docs: List) -> DocumentSummaryBase:
    """Iteratively updates a fact sheet across a list of document chunks."""
    chain = build_extraction_chain()
    state = DocumentSummaryBase()
    
    for i, doc in enumerate(docs):
        try:
            state = chain.invoke({"input": doc.page_content, "info_base": state.dict()})
            logger.info(f"[Summarizer] Processed chunk {i+1}/{len(docs)}")
        except Exception as e:
            logger.warning(f"[Summarizer] Failed to update on chunk {i}: {e}")
            continue
            
    return state
