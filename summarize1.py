

import os
from PyPDF2 import PdfReader
from typing_extensions import Concatenate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
import langchain
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import(
    AIMessage,
HumanMessage,
SystemMessage
)


openai_key="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"
os.environ["OPENAI_API_KEY"]=openai_key
llm=ChatOpenAI(model_name="gpt-3.5-turbo")
pdfreader=PdfReader('/Users/sreyaskv/Documents/Studybuddy/Examplematerials/satyagraha.pdf')
text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        text += content
docs=[Document(page_content=text)]
def summaryfunc(text):
    template = '''Write a summary of the following text covering all the important points.

    The text: `{text}`
    '''

    initial_prompt = PromptTemplate(
        input_variables=['text'],
        template=template
    )


    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
    chunks=text_splitter.create_documents([text])

    final_combine_prompt='''Provide a final summary of the following text covering all the important points and sprinkle it in with some jokes and then provide a funny and interesting title for the summary.Put the title at the top. Dont mention that you summarised this from a text
    text:`{text}`'''

    final_prompt = PromptTemplate(
        input_variables=['text'],
        template=final_combine_prompt
    )

    chain=load_summarize_chain(
        llm,
        chain_type='map_reduce',
        map_prompt=initial_prompt,
        combine_prompt=final_prompt,
        verbose=False
    )
    output_summary=chain.run(docs)
    return output_summary

