import streamlit as st 
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
import os
import openai
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SimpleSequentialChain
from typing import Any, List
from pydantic import BaseModel
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.openai_functions.utils import (
    _convert_schema,
    _resolve_schema_references,
    get_llm_kwargs,
)
from langchain.output_parsers.openai_functions import (
    JsonKeyOutputFunctionsParser,
    PydanticAttrOutputFunctionsParser,
)
from langchain.prompts import ChatPromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType



openai_api_key = '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'

summarization_template = """You will be provided with an article text. Your task is to summarize the article text.
Text:
{input}
"""

translation_template = """You will be provided with an article text. Your task is to translate the article text to English as close as possible. 
Text:
{input}
"""

tag_extraction_template = """First: 1. Look for athletes names in the summary. These will be names of people that play in an event or a sport for their national teams. If you don't see an athlete name and there is only mention of the national team, please write: "No Athlete Mentioned".
Then: 2. Look for the country that the athlete is playing for. Please only return countries that are participating in an event. When you find a country, output as noc the three-letter International Olympic Committe acronym for that country. If you don't see a country participating in an event, please write: "No NOC Participating".
Next: 3: Look for any mention of mental health and synonyms. I would like one tag that is binary and another tag with the words mentioned. Example: mental_health_binary = 1, mental_health = anxiety.
Finally: 4. Look for the discipline the athletes participate in. This is the most important tag.

Finally, please tell me why you chose thos specific tags.
Passage:
{input}
"""


#remove the prompt above Then, to return it to the first version
st.set_page_config(page_title = "Extract Tags from Articles", page_icon = ":robot:") #renames the title of the page in the browser
st.header("Extract Tags from Articles")

col1, col2 = st.columns(2)
with col1:
    st.markdown("### ML Team, August 2023")
with col2: 
    st.image(image='/Users/olga/Desktop/tags_extraction/Olympic_rings_without_rims.svg.png', width= 100)


##col1, col2 = st.columns(2)

#with col1:
st.markdown("This app is going to extract the following tags from the article you input: athlete, noc, mental heatlh, and sport")

#with col2:
    #st.image(image='/Users/olga/Desktop/tags_extraction/Olympic_rings_without_rims.svg.png', width= 100)

st.markdown("## Enter your article here")

def get_text():
    input_text = st.text_area(label = "", placeholder = "Enter your article text here...", key = "article_input")
    return input_text

article_input = get_text()

st.markdown("### The tags extracted from your article:")


if article_input:
    llm = OpenAI(model_name="gpt-3.5-turbo-0613", temperature=0.2, openai_api_key=openai_api_key)

    prompt_summarization = PromptTemplate.from_template(template=summarization_template)
    chain_sum = LLMChain(llm=llm, prompt=prompt_summarization)
    prompt_translation = PromptTemplate.from_template(template=translation_template)
    chain_trans = LLMChain(llm=llm, prompt=prompt_translation)
    prompt_extraction = PromptTemplate.from_template(template=tag_extraction_template)
    chain_extr = LLMChain(llm=llm, prompt=prompt_extraction)
    

    overall_chain = SimpleSequentialChain(chains=[chain_sum, chain_trans, chain_extr], verbose=True)
    output = overall_chain.run(article_input)
    
    st.write(output)
 
    