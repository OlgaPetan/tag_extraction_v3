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



#openai_api_key = ''

summarization_template = """You will be provided with an article text. Keep all athletes and names you find in the original article text.
Text:
{input}
"""

translation_template = """You will be provided with an article text. Your task is to translate the article text to English as close as possible. 
Text:
{input}
"""

tag_extraction_template = """First: 1. Look for athletes names in the summary. These will be names of people that play a sport for their national teams or clubs. Example: If the article says Chinese athlete or People's republic of China, output CHN.
Then: 2. Look for the country that the athlete is playing for. Please only return countries that are participating in an event. When you find a country, output as noc the three-letter International Olympic Committe acronym for that country.
Next: 3: Look for any mention of mental health and synonyms. I would like one tag that is binary and another tag with the words mentioned. Example: mental_health_binary = 1, mental_health = anxiety.
Then: 4. Look for the discipline the athletes participate in. This is the most important tag.
Next: 5. Please extract the following topics and make them binary:
         1. Blast from the past: please put 1 in this topic if the article talks about events from the past and 0 otherwise
         2. Before they were stars: please put 1 in this topic if the article describes how athletes were like before they became famous and 0 otherwise
         3. Classic finals: please put 1 in this topic if the article describes a final match between teams leading national teams or teams that were important in that period for that event and 0 otherwise
         4. Incredible teams: please put 1 in this topic if the article describes a team that had to overcome obstacles during the preparation for an event or during the event and ended up winning and 0 otherwise
         5. Live blog: please put 1 in this topic if the article is a live blog and 0 otherwise
         6. First medal: please put 1 in this topic if the article talks about an athlete or a national team winning their first medal in that event or in general and 0 otherwise
         7. Day in the life: please put 1 in this topic if the article talks about what an athlete does during a normal day, their routine, or a day in their life and 0 otherwise

Next to each tag, please output a number informing me how many times you saw that tag being mentioned in the original article and not the summarization. For example, how many time was that NOC mentioned, how many times was that athlete mentioned, etc. The athlete's first or last name could be missing but count it anyway. 
Example: Ailing (Eileen) Gu: 3 mentions, Cassie Sharpe: 2 mentions, CHN: 3 mentions


Finally, please tell me why you chose those specific tags.
Passage:
{input}
"""
 
st.set_page_config(page_title = "Extract Tags from Articles", page_icon = ":robot:") #renames the title of the page in the browser
st.header("Extract Tags from Articles")

col1, col2 = st.columns(2)
with col1:
    st.markdown("### ML Team, August 2023")
with col2: 
    st.image(image='Olympic_rings_without_rims.svg.png', width= 100)


##col1, col2 = st.columns(2)

#with col1:
st.markdown("This app is going to extract the following tags from the article you input: athlete, noc, mental heatlh, and sport")

#with col2:
    #st.image(image='/Users/olga/Desktop/tags_extraction/Olympic_rings_without_rims.svg.png', width= 100)

st.markdown("## Enter your article here")

def get_api_key():
    input_text = st.text_input(label="OpenAI API Key ",  placeholder="Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input")
    return input_text

openai_api_key = get_api_key()

def get_text():
    input_text = st.text_area(label = "", placeholder = "Enter your article text here...", key = "article_input")
    return input_text

article_input = get_text()

st.markdown("### The tags extracted from your article:")

if article_input:
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0613", temperature=0.2, openai_api_key=openai_api_key)

    prompt_summarization = PromptTemplate(input_variables=["input"], template=summarization_template)
    chain_sum = LLMChain(llm=llm, prompt=prompt_summarization)
    prompt_translation = PromptTemplate(input_variables=["input"], template=translation_template)
    chain_trans = LLMChain(llm=llm, prompt=prompt_translation)
    prompt_extraction = PromptTemplate(input_variables=["input"], template=tag_extraction_template)
    chain_extr = LLMChain(llm=llm, prompt=prompt_extraction)

    overall_chain = SimpleSequentialChain(chains=[chain_sum, chain_trans, chain_extr], verbose=True)
    output = overall_chain.run({article_input})

    st.write(output)
