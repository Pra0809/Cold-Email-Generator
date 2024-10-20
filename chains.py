
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv
load_dotenv()

class Chain:
    def __init__(self):
     self.llm = ChatGroq(temperature=0, groq_api_key='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', model_name="llama-3.1-70b-versatile")

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """
        ### JOB DESCRIPTION:
        {job_description}
        
        ### INSTRUCTION:
        You are Prashant Chandra, a Business Development Executive at AlgoSphere, an AI & Software Consulting 
        company specializing in optimizing business processes through automation. Your firm helps enterprises 
        enhance scalability, streamline operations, reduce costs, and improve overall efficiency through tailored 
        solutions.
        Write a cold email to a prospective client for the job mentioned earlier  explaining how AlgoSphere can meet their needs. 
        Mention the company's expertise in delivering customized data analytics and automation solutions, and 
        highlight the most relevant projects from AlgoSphere's portfolio using the links provided ({link_list}). 
        Do not provide a preamble.
        ### EMAIL (NO PREAMBLE):
        
        """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content

if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))