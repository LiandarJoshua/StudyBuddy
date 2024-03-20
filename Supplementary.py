from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Load the OpenAI model
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=3000)

# Prompt template for providing study materials
template = """
You are an educational assistant. Given the topic "{topic}", please provide study materials including YouTube video links, research paper links, and online article links to assist the student in further studies.
### RESPONSE_JSON
{response_json}
"""
study_materials_prompt = PromptTemplate(
    input_variables=["topic", "response_json"],
    template=template,
)

# LLMChain for providing study materials
study_materials_chain = LLMChain(
    llm=llm, prompt=study_materials_prompt, output_key="study_materials", verbose=True
)

# Function to generate study materials for a given topic
def provide_study_materials(topic):
    try:
        response = study_materials_chain(
            {
                "topic": topic,
                "response_json": None,  # Placeholder for JSON data (not used)
            }
        )
        if isinstance(response, dict):
            study_materials = response.get("study_materials", None)
            if study_materials is not None:
                return study_materials
            else:
                return "Error: Study materials not found"
        else:
            return response
    except Exception as e:
        return f"Error: {str(e)}"

# Example usage:
if __name__ == "__main__":
    topic = input("Enter topic you'd like materials on:")

    study_materials = provide_study_materials(topic)
    print(study_materials)
