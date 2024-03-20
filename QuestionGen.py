from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import PyPDF2

# Load the OpenAI model
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=3000)

# Prompt template for generating multiple-choice questions
template = """
Text: {text}
You are an expert MCQ maker. Given the above text, it is your job to\
create a quiz of {number} multiple choice questions which are of minimum difficulty {grade} out of 10 in {tone} tone.
Make sure that questions are not repeated and check all the questions to be conforming to the text as well.
Make sure to format your response like the RESPONSE_JSON below and use it as a guide.\
Ensure to make the {number} MCQs. Don't stop until the required number of questions is generated.
Make one third of the questions easy, one third of the questions medium difficulty
and length, and make a third of the questions long to read and hard to understand. Thus make the questions in a hard entrance exam format."
### RESPONSE_JSON
{response_json}
"""
quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "grade", "tone", "response_json"],
    template=template,
)

# LLMChain for generating multiple-choice questions
quiz_chain = LLMChain(
    llm=llm, prompt=quiz_generation_prompt, output_key="quiz", verbose=True
)

# Function to parse a PDF file and extract its text
def parse_pdf(file_path):
    with open(file_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

# Function to generate multiple-choice questions from a PDF file
def generate_questions_from_pdf(pdf_file_path, number, grade, tone):
    try:
        text = parse_pdf(pdf_file_path)
        response = quiz_chain(
            {
                "text": text,
                "number": number,
                "grade": grade,
                "tone": tone,
                "response_json": None,  # Placeholder for JSON data (not used)
            }
        )
        if isinstance(response, dict):
            quiz = response.get("quiz", None)
            if quiz is not None:
                return quiz
            else:
                return "Error: Quiz not generated"
        else:
            return response
    except Exception as e:
        return f"Error: {str(e)}"

# Example usage:
if __name__ == "__main__":
    pdf_file_path = r"C:\Users\joshu\Downloads\Various Indian Languages and Literature Notes.pdf"
    number_of_questions = 6
    grade_level = 7
    quiz_tone = "curious"

    questions = generate_questions_from_pdf(pdf_file_path, number_of_questions, grade_level, quiz_tone)
    print(questions)
