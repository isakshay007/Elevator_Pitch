import streamlit as st
from lyzr_automata.ai_models.openai import OpenAIModel
from lyzr_automata import Agent, Task
from lyzr_automata.pipelines.linear_sync_pipeline import LinearSyncPipeline
from PIL import Image
from lyzr_automata.tasks.task_literals import InputType, OutputType
import os

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["apikey"]

st.markdown(
    """
    <style>
    .app-header { visibility: hidden; }
    .css-18e3th9 { padding-top: 0; padding-bottom: 0; }
    .css-1d391kg { padding-top: 1rem; padding-right: 1rem; padding-bottom: 1rem; padding-left: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

image = Image.open("./logo/lyzr-logo.png")
st.image(image, width=150)

# App title and introduction
st.title("LinkedIn Elevator Pitch Generator👨‍💻")
st.markdown("Welcome to the LinkedIn Elevator Pitch Generator! Craft a compelling elevator pitch tailored to your professional profile with expert guidance")

input = st.text_input("Please enter your name, professional role, experience, and any notable achievements. Feel free to include any other relevant details as well.",placeholder=f"""Type here""")

open_ai_text_completion_model = OpenAIModel(
    api_key=st.secrets["apikey"],
    parameters={
        "model": "gpt-4-turbo-preview",
        "temperature": 0.2,
        "max_tokens": 1500,
    },
)


def pitch_generation(input):
    generator_agent = Agent(
        role="LINKEDIN ELEVATOR PITCH expert",
        prompt_persona=f"Your task  is to CREATE a COMPELLING elevator pitch for the user, TAILORED to their provided details such as NAME, PROFESSIONAL ROLE, INDUSTRY, EXPERIENCE, KEY SKILLS, NOTABLE ACHIEVEMENTS, PASSIONS, and GOALS."
    )

    prompt = f"""
 You are an Expert LINKEDIN ELEVATOR PITCH CONSULTANT. Always introduce yourself. Your task is to CRAFT a CONCISE yet POWERFUL elevator pitch for the user, carefully TAILORED to their specific details such as NAME, PROFESSIONAL ROLE, INDUSTRY, EXPERIENCE, KEY SKILLS, NOTABLE ACHIEVEMENTS, PASSIONS, and GOALS.

 To CREATE an EFFECTIVE LinkedIn elevator pitch, follow these steps:

 1. COLLECT all necessary details from the user. If any information is missing or unclear, ASK for clarification or make EDUCATED ASSUMPTIONS based on industry standards.

 2. IDENTIFY and EMPHASIZE KEYWORDS related to their role and industry which will make the pitch STAND OUT.

 3. START the paragraph with an INTRODUCTION that immediately captures attention by mentioning the user's NAME and CURRENT JOB TITLE.

 4. INCORPORATE their EXPERIENCE and SKILLS along with any UNIQUE ACHIEVEMENTS that highlight their distinct value in the field.

 5. EXPRESS their PASSIONS and AMBITIONS in a way that aligns with their CAREER GOALS to demonstrate both personality and direction.

 6. CONCLUDE the paragraph with a STRONG closing statement that succinctly encapsulates their professional essence in 50-120 words, leaving a MEMORABLE impression and with a Let's connect sentence.

 Furthermore, OFFER advice on how they can UPDATE and IMPROVE their elevator pitch as they progress professionally.

 Remember: You MUST use your EXPERTISE to ensure each word contributes to an IMPACTFUL narrative within this brief format.



      """

    generator_agent_task = Task(
        name="Pitch Generation",
        model=open_ai_text_completion_model,
        agent=generator_agent,
        instructions=prompt,
        default_input=input,
        output_type=OutputType.TEXT,
        input_type=InputType.TEXT,
    ).execute()

    return generator_agent_task 
   
if st.button("Generate!"):
    solution = pitch_generation(input)
    st.markdown(solution)

with st.expander("ℹ️ - About this App"):
    st.markdown("""
    This app uses Lyzr Automata Agent Optimize your code. For any inquiries or issues, please contact Lyzr.

    """)
    st.link_button("Lyzr", url='https://www.lyzr.ai/', use_container_width=True)
    st.link_button("Book a Demo", url='https://www.lyzr.ai/book-demo/', use_container_width=True)
    st.link_button("Discord", url='https://discord.gg/nm7zSyEFA2', use_container_width=True)
    st.link_button("Slack",
                   url='https://join.slack.com/t/genaiforenterprise/shared_invite/zt-2a7fr38f7-_QDOY1W1WSlSiYNAEncLGw',
                   use_container_width=True)