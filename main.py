import warnings
warnings.filterwarnings('ignore')

from crewai import Task, Agent, Crew, Process
from langchain_groq import ChatGroq


groq_llm = ChatGroq(
    groq_api_key=os.environ["GROQ_API_KEY"],  # or rely on env var
    model_name="llama3-70b-8192",      # or "mixtral-8x7b-32768"
    temperature=0.6
)



# --- Agents ---
symptom_collector = Agent(
    role="Symptom Collector",
    goal="Gather patient symptoms and medical history clearly.",
    backstory=(
        "You are the first contact with patients. "
        "You ask clarifying questions and gather details on symptoms, "
        "medical history, and lifestyle factors."
    ),
    verbose=True,
    llm=groq_llm
)

diagnosis_agent = Agent(
    role="Diagnosis Specialist",
    goal="Analyze symptoms and suggest possible conditions.",
    backstory=(
        "You are a diagnostic expert trained on medical literature and guidelines. "
        "You carefully analyze symptoms to list possible conditions."
    ),
    verbose=True,
    llm=groq_llm
)

treatment_agent = Agent(
    role="Treatment Planner",
    goal="Recommend safe treatment and lifestyle options.",
    backstory=(
        "You are a medical consultant who suggests safe, guideline-based treatment options. "
        "You always provide lifestyle and follow-up care advice alongside medication guidance."
    ),
    verbose=True,
    llm=groq_llm
)

# --- Tasks ---
symptom_task = Task(
    description=(
        "Collect and structure the patient's input symptoms and history: {patient_input}. "
        "Make sure the final output is structured and detailed."
    ),
    expected_output="A structured list of symptoms, history, and lifestyle details.",
    agent=symptom_collector
)

diagnosis_task = Task(
    description=(
        "Based on the collected symptoms, analyze and suggest possible conditions. "
        "Explain your reasoning briefly."
    ),
    expected_output="A list of 2-5 possible conditions with explanation.",
    agent=diagnosis_agent
)

treatment_task = Task(
    description=(
        "Based on the possible conditions, suggest safe treatment options, "
        "including medications (general categories only, no dosages), lifestyle adjustments, "
        "and follow-up advice."
    ),
    expected_output="A treatment recommendation with medications categories, lifestyle changes, and follow-up.",
    agent=treatment_agent
)

# --- Crew ---
crew = Crew(
    agents=[symptom_collector, diagnosis_agent, treatment_agent],
    tasks=[symptom_task, diagnosis_task, treatment_task],
    process=Process.sequential
)
