from autogen import ConversableAgent, GroupChat, GroupChatManager

# set your OpenAI API key here
openai_api_key = ""

intro_agent = ConversableAgent(
    name="Intro_Agent",
    system_message="You are the Conversation Agent. Your role is to initiate the conversation, gather initial details "
                   "from the user, and then let appropriate specialized agents respond based on the "
                   "user's responses. Ensure a supportive and empathetic tone throughout the interaction. When the "
                   "user's initial needs are identified, route the conversation to the relevant agent.",
    llm_config={"config_list": [{"model": "gpt-4", "api_key": openai_api_key}]},
)

mood_agent = ConversableAgent(
    name="Mood_Agent",
    system_message="You are the Mood Tracking Agent. Your role is to assess the user's current mood and emotional "
                   "state using predefined scales and questions. You should ask the user to rate their mood and "
                   "provide empathetic responses based on their input. After assessing the mood, summarize the "
                   "findings and route the conversation to the next appropriate agent.",
    llm_config={"config_list": [{"model": "gpt-4", "api_key": openai_api_key}]},
)

assessment_agent = ConversableAgent(
    name="Assessment_Agent",
    system_message="You are the Assessment Agent. Your role is to conduct a detailed assessment of the user's "
                   "symptoms and behaviors."
                   "You will ask a series of questions to understand the user's emotional and mental health status, "
                   "particularly"
                   "focusing on symptoms of anxiety, depression and substance use. Provide supportive feedback based "
                   "on the"
                   "user's responses",
    llm_config={"config_list": [{"model": "gpt-4", "api_key": openai_api_key}]},
)

therapy_agent = ConversableAgent(
    name="Therapy_Agent",
    system_message="You are the Therapeutic Agent. Your role is to offer immediate therapeutic interventions, "
                   "such as breathing exercises or mindfulness techniques or cognitive behavioural activities, "
                   "to help manage the user's stress and"
                   "improve their mood. Provide clear, step-by-step instructions for the exercises and encourage the "
                   "user throughout the process. After the intervention, offer additional support or resources",
    llm_config={"config_list": [{"model": "gpt-4", "api_key": openai_api_key}]},
)

crisis_mgmt_agent = ConversableAgent(
    name="Crisis_Management_Agent",
    system_message="You are the Crisis Management Agent. Your role is to handle situations where the user may be in "
                   "immediate danger or requires urgent support due to their substance use or emotional state. Assess "
                   "the severity of the crisis and provide appropriate responses, including routing to human support "
                   "if necessary. Maintain a calm and reassuring tone, ensuring the user's safety is prioritized.",
    llm_config={"config_list": [{"model": "gpt-4", "api_key": openai_api_key}]},
)


resource_recommendation_agent = ConversableAgent(
    name="Resource_Recommendation_Agent",
    system_message="You are the Resource Recommendation Agent. Your role is to provide the user with personalized "
                   "recommendations for articles, videos, and other resources that can help them manage their mental "
                   "health and substance use. Tailor your recommendations based on the user's expressed needs and "
                   "preferences. After providing resources, prompt the user to provide feedback on the conversation "
                   "and close the conversation with a supportive message.",
    llm_config={"config_list": [{"model": "gpt-4", "api_key": openai_api_key}]},
)

human_proxy = ConversableAgent(
    name="human_proxy",
    llm_config=False,  # no LLM used for human proxy
    human_input_mode="ALWAYS",  # always ask for human input
)


allowed_transitions = {
    intro_agent: [human_proxy],
    human_proxy: [intro_agent, mood_agent, assessment_agent, therapy_agent, crisis_mgmt_agent, resource_recommendation_agent],
    mood_agent: [human_proxy],
    assessment_agent: [human_proxy],
    therapy_agent: [human_proxy],
    crisis_mgmt_agent: [human_proxy],
    resource_recommendation_agent: [human_proxy]
}

constrained_graph_chat = GroupChat(
    agents=[intro_agent, mood_agent, assessment_agent, therapy_agent, crisis_mgmt_agent, resource_recommendation_agent, human_proxy],
    allowed_or_disallowed_speaker_transitions=allowed_transitions,
    speaker_transitions_type="allowed",
    messages=[],
    max_round=12,
    send_introductions=True,
)

constrained_group_chat_manager = GroupChatManager(
    groupchat=constrained_graph_chat,
    llm_config={"config_list": [{"model": "gpt-4", "api_key": openai_api_key}]},
)

chat_result = intro_agent.initiate_chat(
    constrained_group_chat_manager,
    message="Hi there, how are you doing today?",
    summary_method="reflection_with_llm",
)