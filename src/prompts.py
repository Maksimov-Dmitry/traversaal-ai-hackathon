RAG_SYSTEM_PROMPT = "You are a helpful assistant, who recommends the hotels based only on my preferences."

RAG_CONTEXT_TEMPLATE = """
    {id}: {hotel_name}
    {description}
"""

RAG_USER_PROMPT = """
    Here are the information about most relevant hotels to my query
    ---------------------
    {context}
    ---------------------
    Present these results to me and justify the ranking (explain why a hotel matches my preferences). Don't draw ANY conclusion and don't based on own knowledge.
    Query: {query}
    Answer: 
"""

AGENT_USER_PROMPT = """
    Answer the following question as best you can. You have access to the following tools:

    hotels_data_base: A tool which present information about most relevant hotels based on the query. The information contains pros and cons of the hotel based on reviews, reviews ratings and ammenities. It is usefull when user want to get hotels recommendations. In this case Action Input should be query which will be complete and usefull to retrive the most relevant hotels.
    ares_api: An API which performs real-time internet searches. It can be usefull than you need specific information about the hotel or the locataion or smth else from the internet. In this case Action Input should be query which will be complete and usefull to retrive the information from the Internet.
    nothing: If you are sure you can answer the user's query without additional tools. In this case Action Input should be just an answer.

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [hotels_data_base, ares_api, nothing]
    Action Input: the input to the action

    Begin!

    Question: {input}
    Thought: 
"""

AGENT_SYSTEM_PROMPT = "You are a helpful assistant for a hotel recommendation system based on my preferences. Answer all questions to the best of your ability."

REVIEWS_SYSTEM_PROMPT = "You are a helpful assistant. Your goal is to underpin the strong and the weak points (features, amenities). If you can't find strong or weak points, don't write ANYTHING about them. The information consists of hotel reviews, i.e. Title of the review and the Review itself."
REVIEWS_USER_PROMPT = """{text} Good Example:
    ### Strong Points:
    - The hotel boasts a favorable location with sea views and proximity to Zeitinburnu train station.
    - Upgraded rooms, fitness facilities, and the outdoor pool area are well-received.
    - The staff, including specific individuals like Mr. Levent, Cihan, and Buse, have been commended for their service.
    - Room cleanliness is frequently mentioned as a positive aspect.

    ### Weak Points:
    - Inconsistency in customer service, with some guests reporting a lack of assistance with luggage and unfriendly reception.
    - Miscommunication regarding room rates and issues with overcharges.
    - Some guests have found the hotel's amenities, such as the narrow balcony and the pool's restrictive rules, to be lacking.
    - A few guests reported cleanliness issues in the bathroom and concerns with room repairs.
"""

TRAVERSIALAI_USER_PROMPT = """
    Based on the information retrived from the internet, answer the following question as best you can.
    ---------------------
    {context}
    ---------------------
    Query: {query}
    Answer: 
"""
