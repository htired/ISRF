
seq_templates = [
    'Given the following purchase history of user_{}: item_{}, predict next possible item to be purchased by the user.',
    'I find the purchase history list of user_{}: item_{}. I wonder what is the next item to recommend to the user. Can you help me decide?',
    'Here is the purchase history list of user_{}: item_{}. Try to recommend next item to the user.',
    'Given the following purchase history of user_{}: item_{}, predict next possible item for the user.',
    'Based on the purchase history of user_{}: item_{}, can you decide the next item likely to be purchased by the user?',
    'Here is the purchase history of user_{}: item_{}. What to recommend next for the user?',
    'According to the purchase history of user_{}: item_{}, can you recommend the next possible item to the user?',
    'user_{} item_{}',
]
seq_early_templates = [
    "Looking at user_{}'s interaction history: item_{}, which item do you think they engaged with first?",
    # "User_{} has interacted with these items: item_{}. Can you guess which one was among their earliest interactions?",
    # "Here's what user_{} has interacted with: item_{}. What do you think was one of their first interactions?",
    # "Based on user_{}'s past interactions: item_{}, which item seems like an early favorite?",
    # "Given user_{}'s interaction sequence: item_{}, which item stands out as one of the earliest?",
    # "User_{}'s engagement history is as follows: item_{}. Which item do you think they encountered first?",
    # "Looking at user_{}'s activity: item_{}, can you pinpoint an item they engaged with early on?",
]
topn_templates = [
    'Which item of the following to recommend for user_{}? item_{}',
    'Choose the best item from the candidates to recommend for user_{}? item_{}',
    'Pick the most suitable item from the following list and recommend to user_{}: item_{}',
    'We want to make recommendation for user_{}. Select the best item from these candidates: item_{}',
    'user_{} item_{}',
]

exp_templates = [
    'Generate an explanation for user_{} about this product: item_{}',
    'Can you help generate an explanation of user_{} for item_{}?',
    'Help user_{} generate an explanation about this product: item_{}',
    'Generate user_{}\'s purchase explanation about item_{}',
    'Help user_{} generate an explanation for item_{}',
    'Can you help generate an explanation for user_{} about the product: item_{}',
    'Write an explanation for user_{} about item_{}',
    'Generate an explanation for user_{} about item_{}',
    'user_{} item_{}',
]

base_prompts = [
    "Step 1: Summarize the main categories of products the user has recently browsed.",
    "Step 2: Analyze these categories and the browsing sequence to identify stages of interest shift.",
    "Step 3: Determine the current focus of interest based on recent browsing, and summarize the user's present preferences.",
    "Step 4: Based on the above analysis, infer any common needs the user may not have satisfied yet.",
    "Step 5: Summarize the user's brand or product style preferences based on browsing history.",
    "Step 6: Integrate all the information to identify the next best product category to recommend, and explain your reasoning.",
    "Step 7: Provide the final recommended product and explain your reasoning in detail."
]
