# -*- coding: utf-8 -*-

import json
from datetime import datetime, timedelta
from tqdm import tqdm

# =====================================================================================
# --------------------------------- 【 expert configuration 】 ---------------------------------
#         As a data expert, I recommend tuning these parameters for optimal results.
# =====================================================================================

# 1. File Paths
INPUT_FILE = "chatdata_text.json"
OUTPUT_FILE = "ft_dataset_multi.jsonl"

# 2. Session Definition
# How long of a silence indicates a new conversation has started?
# A good default is 30-60 minutes.
SESSION_BREAK_MINUTES = 10

# 3. Message Grouping
# How long between two messages from the same person to be considered a single "turn"?
CONTINUATION_WINDOW_SECONDS = 45 # Increased slightly for more tolerance

# 4. Data Point Filtering
# To ensure quality, we can filter out very short or trivial sessions.
MIN_SESSION_MESSAGES = 3 # A session must have at least 3 messages to be useful.
MIN_CONTEXT_TURNS = 1    # A training example must have at least 1 turn of history.
MAX_SINGLE_MESSAGE_LENGTH = 30 # A very long message is likely to be meaningless.

# 5. Special Tokens
MESSAGE_SPLIT_TOKEN = "<|split|>"

# =====================================================================================
# --------------------------------- 【 script logic 】 ---------------------------------
# =====================================================================================

def parse_time(time_str):
    """Handles various timestamp formats."""
    try:
        return datetime.fromisoformat(time_str.replace('Z', '+00:00'))
    except ValueError:
        return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")

def split_into_sessions(messages, break_minutes):
    """Splits the entire message history into distinct conversation sessions."""
    if not messages:
        return []

    sessions = []
    current_session = [messages[0]]
    
    for i in range(1, len(messages)):
        time_delta = messages[i]['time'] - messages[i-1]['time']
        if time_delta > timedelta(minutes=break_minutes):
            # If a break is detected, save the old session and start a new one
            if len(current_session) >= MIN_SESSION_MESSAGES:
                sessions.append(current_session)
            current_session = []
        current_session.append(messages[i])
    
    # Add the last session if it's valid
    if len(current_session) >= MIN_SESSION_MESSAGES:
        sessions.append(current_session)
        
    return sessions

def _group_messages(message_list):
    """Helper function to group consecutive messages from the same sender."""
    if not message_list:
        return []
        
    grouped = []
    current_role = "user" if message_list[0]['is_self'] else "assistant"
    current_content = []

    for msg in message_list:
        role = "user" if msg['is_self'] else "assistant"
        if role == current_role:
            current_content.append(msg['content'])
        else:
            # Role changed, save the previous block
            separator = '\n' if current_role == 'user' else MESSAGE_SPLIT_TOKEN
            grouped.append({"role": current_role, "content": separator.join(current_content)})
            # Start a new block
            current_role = role
            current_content = [msg['content']]
    
    # Add the final block
    separator = '\n' if current_role == 'user' else MESSAGE_SPLIT_TOKEN
    grouped.append({"role": current_role, "content": separator.join(current_content)})
    
    return grouped

def create_multi_turn_dataset():
    """Main function to generate the multi-turn dataset from sessions."""
    print("="*50)
    print("Starting session-based multi-turn dataset creation...")
    
    # 1. Load and pre-process all messages
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        all_messages = json.load(f)
    for msg in all_messages:
        msg['time'] = parse_time(msg['time'])
    all_messages.sort(key=lambda x: x['time'])

    original_count = len(all_messages)
    all_messages = [
        msg for msg in all_messages 
        if len(msg.get('content', '')) <= MAX_SINGLE_MESSAGE_LENGTH
    ]
    print(f"Filtered out {original_count - len(all_messages)} overly long messages.")
    
    # 2. Split into conversation sessions
    print(f"\n[Step 1/3] Splitting {len(all_messages)} messages into sessions...")
    sessions = split_into_sessions(all_messages, SESSION_BREAK_MINUTES)
    print(f"Found {len(sessions)} valid conversation sessions.")

    # 3. Generate training examples from each session
    print("\n[Step 2/3] Generating training examples from sessions...")
    training_examples = []
    for session in tqdm(sessions, desc="Processing Sessions"):
        # Iterate through messages in the session to find assistant responses
        for i in range(1, len(session)):
            # Find the start of an assistant's turn
            if not session[i]['is_self'] and session[i-1]['is_self']:
                # The context is all messages in the session up to this point
                context_messages = session[:i]
                
                # The target is this assistant message and any continuations
                assistant_block = [session[i]]
                j = i + 1
                while j < len(session) and not session[j]['is_self']:
                    time_delta = session[j]['time'] - session[j-1]['time']
                    if time_delta.total_seconds() <= CONTINUATION_WINDOW_SECONDS:
                        assistant_block.append(session[j])
                        j += 1
                    else:
                        break
                
                # Combine context and target into one history
                full_turn_messages = context_messages + assistant_block
                
                # Group the raw messages into clean user/assistant turns
                grouped_messages = _group_messages(full_turn_messages)

                # Apply final quality filter
                if len(grouped_messages) >= (MIN_CONTEXT_TURNS * 2): # user/asst pair
                    training_examples.append({"messages": grouped_messages})

    print(f"Generated {len(training_examples)} high-quality training examples.")
    
    # 4. Save to output file
    print(f"\n[Step 3/3] Saving dataset to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for example in training_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    print("\n" + "="*50)
    print("🎉 Dataset creation successful!")
    print(f"File saved at: {OUTPUT_FILE}")
    print("="*50)

if __name__ == "__main__":
    create_multi_turn_dataset()