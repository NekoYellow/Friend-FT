import json
from datetime import datetime, timedelta

# 1. Constants and Configuration
REPLY_WINDOW_MINUTES = 10
CONTINUATION_WINDOW_SECONDS = 30
INITIATION_WINDOW_MINUTES = 30
CONTEXT_TURNS = 2
MESSAGE_SPLIT_TOKEN = "<|split|>"
INITIATION_USER_CONTENT = "..."
INPUT_FILE = "chatdata_text.json"
OUTPUT_FILE = "ft_dataset.jsonl"

def parse_time(time_str):
    """Convert various time string formats to datetime object."""
    try:
        # Try ISO format first
        return datetime.fromisoformat(time_str.replace('Z', '+00:00'))
    except ValueError:
        # Try standard format
        return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")

def process_chat_log():
    # 2. Load and Pre-process Data
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        messages = json.load(f)
    
    # Convert time strings to datetime objects
    for msg in messages:
        msg['time'] = parse_time(msg['time'])
    
    # Sort messages by time if not already sorted
    messages.sort(key=lambda x: x['time'])
    
    # Open output file in append mode
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        i = 0
        while i < len(messages):
            # Skip if current message is from user
            if messages[i]['is_self']:
                i += 1
                continue
            
            # 4. Identify Friend's Message & Group Continuations
            friend_block_start = i
            friend_messages = [messages[i]['content']]
            
            # Look ahead for continuous messages from friend
            j = i + 1
            while j < len(messages):
                if not messages[j]['is_self'] and \
                   (messages[j]['time'] - messages[j-1]['time']).total_seconds() <= CONTINUATION_WINDOW_SECONDS:
                    friend_messages.append(messages[j]['content'])
                    j += 1
                else:
                    break
            
            friend_block_end = j - 1
            friend_block_content = MESSAGE_SPLIT_TOKEN.join(friend_messages)
            
            # 5. Determine Scenario and Group User's Reply
            scenario = None
            user_content = None
            
            # Look for user's messages before friend's block
            if friend_block_start > 0:
                # Find the last message before friend's block
                last_msg_before = messages[friend_block_start - 1]
                time_delta = messages[friend_block_start]['time'] - last_msg_before['time']
                
                if time_delta.total_seconds() > INITIATION_WINDOW_MINUTES * 60:
                    # Scenario A: Initiation
                    scenario = "initiation"
                    user_content = INITIATION_USER_CONTENT
                elif last_msg_before['is_self'] and time_delta.total_seconds() <= REPLY_WINDOW_MINUTES * 60:
                    # Scenario B: Reply - collect user's messages
                    scenario = "reply"
                    user_messages = []
                    k = friend_block_start - 1
                    while k >= 0 and messages[k]['is_self']:
                        user_messages.insert(0, messages[k]['content'])
                        k -= 1
                    user_content = '\n'.join(user_messages)
            
            # 6. If valid scenario, construct context and write to file
            if scenario:
                # Collect conversation history
                history = []
                current_turn = {
                    "role": "user",
                    "content": user_content
                }
                
                # Add the current turn
                data_point = {
                    "messages": history + [
                        current_turn,
                        {
                            "role": "assistant",
                            "content": friend_block_content
                        }
                    ]
                }
                
                # Write to output file
                outfile.write(json.dumps(data_point, ensure_ascii=False) + '\n')
            
            # 7. Advance loop index
            i = friend_block_end + 1

if __name__ == "__main__":
    process_chat_log()
    print(f"Dataset successfully created at {OUTPUT_FILE}")