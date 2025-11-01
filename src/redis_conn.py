import json, datetime
from typing import Optional, List, Union
from dataclasses import dataclass
import discord
import context


@dataclass
class StoredMessageData:
    id: int
    author_id: int
    author_name: str
    role: str
    content: str
    timestamp: str
    attachments: List[str]


def save_message_redis(
    message_id: Union[int, str], 
    message_content: str, 
    author: Union[discord.User, discord.Member], 
    channel_id: Union[int, str], 
    attachments: Optional[List[discord.Attachment]] = []
) -> None:
    if not context.redis:
        return None
    
    # message_content = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n\n' + message_content + "\n\nSent by: " + str(author.name)
    
    payload = {
        "id": message_id,
        "author_id": author.id,
        "author_name": author.name,
        "role": "assistant" if author.id == context.discord.user.id else "user",
        "content": message_content,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "attachments": [a.url for a in attachments],
    }
    context.redis.hset(
        f"messages:{channel_id}",
        message_id,
        json.dumps(payload),
    )
    
def get_messages(
    message:discord.Message, 
    format: bool = False
) -> List[dict]:
    if not context.redis:
        return [{"role": "assistant" if message.author.id == context.discord.user.id else "user", "content": message.content}]

    # Read stored messages
    raw = []
    for msg in context.redis.hvals(f"messages:{message.channel.id}"):
        if isinstance(msg, bytes):
            msg = msg.decode()
        raw.append(json.loads(msg))

    if not format:
        return raw
    
    # # Sort by timestamp
    # raw.sort(key=lambda m: m.get("timestamp", ""))

    formatted = []
    for m in raw:
        ts = m.get("timestamp")
        if ts:
            try:
                dt = datetime.fromisoformat(ts.replace("Z", ""))
                ts_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                ts_str = ts
        else:
            ts_str = ""

        formatted.append({
            "role": m["role"],
            "content": f"{ts_str} {m['content']}\n\nSent by: {m["author_name"]}"
        })
        ## TODO: Take the attachments in the stored messages: "attachments": [a.url for a in attachments]
        ##          Check if any of them are images.
        ##          If they are images, we can include it in the chat using `images` argument

    system_instruction = {
        "role": "system",
        "content": (
            "Messages include metadata like timestamps and sender names. "
            "Ignore metadata in your reasoning unless it directly affects the prompt."
            "Respond normally as plain text. "
            "Do not include timestamps or author tags in your reply."
        )
    }
    return [system_instruction] + formatted
        
        
def get_message(
    channel_id: int, 
    message_id: int
) -> Optional[StoredMessageData]:
    if not context.redis:
        return None

    msg_json = context.redis.hget(f"messages:{channel_id}", message_id)
    if msg_json:
        return json.loads(msg_json)
    
    return None


def get_all_message_ids(
    channel_id: int
) -> List[str]:
    if not context.redis:
        return []

    # Get all fields (message IDs) in the hash
    message_ids = context.redis.hkeys(f"messages:{channel_id}")
    
    # If your Redis client returns bytes, decode to str
    message_ids = [mid.decode() if isinstance(mid, bytes) else str(mid) for mid in message_ids]

    return message_ids


def delete_messages(
    channel_id: int
) -> bool:
    if not context.redis:
        return False

    deleted = context.redis.delete(f"messages:{channel_id}")
    return bool(deleted)



    
def is_superadmin(
    user_id: int
) -> bool:
    if context.super_admin_ids is not None:
        super_admin_ids = [int(id.strip()) for id in context.super_admin_ids.split(",")]
        if (user_id in super_admin_ids):
            return True
        
    if context.redis:
        if context.redis.sismember(f"super_admins", str(user_id)):
            return True
    return False

def add_super_admin(
    user_id: int
) -> bool:
    if not context.redis:
        return False
    
    exists = context.redis.sismember("super_admins", str(user_id))
    if exists:
        return True
    
    context.redis.sadd("super_admins", str(user_id))
    return True

def remove_super_admin(
    user_id: int
) -> bool:
    if not context.redis:
        return False
    
    exists = context.redis.sismember("super_admins", str(user_id))
    if not exists:
        return True  # not present
    
    context.redis.srem("super_admins", str(user_id))
    return True


    
    
def is_admin(
    user_id: int, 
    guild_id: int = None
) -> bool:
    superadmin_check = is_superadmin(user_id)
    if superadmin_check:
        return True
    
    if context.redis and guild_id is not None:
        if context.redis.sismember(f"admins:{guild_id}", str(user_id)):
            return True
    
    return False

def add_server_admin(
    user_id: int, 
    guild_id: int
) -> bool:
    if not context.redis:
        return False
    
    exists = context.redis.sismember(f"admins:{guild_id}", str(user_id))
    if exists:
        return True
    
    context.redis.sadd(f"admins:{guild_id}", str(user_id))
    return True

def remove_server_admin(
    user_id: int, 
    guild_id: int
) -> bool:
    if not context.redis:
        return False
    
    exists = context.redis.sismember(f"admins:{guild_id}", str(user_id))
    if not exists:
        return True  # not present
        
    context.redis.srem(f"admins:{guild_id}", str(user_id))
    return True




def is_dm_allowed(
    user_id: int
) -> bool:
    admin_check = is_admin(user_id)
    if admin_check:
        return True
    
    
    if context.redis:
        if context.redis.sismember(f"dm_whitelist", str(user_id)):
            return True
    
    return False

def add_dm_whitelist(
    user_id: int
) -> bool:
    if not context.redis:
        return False
    
    exists = context.redis.sismember("dm_whitelist", str(user_id))
    if exists:
        return True
    
    context.redis.sadd("dm_whitelist", str(user_id))
    return True

def remove_dm_whitelist(
    user_id: int
) -> bool:
    if not context.redis:
        return False
    
    exists = context.redis.sismember("dm_whitelist", str(user_id))
    if not exists:
        return True  # not present
    
    context.redis.srem("dm_whitelist", str(user_id))
    return True
    
    
    
    
def is_trusted_server(
    server_id: int
) -> bool:
    if context.discord_server_ids is not None:
        discord_server_ids = [int(id.strip()) for id in context.discord_server_ids.split(",")]
        if (server_id in discord_server_ids):
            return True
    
    if context.redis:
        if context.redis.sismember(f"trusted_servers", str(server_id)):
            return True
        
    return False

def add_trusted_server(
    server_id: int
) -> bool:
    if not context.redis:
        return False
    
    exists = context.redis.sismember("trusted_servers", str(server_id))
    if exists:
        return True
    
    context.redis.sadd("trusted_servers", str(server_id))
    return True
        
def remove_trusted_server(
    server_id: int
) -> bool:
    if not context.redis:
        return False

    exists = context.redis.sismember("trusted_servers", str(server_id))
    if not exists:
        return True

    context.redis.srem("trusted_servers", str(server_id))
    return True




# def is_followed_channel(channel_id: int):
#     if context.redis:
#         if context.redis.sismember(f"followed_channel", str(channel_id)):
#             return True
        
#     return False

# def add_followed_channel(channel_id: int):
#     if not context.redis:
#         return False
    
#     exists = context.redis.sismember("followed_channel", str(channel_id))
#     if exists:
#         return False
    
#     context.redis.sadd("followed_channel", str(channel_id))
#     return True


# def remove_followed_channel(channel_id: int):
#     if not context.redis:
#         return False
    
#     exists = context.redis.sismember("trusted_servers", str(server_id))
#     if not exists:
#         return False 
    
#     context.redis.srem("followed_channel", str(channel_id))
#     return True
        


def set_current_model(
    channel_id:int, 
    new_model:str
) -> bool:
    if context.redis:
        context.redis.set(f"model:{channel_id}", new_model)
        return True
    
    return False

def get_current_model(
    channel_id:int
) -> str:
    if context.redis:
        model = context.redis.get(f"model:{channel_id}")
        if model:
            if isinstance(model, bytes):
                model = model.decode()
            return model
    
    return context.llama_default_model