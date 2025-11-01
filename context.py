## File for accessing contextual variables from main.py into bentebot.py
import ollama
from discord.ext import commands
import redis as rediss

redis: Optional[rediss.Redis] = None
llama: Optional[ollama.AsyncClient] = None
discord: Optional[commands.Bot] = None
llama_default_model: str = None
super_admin_ids: Optional[str] = None
discord_server_ids: Optional[str] = None