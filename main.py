import sys, os, logging
import ollama
import discord
from discord.ext import commands
from dotenv import load_dotenv
from src.bentebot import bentebot
import redis

import context



THIS_PATH = os.path.dirname(os.path.realpath(__file__))

load_dotenv('.env')
logging.basicConfig(filename=f'{THIS_PATH}\\bot.log', level=logging.INFO, format='%(asctime)s %(message)s')

os.environ["OMP_NUM_THREADS"] = "4"
ENVIRONMENT = str(os.getenv("ENV"))
print(ENVIRONMENT, file=sys.stdout, flush=True)


if __name__ == '__main__':
    intents = discord.Intents.default()
    intents.message_content = True
    # disc = discord.Client(intents=intents)
    disc = commands.Bot(command_prefix="/", intents=intents)
    
    
    # Redis initialization
    redis_client = None
    if ENVIRONMENT != "dev":
        redis_host = str(os.getenv("REDIS_HOST"))
        redis_port = os.getenv("REDIS_PORT")
        if not redis_host == "" and redis_port:
            redis_client = redis.Redis(host=redis_host, port=int(redis_port), db=0, decode_responses=True)
    
    # Ollama initialization
    host = str(os.getenv("OLLAMA_HOST_URL"))
    llama_default_model = str(os.getenv("OLLAMA_DEFAULT_MODEL", "phi"))
    auth_name = os.getenv("BASIC_AUTH_USERNAME")
    auth_pass = os.getenv("BASIC_AUTH_PASSWORD")
    verify_ssl = os.getenv("VERIFY_SSL", "True")
    if verify_ssl.isdigit():
        verify_ssl = bool(int(verify_ssl))
    else:
        verify_ssl = verify_ssl.lower() == "true" # converts to bool
    llama = None
    if auth_name and auth_pass:
        llama = ollama.AsyncClient(host, auth=(auth_name, auth_pass), verify=verify_ssl)
    else:
        llama = ollama.AsyncClient(host, verify=verify_ssl)
    
    
    ## Set contextual variables for bentebot to function
    context.redis = redis_client
    context.llama = llama
    context.discord = disc
    context.llama_default_model = llama_default_model
    context.super_admin_ids = os.getenv("SUPER_ADMINS")
    context.discord_server_ids = os.getenv("DISCORD_SERVER_IDS")
        
    bentebot().run(os.getenv("DISCORD_TOKEN"))