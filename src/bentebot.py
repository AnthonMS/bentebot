import sys, os, json
import discord
import logging
from discord import app_commands

# Add the parent directory of `src/` to sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import context



class bentebot:
    def __init__(self):
        # register event handlers
        context.discord.event(self.on_ready)
        context.discord.event(self.on_message)
        
            
        # Register slash commands
        context.discord.tree.add_command(
            app_commands.Command(
                name="hello",
                description="Say hello to my little bot!",
                callback=self.slash_hello  # async function taking `interaction`
            )
        )
        
        context.discord.tree.add_command(
            app_commands.Command(
                name="test",
                description="Get a message by ID",
                callback=self.slash_test,
            )
        )
        
    def run(self, token):
        try:
            context.discord.run(token)
        except Exception:
            logging.exception('Discord client encountered an error')



    async def on_ready(self):
        activity = discord.Activity(name='bentebot', state='under development', type=discord.ActivityType.custom)
        await context.discord.change_presence(activity=activity)

        logging.info(
            'Ready! Invite URL: %s',
            discord.utils.oauth_url(
                context.discord.application_id,
                permissions=discord.Permissions(
                    administrator=True,
                ),
                scopes=['bot'],
            ),
        )
        
        # Sync slash commands to Discord
        await context.discord.tree.sync()
        logging.info('Slash commands synced!')



    async def on_message(self, message):
        if context.discord.user == message.author:
            # don't respond to ourselves
            return
        if message.guild is not None: # If server
            trusted_server = await self.is_trusted_server(message.guild.id)
            if not trusted_server:
                logging.info(f"{message.author.id} tried to summon me '{content}' in untrusted server '{message.guild.id}'...")
                await message.add_reaction('🚫')
                return
            
            await self.on_channel_message(message)
        else: # if DM
            dm_allowed = await self.is_dm_allowed(message.author.id)
            if not dm_allowed:
                logging.info(f"{message.author.id} tried to DM me '{content}' without DM permission...")
                await message.add_reaction('🚫')
                return
            
            await self.on_direct_message(message)
        
        
    
    
    async def on_channel_message(self, message):
        message_content = message.content.replace(f'<@{context.discord.user.id}>', '').strip()
        message_id = message.id
        author = message.author
        channel_id = message.channel.id
        attachments = message.attachments
        
        await self.save_message_redis(message_id, message_content, author, channel_id, attachments)
    
    async def on_direct_message(self, message):
        message_content = message.content.replace(f'<@{context.discord.user.id}>', '').strip()
        message_id = message.id
        author = message.author
        channel_id = message.channel.id
        attachments = message.attachments
        
        await self.save_message_redis(message_id, message_content, author, channel_id, attachments)
        
        
        
        
        
        
    
    ##
    ##
    ##
    ##
    ##
    ##
    ## _____________ +SLASH COMMANDS  _____________ ##
        
        
    async def slash_hello(self, interaction: discord.Interaction):
        await interaction.response.send_message(f"Hello {interaction.user.mention}! How's it hanging?")
        
    
    
    async def slash_test(self, interaction: discord.Interaction, message_id: int):
        """Fetch a stored message by ID"""
        channel_key = f"messages:{interaction.channel.id}"
        if not context.redis:
            await interaction.response.send_message("Redis not connected.", ephemeral=True)
            return

        messages = context.redis.lrange(channel_key, 0, -1)
        for msg_json in messages:
            msg = json.loads(msg_json)
            if msg["id"] == message_id:
                await interaction.response.send_message(
                    f"Message ID {message_id}:\n{msg['content']}", ephemeral=True
                )
                return

        await interaction.response.send_message(f"Message ID {message_id} not found.", ephemeral=True)




    ## _____________ -SLASH COMMANDS  _____________ ##
    ##
    ##
    ##
    ##
    ##
    ##
    ## _____________ +HELPERS  ____________________ ##
    
    
    async def save_message_redis(self, message_id, message_content, author, channel_id, attachments = []):
        if not context.redis:
            return None
        
        messsage_content = datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n\n' + message_content + "\n\nSent by: " + str(author.name)
        
        context.redis.rpush(f"messages:{channel_id}", json.dumps({
            "author": author.id,
            "content": messsage_content,
            "id": message_id,
            "attachments": [attachment.url for attachment in attachments],
        }))
        
        
    async def is_admin(self, user_id: int, guild_id: int = None):
        if context.super_admin_ids is not None:
            super_admin_ids = [int(id.strip()) for id in context.super_admin_ids.split(",")]
            if (user_id in super_admin_ids):
                return True
        
        if context.redis and guild_id is not None:
            if context.redis.sismember(f"admins:{guild_id}", str(user_id)):
                return True
        
        return False
    
    async def is_dm_allowed(self, user_id: int):
        is_admin = await self.is_admin(user_id)
        if is_admin:
            return True
        
        
        if context.redis:
            if context.redis.sismember(f"dm_whitelist", str(user_id)):
                return True
        
        
        return False
        
    async def is_trusted_server(self, server_id: int):
        if context.discord_server_ids is not None:
            discord_server_ids = [int(id.strip()) for id in context.discord_server_ids.split(",")]
            if (server_id in discord_server_ids):
                return True
        
        if context.redis:
            if context.redis.sismember(f"trusted_servers", str(server_id)):
                return True
            
        return False
        
    
    ## _____________ -HELPERS  ____________________ ##
    ##
    ##
    ##
    ##
    ##
    ##