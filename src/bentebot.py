import sys, os, io, json, datetime
import discord
import logging
import asyncio
from discord import app_commands
from .Response import Response

# Add the parent directory of `src/` to sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import context
from src.redis_conn import (
    save_message_redis,
    get_messages,
    get_message,
    get_all_message_ids,
    delete_messages,
    is_superadmin,
    add_super_admin,
    remove_super_admin,
    is_admin,
    add_server_admin,
    remove_server_admin,
    is_dm_allowed,
    add_dm_whitelist,
    remove_dm_whitelist,
    is_trusted_server,
    add_trusted_server,
    remove_trusted_server,
    set_current_model,
    get_current_model
)
from src.ollama_conn import (
    ollama_conn,
    get_model_list
)


class bentebot:
    def __init__(self):
        self.ollama_conn = ollama_conn()
        # register event handlers
        context.discord.event(self.on_ready)
        context.discord.event(self.on_message)
        
        self.register_slash_commands()
        
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



    async def on_message(self, message):
        if context.discord.user == message.author:
            # don't respond to ourselves
            return
        
        message_content = message.content.replace(f'<@{context.discord.user.id}>', '').strip()
        message_id = message.id
        author = message.author
        channel_id = message.channel.id
        attachments = message.attachments
        
        if message.guild is not None: # If server
            trusted_server = is_trusted_server(message.guild.id)
            if not trusted_server:
                return
            
            ## TODO: Before saving msg to redis, check if this channel is on a ignore list
            ##      TODO 2: Create ignore list logic. Redis getters & setters and implementation
            save_message_redis(message_id, message_content, author, channel_id, attachments)
            
            ## Check if we are mentioned in this message.
            if context.discord.user not in message.mentions:
                return
            
            ## TODO: Should only respond to users with a certain role attached to them: "bot commander", or if they are admins.
            await self.on_channel_message(message)
        else: # if DM
            dm_allowed = is_dm_allowed(message.author.id)
            if not dm_allowed:
                logging.info(f"{message.author.id} tried to DM me '{message_content}' without DM permission...")
                await message.add_reaction('🚫')
                return
            
            save_message_redis(message_id, message_content, author, channel_id, attachments)
            await self.on_direct_message(message)
        
        
    
    
    
    async def on_channel_message(self, message):
        ## Create and start writing task with ollama chatbot
        self.ollama_conn.add_task(message)
    
    
    async def on_direct_message(self, message):
        ## Create and start writing task with ollama chatbot
        self.ollama_conn.add_task(message)
    
    
    
    def register_slash_commands(self):
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
        
        context.discord.tree.add_command(
            app_commands.Command(
                name="model",
                description="Variety of model commands. Set, Get, List, Pull, Delete, Help",
                callback=self.slash_model,
            )
        )
        
        
        context.discord.tree.add_command(
            app_commands.Command(
                name="trust_server",
                description="Add/Remove current server to trusted servers. Use `action:help` for usage.",
                callback=self.slash_trust_server,
            )
        )
        
        context.discord.tree.add_command(
            app_commands.Command(
                name="dm_whitelist",
                description="Add/Remove tagged user to DM whitelist. Use `action:help` for usage.",
                callback=self.slash_dm_whitelist,
            )
        )
        
        context.discord.tree.add_command(
            app_commands.Command(
                name="admin",
                description="Add/Remove tagged user to server admin. Use `action:help` for usage.",
                callback=self.slash_server_admin,
            )
        )
        
        context.discord.tree.add_command(
            app_commands.Command(
                name="superadmin",
                description="Add/Remove tagged user to superadmin. Use `action:help` for usage.",
                callback=self.slash_superadmin,
            )
        )
        
        context.discord.tree.add_command(
            app_commands.Command(
                name="wipe",
                description="Wipe memory of current chat.",
                callback=self.slash_wipe_redis,
            )
        )
        
    
    
    async def slash_trust_server(self, interaction: discord.Interaction, action:str):
        # Action = "add" / "remove"
        admin_check = is_superadmin(interaction.user.id)
        if not admin_check:
            await interaction.response.send_message(
                "Not authorized.",
                ephemeral=True
            )
            return
        
        action = action.lower()
        if interaction.guild is None: # DM
            msg = "Can't trust DM."
        else: # Server or group
            if action == "add":
                result = add_trusted_server(interaction.guild.id)
                msg = f"✅ Added {interaction.guild.name} to trusted servers." if result else f"Redis is not connected."
                if result:
                    logging.info(
                        f"Trusted server added by {interaction.user.name} ({interaction.user.id}): "
                        f"{interaction.guild.name} ({interaction.guild.id})"
                    )
            elif action == "remove":
                result = remove_trusted_server(interaction.guild.id)
                msg = f"🗑️ Removed {interaction.guild.name} from trusted servers." if result else f"Redis is not connected."
                if result:
                    logging.info(
                        f"Trusted server removed by {interaction.user.name} ({interaction.user.id}): "
                        f"{interaction.guild.name} ({interaction.guild.id})"
                    )
            elif action == "help":
                    msg = (
                        "ℹ️ **Trust Server Command Help**\n"
                        "Use this command to manage trusted servers.\n\n"
                        "**Usage:** `/trust_server action:<add|remove|help>`\n"
                        "- `add` → Adds the current server to the trusted list.\n"
                        "- `remove` → Removes the current server from the trusted list.\n"
                        "- `help` → Displays this help message."
                    )
            else:
                msg = f"⚠️ Invalid action. Use `/trust_server action:help` for usage info."
        
        await interaction.response.send_message(msg, ephemeral=True)
            
    
    async def slash_dm_whitelist(self, interaction: discord.Interaction, action:str, tagged_user:discord.User):
        # Action = "add" / "remove"
        admin_check = is_superadmin(interaction.user.id)
        if not admin_check:
            await interaction.response.send_message(
                "Not authorized.",
                ephemeral=True
            )
            return
        action = action.lower()
        if action == "add":
            result = add_dm_whitelist(tagged_user.id)
            msg = f"✅ Added {tagged_user.mention} to DM whitelist." if result else f"Redis is not connected."
            if result:
                logging.info(
                    f"User added to DM Whitelist by {interaction.user.name} ({interaction.user.id}): "
                    f"{tagged_user.name} ({tagged_user.id})"
                )
        elif action == "remove":
            result = remove_dm_whitelist(tagged_user.id)
            msg = f"🗑️ Removed {tagged_user.mention} from DM whitelist." if result else "Redis is not connected."
            if result:
                logging.info(
                    f"User removed from DM Whitelist by {interaction.user.name} ({interaction.user.id}): "
                    f"{tagged_user.name} ({tagged_user.id})"
                )
        elif action == "help":
                msg = (
                    "ℹ️ **DM Whitelist Command Help**\n"
                    "Use this command to manage who can DM the bot.\n\n"
                    "**Usage:** `/dm_whitelist action:<add|remove|help> tagged_user:@user`\n"
                    "- `add` → Adds the mentioned user to DM whitelist.\n"
                    "- `remove` → Removes the mentioned user from DM whitelist.\n"
                    "- `help` → Displays this help message."
                )
        else:
            msg = "⚠️ Invalid action. Use `/dm_whitelist action:help` for usage info."
        
        await interaction.response.send_message(msg, ephemeral=True)
        
    
    async def slash_server_admin(self, interaction: discord.Interaction, action:str, tagged_user:discord.User):
        # Action = "add" / "remove"
        admin_check = is_admin(interaction.user.id, interaction.guild.id if interaction.guild else None)
        if not admin_check:
            await interaction.response.send_message(
                "Not authorized.",
                ephemeral=True
            )
            return
        action = action.lower()
        user_id = tagged_user.id
        if action == "add":
            result = add_server_admin(user_id, interaction.guild.id)
            msg = f"✅ Added {tagged_user.mention} to server admin." if result else f"Redis is not connected."
            if result:
                logging.info(
                    f"User added to server admin by {interaction.user.name} ({interaction.user.id}): "
                    f"{tagged_user.name} ({tagged_user.id}) - ({interaction.guild.name}) ({interaction.guild.id})"
                )
        elif action == "remove":
            result = remove_server_admin(user_id, interaction.guild.id)
            msg = f"🗑️ Removed {tagged_user.mention} from server admin." if result else "Redis is not connected."
            if result:
                logging.info(
                    f"User removed from server admin by {interaction.user.name} ({interaction.user.id}): "
                    f"{tagged_user.name} ({tagged_user.id}) - ({interaction.guild.name}) ({interaction.guild.id})"
                )
        elif action == "help":
                msg = (
                    "ℹ️ **Server Admin Command Help**\n"
                    "Use this command to manage server admin privileges.\n\n"
                    "**Usage:** `/admin action:<add|remove|help> tagged_user:@user`\n"
                    "- `add` → Adds the mentioned user as server admin in current server.\n"
                    "- `remove` → Removes the mentioned user as server admin in current server.\n"
                    "- `help` → Displays this help message."
                )
        else:
            msg = "⚠️ Invalid action. Use `/admin action:help` for usage info."
        
        await interaction.response.send_message(msg, ephemeral=True)
        
    
    async def slash_superadmin(self, interaction: discord.Interaction, action:str, tagged_user:discord.User):
        # Action = "add" / "remove"
        admin_check = is_superadmin(interaction.user.id)
        if not admin_check:
            await interaction.response.send_message(
                "Not authorized.",
                ephemeral=True
            )
            return
        action = action.lower()
        user_id = tagged_user.id
        if action == "add":
            result = add_super_admin(user_id)
            msg = f"✅ Added {tagged_user.mention} to super admin." if result else f"Redis is not connected."
            if result:
                logging.info(
                    f"User added to super admin by {interaction.user.name} ({interaction.user.id}): "
                    f"{tagged_user.name} ({tagged_user.id})"
                )
        elif action == "remove":
            result = remove_super_admin(user_id)
            msg = f"🗑️ Removed {tagged_user.mention} from super admin." if result else "Redis is not connected."
            if result:
                logging.info(
                    f"User removed from super admin by {interaction.user.name} ({interaction.user.id}): "
                    f"{tagged_user.name} ({tagged_user.id})"
                )
        elif action == "help":
                msg = (
                    "ℹ️ **Super Admin Command Help**\n"
                    "Use this command to manage super admin privileges.\n\n"
                    "**Usage:** `/superadmin action:<add|remove|help> tagged_user:@user`\n"
                    "- `add` → Adds the mentioned user as super admin.\n"
                    "- `remove` → Removes the mentioned user as super admin.\n"
                    "- `help` → Displays this help message."
                )
        else:
            msg = "⚠️ Invalid action. Use `/superadmin action:help` for usage info."
        
        await interaction.response.send_message(msg, ephemeral=True)
        
        
    
    async def slash_wipe_redis(self, interaction: discord.Interaction):
        if interaction.guild is None: # DM
            if is_dm_allowed(interaction.user.id) or is_admin(interaction.user.id):
                result = delete_messages(interaction.channel_id)
                msg = "Memory Wiped..."
                if result:
                    logging.info(
                        f"DM history wiped by {interaction.user.name} ({interaction.user.id}): "
                    )
            else:
                msg = "Not authorized..."
        else: # Server or group
            if is_admin(interaction.user.id):
                result = delete_messages(interaction.channel_id)
                msg = "Memory Wiped..."
                if result:
                    logging.info(
                        f"Channel message history wiped by {interaction.user.name} ({interaction.user.id}): "
                        f"{interaction.guild.name} ({interaction.guild.id}) ({interaction.channel.name}) ({interaction.channel_id})"
                    )
            else:
                msg = "Not authorized..."
                
        await interaction.response.send_message(msg, ephemeral=True)
            
    
    
    ## TODO: Create slash command to pull new models - (Superadmin only)
    ## TODO: Create slash command to delete models - (Superadmin only)
    async def slash_model(self, interaction: discord.Interaction, action: str = "current", model: str = None):
        admin_check = is_admin(interaction.user.id, interaction.guild.id if interaction.guild else None)
        if not admin_check:
            await interaction.response.send_message(
                "Not authorized.",
                ephemeral=True
            )
            return
        
        action = action.lower()
        ## slash command to see current Ollama Model being used - (Admin only)
        if action == "current":
            current_model = get_current_model(interaction.channel_id)
            msg = f"**Current model:** {current_model}"
        ## slash command to list available models which are downloaded already - (Admin only)
        elif action == "list":
            model_list = await get_model_list()
            if not model_list:
                msg = "**No models available.**"
            else:
                formatted = "\n".join(f"{i+1}. {name}" for i, name in enumerate(model_list))
                msg = f"**Available Models:**\n```\n{formatted}\n```"
        ## slash command to change current model - (Admin only)
        elif action == "set":
            if not model:
                msg = "**Error:** You must provide a model name to set."
            else:
                model_list = await get_model_list()
                if model not in model_list:
                    msg = f"**Error:** Model `{model}` not found. Use `/model list` to see available models."
                else:
                    success = set_current_model(interaction.channel_id, model)
                    if success:
                        msg = f"✅ **Model set to:** {model}"
                    else:
                        msg = "**Error:** Could not save model. Redis may not be available."
        elif action == "help":
                msg = (
                    "ℹ️ **Model Command Help**\n"
                    "Use this command to manage the Ollama models.\n\n"
                    "**Usage:** `/model action:<current|list|set|help> model:<model_name>`\n"
                    "- `current` → Shows the current model in use.\n"
                    "- `list` → Lists all available models.\n"
                    "- `set` → Sets the current model. Must provide a model name.\n"
                    "- `help` → Displays this help message."
                )
        else:
            msg = "⚠️ Invalid action. Use `/model action:help` for usage info."
                
        await interaction.response.send_message(msg, ephemeral=True)
 
    ## async def slash_logs(self, interaction: discord.Interaction)
    ### Slash command to see logs. there should be an action: "file" / "read" / "clear"
    ###     file should make the log file downloadable in discord
    ###     read should take another argument `amount` which take the amount of lines to read
    ###     clear should clear the log file
    ## Obviously superadmin
 
    async def slash_hello(self, interaction: discord.Interaction):
        await interaction.response.send_message(f"Hello {interaction.user.mention}! How's it hanging?")
        
    
    
    async def slash_test(self, interaction: discord.Interaction, message_id: str):
        """Fetch a stored message by ID"""
        if not context.redis:
            await interaction.response.send_message("Redis not connected.", ephemeral=True)
            return
        
        if not is_admin(interaction.user.id, interaction.guild.id if interaction.guild else None):
            await interaction.response.send_message("no", ephemeral=True)
            return
        
        channel_id = interaction.channel.id
        if message_id.isdigit():
            stored_msg = get_message(channel_id, message_id)
            if stored_msg is not None:
                await interaction.response.send_message(f"Message ID {message_id} content: {stored_msg['content']}", ephemeral=True)
            else:
                await interaction.response.send_message(f"Message ID {message_id} not found.", ephemeral=True)
        else:
            # Then check if its an "?", if so, we want to respond with a comma seperated list of message_ids stored in redis.
            message_ids = get_all_message_ids(channel_id)
            if message_ids:
                await interaction.response.send_message(
                    ", ".join(message_ids), ephemeral=True
                )
            else:
                await interaction.response.send_message(
                    "No messages stored in this channel.", ephemeral=True
                )
