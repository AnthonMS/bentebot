import sys, os, io, json, datetime
import discord
import logging
import asyncio
from discord import app_commands
from .Response import Response

# Add the parent directory of `src/` to sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import context



class bentebot:
    def __init__(self):
        self.writing_tasks = {}
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
        
        context.discord.tree.add_command(
            app_commands.Command(
                name="model",
                description="Variety of model commands. Set, Get, List, Pull, Delete",
                callback=self.slash_model,
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
        
        message_content = message.content.replace(f'<@{context.discord.user.id}>', '').strip()
        message_id = message.id
        author = message.author
        channel_id = message.channel.id
        attachments = message.attachments
        
        if message.guild is not None: # If server
            trusted_server = await self.is_trusted_server(message.guild.id)
            if not trusted_server:
                logging.info(f"{message.author.id} tried to summon me '{message_content}' in untrusted server '{message.guild.id}'...")
                await message.add_reaction('🚫')
                return
            
            await self.save_message_redis(message_id, message_content, author, channel_id, attachments)
            
            ## Check if we are mentioned in this message.
            if context.discord.user not in message.mentions:
                return
            
            await self.on_channel_message(message)
        else: # if DM
            dm_allowed = await self.is_dm_allowed(message.author.id)
            if not dm_allowed:
                logging.info(f"{message.author.id} tried to DM me '{message_content}' without DM permission...")
                await message.add_reaction('🚫')
                return
            
            await self.save_message_redis(message_id, message_content, author, channel_id, attachments)
            await self.on_direct_message(message)
        
        
    
    
    
    async def on_channel_message(self, message):
        # await message.add_reaction('🤔')
        await message.channel.send("You mentioned me?")
    
    
    
    async def on_direct_message(self, message):
        # message_content = message.content.replace(f'<@{context.discord.user.id}>', '').strip()
        # if message_content.lower() == "hello":
        #     await message.channel.send("hello")
        
        r = Response(message)
        writing = asyncio.create_task(self.writing(r))
        self.writing_tasks[message.id] = (r, writing)
    
    
    
        
    ## TODO: Split the Ollama Logic up into another script
    async def writing(self, response):
        full_response = ""
        try:
            thinking = asyncio.create_task(self.thinking(response.message))
            messages = await self.get_messages(response.message, True)
            # converted_messages = 
            
            async for part in self.chat(messages, context.llama_default_model):
                if thinking is not None and not thinking.done():
                    thinking.cancel()
                    
                sys.stdout.write(part['message']['content'])
                sys.stdout.flush()
                
                part_content = part['message']['content']
                full_response += part_content
                await response.write(part_content, end='...')
                    
            await response.write('')
        except asyncio.CancelledError:
            await response.message.add_reaction('❌')
        except Exception as e:
            await response.message.add_reaction('💩')
            logging.error("Error answering")
            logging.error(e)
            pass
        finally:
            if thinking is not None and not thinking.done():
                thinking.cancel()
            del self.writing_tasks[response.message.id]  # Remove the task from the dictionary
             # save bot reply
            bot_msg = response.r
            if bot_msg:
                await self.save_message_redis(
                    message_id=bot_msg.id,
                    message_content=full_response,
                    author=bot_msg.author,
                    channel_id=bot_msg.channel.id,
                    attachments=[],
                    # attachments=bot_msg.attachments,
                )
       
    async def thinking(self, message, timeout=999):
        try:
            await message.add_reaction('🤔')
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logging.error("Error thinking")
            logging.error(e)
            await message.add_reaction('💩')
            pass
        finally:
            await message.remove_reaction('🤔', context.discord.user)
     
    async def chat(self, messages, model=None, milliseconds=1000):
        if model is None:
            model = context.llama_default_model
        sb = io.StringIO() # create new StringIO object that can write and read from a string buffer
        t = datetime.datetime.now()
        try:
            generator = await context.llama.chat(model, messages=messages, stream=True)
            async for part in generator:
                sb.write(part['message']['content']) # write content to StringIO buffer
                # print(part['message']['content'], end='', flush=True)
                sys.stdout.write(part['message']['content'])
                sys.stdout.flush()
            
                if milliseconds is None:
                    # If milliseconds is None, yield every time we get a return from the stream
                    part['message']['content'] = sb.getvalue()
                    yield part
                    sb.seek(0, io.SEEK_SET)
                    sb.truncate()
                elif part['done'] or datetime.datetime.now() - t > datetime.timedelta(milliseconds=milliseconds):
                    part['message']['content'] = sb.getvalue()
                    yield part
                    t = datetime.datetime.now()
                    sb.seek(0, io.SEEK_SET) # change current position in StringIO buffer (io.SEEK_SET = position is relative to beginning of buffer)
                    sb.truncate() # resizes StringIO buffer to current position. Since current position was just set to 0, this clears the buffer
                
        except Exception as e:
            logging.error("Error getting AI chat response")
            logging.error(e)
        
        
    async def generate(self, content, model=None):
        if model is None:
            model = context.llama_default_model
        sb = io.StringIO()
        t = datetime.datetime.now()
        try:
            generator = await context.llama.generate(model=model, prompt=content, keep_alive=-1, stream=True)
            async for part in generator:
                sb.write(part['response'])

                if part['done'] or datetime.datetime.now() - t > datetime.timedelta(seconds=1):
                    part['response'] = sb.getvalue()
                    yield part
                    t = datetime.datetime.now()
                    sb.seek(0, io.SEEK_SET)
                    sb.truncate()

        except Exception as e:
            logging.error("Error getting AI generate response")
            logging.error(e)
        
        
    
    ##
    ##
    ##
    ##
    ##
    ##
    ## _____________ +SLASH COMMANDS  _____________ ##
    
    ## DONE: Create slash command to see current Ollama Model being used - (Admin only)
    ## DONE: Create slash command to list available models which are downloaded already - (Admin only)
    ## DONE: Create slash command to change current model - (Admin only)
    ## TODO: Create slash command to pull new models - (Admin only)
    ## TODO: Create slash command to delete models - (Admin only)
    ## TODO: Create slash command to wipe redis memory so chatbot "forgets" chat history - (Admin only)
    ## TODO: Create slash command to add/remove user to dm_whitelist - (Admin only)
    ## TODO: Create slash command to add/remove user from channel admin ( admins:{guild_id} ) - (Admin only)
    
    ## TODO: Make it so being an admin/dm_allowed/regular_allowed_user can be set through Discord Roles on servers?
    
    
    async def slash_model(self, interaction: discord.Interaction, action: str = "current", arg2: str = None):
        is_admin = await self.is_admin(interaction.user.id, interaction.guild.id if interaction.guild else None)
        if not is_admin:
            await interaction.response.send_message(
                "Not authorized.",
                ephemeral=True
            )
            return
        
        action = action.lower()
        if action == "current":
            current_model = await self.get_current_model(interaction.channel_id)
            await interaction.response.send_message(f"**Current model:** {current_model}")
            return
        elif action == "list":
            model_list = await self.get_model_list()
            if not model_list:
                await interaction.response.send_message("**No models available.**")
                return
            # Format nicely as numbered list in a code block
            formatted = "\n".join(f"{i+1}. {name}" for i, name in enumerate(model_list))
            await interaction.response.send_message(f"**Available Models:**\n```\n{formatted}\n```")
            return
        elif action == "set":
            if not arg2:
                await interaction.response.send_message(
                    "**Error:** You must provide a model name to set.",
                    ephemeral=True
                )
                return
            
            model_list = await self.get_model_list()
            if arg2 not in model_list:
                await interaction.response.send_message(
                    f"**Error:** Model `{arg2}` not found. Use `/model list` to see available models.",
                    ephemeral=True
                )
                return
            
            success = await self.set_current_model(interaction.channel_id, arg2)
            if success:
                await interaction.response.send_message(
                    f"**Model set to:** {arg2}",
                    ephemeral=True
                )
            else:
                await interaction.response.send_message(
                    "**Error:** Could not save model. Redis may not be available.",
                    ephemeral=True
                )
                
            return
        
    async def slash_hello(self, interaction: discord.Interaction):
        await interaction.response.send_message(f"Hello {interaction.user.mention}! How's it hanging?")
        
    
    
    async def slash_test(self, interaction: discord.Interaction, message_id: str):
        """Fetch a stored message by ID"""
        if not context.redis:
            await interaction.response.send_message("Redis not connected.", ephemeral=True)
            return
        
        if not await self.is_admin(interaction.user.id, interaction.guild.id if interaction.guild else None):
            await interaction.response.send_message("no", ephemeral=True)
            return
        
        channel_id = interaction.channel.id
        if message_id.isdigit():
            stored_msg = await self.get_message(channel_id, message_id)
            if stored_msg is not None:
                await interaction.response.send_message(f"Message ID {message_id} content: {stored_msg['content']}", ephemeral=True)
            else:
                await interaction.response.send_message(f"Message ID {message_id} not found.", ephemeral=True)
        else:
            # Then check if its an "?", if so, we want to respond with a comma seperated list of message_ids stored in redis.
            message_ids = await self.get_all_message_ids(channel_id)
            if message_ids:
                await interaction.response.send_message(
                    ", ".join(message_ids), ephemeral=True
                )
            else:
                await interaction.response.send_message(
                    "No messages stored in this channel.", ephemeral=True
                )




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
        
        # message_content = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n\n' + message_content + "\n\nSent by: " + str(author.name)
        
        payload = {
            "id": message_id,
            "author_id": author.id,
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
        
    async def get_messages(self, message, format: bool = False):
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
        
        # Sort by timestamp
        raw.sort(key=lambda m: m.get("timestamp", ""))

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

            author_id = int(m["author_id"])
            guild = message.guild
            author = guild.get_member(author_id) if guild else None
            author_name = author.display_name if author else f"User:{author_id}"

            formatted.append({
                "role": m["role"],
                "content": f"{ts_str} {m['content']}\n\nSent by: {author_name}"
            })

        return formatted
    
    
    def format_messages_for_chat(self, stored_messages):
        # Sort by timestamp
        stored_messages.sort(key=lambda m: m.get("timestamp", ""))

        formatted = []
        for m in stored_messages:
            # Convert timestamp
            ts = m.get("timestamp")
            if ts:
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", ""))
                    timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    timestamp_str = ts
            else:
                timestamp_str = ""

            # Resolve author name via Discord
            author_id = int(m["author_id"])
            author = channel.guild.get_member(author_id) if channel.guild else None
            author_name = author.display_name if author else f"User:{author_id}"

            # Build content like your old style
            content = (
                f"{timestamp_str} "
                f"{m['content']}\n\n"
                f"Sent by: {author_name}"
            )

            formatted.append({
                "role": m["role"],
                "content": content
            })

        return formatted
            
            
    async def get_message(self, channel_id: int, message_id: int):
        if not context.redis:
            return None

        msg_json = context.redis.hget(f"messages:{channel_id}", message_id)
        if msg_json:
            return json.loads(msg_json)
        
        return None
    
    
    async def get_all_message_ids(self, channel_id: int):
        if not context.redis:
            return []

        # Get all fields (message IDs) in the hash
        message_ids = context.redis.hkeys(f"messages:{channel_id}")
        
        # If your Redis client returns bytes, decode to str
        message_ids = [mid.decode() if isinstance(mid, bytes) else str(mid) for mid in message_ids]

        return message_ids
    
        
        
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
    
    
    
    
    
    async def set_current_model(self, channel_id:int, new_model:str):
        if context.redis:
            context.redis.set(f"model:{channel_id}", new_model)
            return True
        
        return False
    
    async def get_current_model(self, channel_id:int):
        if context.redis:
            model = context.redis.get(f"model:{channel_id}")  # no await
            if model:
                if isinstance(model, bytes):
                    model = model.decode()
                return model
        
        return context.llama_default_model
        
    ## TODO: Should this go in redis_conn.py or a llama_conn.py file??
    async def get_model_list(self):
        model_list = await context.llama.list()
        logging.info(f"Model List: \n {model_list}")
        available_models = []
        for model in model_list['models']:
            # Use the correct attribute
            available_models.append(model.model)  
        return available_models
    
    ## _____________ -HELPERS  ____________________ ##
    ##
    ##
    ##
    ##
    ##
    ##