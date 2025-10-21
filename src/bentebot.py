import sys, os
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
        context.discord.tree.add_command(app_commands.Command(
            name="hello",
            description="Say hello to my little bot!",
            callback=self.slash_hello  # async function taking `interaction`
        ))
        
        
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
        
        content = message.content.replace(f'<@{context.discord.user.id}>', '').strip()
        logging.info('Message intent triggered: %s', content)
        
        # optionally process commands manually if needed
        await context.discord.process_commands(message)
        
        
        
    async def slash_hello(self, interaction: discord.Interaction):
        await interaction.response.send_message(f"Hello {interaction.user.mention}! How's it hanging?")