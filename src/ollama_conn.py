import sys, os, io, datetime
import asyncio
import logging
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import context
from src.Response import Response
from src.redis_conn import (
    save_message_redis,
    get_messages
)

class ollama_conn:
    def __init__(self):
        self.writing_tasks = {}
        
    async def add_task(self, message):
        r = Response(message)
        writing_task = asyncio.create_task(self.writing(r))
        self.writing_tasks[message.id] = (r, writing_task)
    
    async def remove_task(self, message_id):
        del self.writing_tasks[message_id] 

    async def writing(self, response):
        full_response = ""
        try:
            thinking = asyncio.create_task(self.think(response.message))
            messages = await get_messages(response.message, True)
            
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
            ## TODO: Fix this!
            del self.writing_tasks[response.message.id]  # Remove the task from the dictionary
                # save bot reply
            bot_msg = response.r
            if bot_msg:
                await save_message_redis(
                    message_id=bot_msg.id,
                    message_content=full_response,
                    author=bot_msg.author,
                    channel_id=bot_msg.channel.id,
                    attachments=[],
                    # attachments=bot_msg.attachments,
                )
    
    async def think(self, message, timeout=999):
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