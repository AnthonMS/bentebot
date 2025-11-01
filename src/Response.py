from typing import Optional
import io
import discord

class Response:
    def __init__(self, message: discord.Message):
        self.message: discord.Message = message
        self.channel: Optional[discord.abc.Messageable] = message.channel
        self.author: Optional[discord.abc.Messageable] = message.author

        self.r: Optional[discord.Message] = None
        self.sb: io.StringIO = io.StringIO()

    async def write(self, s: str, end: str='') -> None:
        if self.sb.seek(0, io.SEEK_END) + len(s) + len(end) > 2000:
            self.r = None
            self.sb.seek(0, io.SEEK_SET)
            self.sb.truncate()

        self.sb.write(s)

        value = self.sb.getvalue().strip()
        if not value:
            return
            
        if self.r:
            await self.r.edit(content=value + end)
            return

        if self.channel:
            self.r = await self.channel.send(value)
        elif self.author:
            self.r = await self.author.send(value)