import io

class Response:
    def __init__(self, message):
        self.message = message
        self.channel = message.channel
        self.author = message.author

        self.r = None
        self.sb = io.StringIO()

    async def write(self, s, end=''):
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