import os
import time
import discord
from discord.ext import commands
import asyncio
from dotenv import load_dotenv
from PIL import Image
from hot_nothotdog_pred import predict_image

load_dotenv()
# Permissions num: 75776
TOKEN = os.getenv('DISCORD_TOKEN')
GUILD = os.getenv('DISCORD_GUILD')
bot = commands.Bot(command_prefix='!')


def is_welcome(message):
    return message.content == f"{message.author.display_name} has entered the chat" and message.author.bot


@bot.event
async def on_ready():
    # Cycle through guilds, find the one matching that of .env variable
    guild = discord.utils.get(bot.guilds, name=GUILD)
    print(
        f'{bot.user.name} is connected to the following guild:\n'
        f'{guild.name}(id: {guild.id})')
    for channel in guild.text_channels:
        await channel.purge(limit=100, check=is_welcome)
        await channel.send(f"{bot.user.name} has entered the chat", delete_after=120)
    # Print members of guild that bot has jsut connected to
    members = '\n - '.join([member.name for member in guild.members])
    print(f'Guild Members:\n - {members}')

bot.run(TOKEN)
