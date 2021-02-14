import os
import time
import discord
from discord.ext import commands
import asyncio
from dotenv import load_dotenv
from PIL import Image
import requests
from hot_nothotdog_pred import predict_image

load_dotenv()
# Permissions num: 75776
TOKEN = os.getenv('DISCORD_TOKEN')
GUILD = os.getenv('DISCORD_GUILD')
MODEL_PATH = os.getenv('MODEL_PATH')
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


@bot.command(name="hotdog")
async def hotdog_cmd(ctx):
    async for message in ctx.channel.history(limit=200):
        if (message.attachments != None and message.attachments != []):
            src = message.attachments[0].url
            break
        elif message.content.startswith('https://cdn.discordapp.com/attachments/'):
            src = message.content
            break
    print(src)
    r = requests.get(src, stream=True, timeout=10)
    with open(f"temp/temp_pic.png", 'wb') as output:
        output.write(r.content)
    img = Image.open('temp/temp_pic.png')
    print(img)
    prob = predict_image(img, MODEL_PATH)
    try:
        os.remove('temp/temp_pic.png')
    except:
        print("Error, temporary picture was already removed or never existed")
    print(prob)
    if prob == 1:
        await ctx.send('\U0001F32D\U0000200D\U0000274C')
    else:
        await ctx.send('\U0001F32D')

bot.run(TOKEN)
