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
MODEL_PATH = os.getenv('MODEL_PATH')
bot = commands.Bot(command_prefix='!')


@bot.event
async def on_ready():
    guilds = bot.guilds
    print(
        f'Connected to the following guilds:\n{[(x.name, x.id) for x in guilds]}')
    for guild in guilds:
        for channel in guild.text_channels:
            await channel.send(f"{bot.user.name} has entered the chat", delete_after=120)


@bot.command(name="hotdog")
async def hotdog_cmd(ctx):
    async for message in ctx.channel.history(limit=200):
        if (message.attachments != None and message.attachments != []):
            src = message.attachments[0].url
            break
        elif message.content.startswith('https://cdn.discordapp.com/attachments/'):
            src = message.content
            break
    r = requests.get(src, stream=True, timeout=10)
    with open(f"temp/temp_pic.png", 'wb') as output:
        output.write(r.content)
    img = Image.open('temp/temp_pic.png')
    prob = predict_image(img, MODEL_PATH)
    try:
        os.remove('temp/temp_pic.png')
    except:
        print("Error, temporary picture was already removed or never existed")
    if prob == 1:
        await ctx.send('\U0001F32D\U0000200D\U0000274C')
    else:
        await ctx.send('\U0001F32D')

bot.run(TOKEN)
