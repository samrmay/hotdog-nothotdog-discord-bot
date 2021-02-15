# Hotdog or not hotdog discord bot
Classification model for hotdog/not hotdog in the form of a discord bot.  
1. Invite bot to your guild [here](https://discord.com/api/oauth2/authorize?client_id=810578153390211123&permissions=75776&scope=bot)
2. Post a photo in a text channel
3. Type ```!hotdog``` to find out if the photo you posted is a hotdog or a not hotdog

## Running locally
1. ```pip install -r requirements.txt```
2. Add DISCORD_TOKEN = (your discord bot's token) and MODEL_PATH = path/to/model to .env
3. run main.py

## Model info
ResNet structure trained in PyTorch. Data set was scraped from google image search query results. Download the model .pth [here](https://www.dropbox.com/s/6vg4y3h9b4pp4ri/jian_yang_model.pth?dl=0)