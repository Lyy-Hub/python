from test.chat import *

turing = Tuling(api_key='3d51ffaa9a344c228887b6e4ebd6c417')
bot = Bot()

#只在某个群内聊天，比如群名是 “python交流群”
xianding = bot.groups().search('python交流群')
@bot.register(chats=xianding)
def communite(msg):
    turing.do_reply(msg)

bot.join()