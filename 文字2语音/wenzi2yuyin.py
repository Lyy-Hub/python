from aip import AipSpeech
import  os

APP_ID = '16696544'
API_KEY = '4CHmdojiNN2Mz9IGWMSfRDy5'
SECRET_KEY = 'K61pAR3YFzTlrdk8y5Q6uHMOeRCtAfLr '

def ReadTxtName(rootdir):
    lines = []
    with open(rootdir, 'r') as file_to_read:
        while True:
            line = file_to_read.readline()
            if not line:
                break
            line = line.strip('\n')
            lines.append(line)
    return lines

client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)
path = os.getcwd()
resultpath= path + '\\文字.txt'

result = client.synthesis(ReadTxtName(resultpath), 'zh', 1, {
    'vol': 5, 'per': 4
})

if not isinstance(result, dict):
    with open('语音.mp3', 'wb') as f:
        f.write(result)