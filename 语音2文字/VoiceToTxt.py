import os
import pyaudio
import wave
from aip import AipSpeech

""" 你的 APPID AK SK """
APP_ID = '16891506'
API_KEY = '0En7WdcWRfKo08SjurFzMVff'
SECRET_KEY = '2GNXrtj8bcu1GXizaHrwFoO3rklQBH6K'

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 10
WAVE_OUTPUT_FILENAME = "yuyin.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* 开始录音（10秒）说普通话，别说滕普！！！")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("* 录音结束")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)


# 读取文件
def get_file_content(filePath):
    cmd_str = "ffmpeg -y  -i %s  -acodec pcm_s16le -f s16le -ac 1 -ar 16000 %s.pcm"%(filePath,filePath)
    os.system(cmd_str)  # 调用系统命令ffmpeg,传入音频文件名即可
    with open(filePath + ".pcm", 'rb') as fp:
        return fp.read()

# 识别本地文件
a = client.asr(get_file_content('yuyin.wav'), 'pcm', 16000, {
    'dev_pid': 1536,
})
# 移除音频文件
os.remove("yuyin.wav")
os.remove("yuyin.wav.pcm")
test_str = "".join(a.get('result'))
with open("文本.txt", "w") as f:
    f.write(test_str)  # 这句话自带文件关闭功能，不需要再写f.close()

print(test_str)