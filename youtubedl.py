#@title Download Youtube WAV
from __future__ import unicode_literals
import yt_dlp
import ffmpeg
import sys

ydl_opts = {
    'format': 'bestaudio/best',
#    'outtmpl': 'output.%(ext)s',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
    }],
    "outtmpl": 'trinhan',  # this is where you can edit how you'd like the filenames to be formatted
}
def download_from_url(url):
    ydl.download([url])
    # stream = ffmpeg.input('output.m4a')
    # stream = ffmpeg.output(stream, 'output.wav')


with yt_dlp.YoutubeDL(ydl_opts) as ydl:
      url = "https://www.youtube.com/watch?v=eJnZBHxuTwM" #@param {type:"string"}
      download_from_url(url)