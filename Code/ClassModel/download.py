import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from bs4 import BeautifulSoup
import requests
import re
import os


cid ='cba91ee3b4f94a1b867d58cfcd61a191'
secret ='d4caa849d8274399812cd510e628ef57'

client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)


def get_album_tracks(uri_info):
    uri = []
    track = []
    duration = []
    explicit = []
    track_number = []
    name = []
    one = sp.album_tracks(uri_info, market='US')
    df1 = pd.DataFrame(one)

    for i, x in df1['items'].items():
        uri.append(x['uri'])
        track.append(x['name'])
        duration.append(x['duration_ms'])
        explicit.append(x['explicit'])
        track_number.append(x['track_number'])
        name.append(x['name'])

    df2 = pd.DataFrame({
        'uri': uri,
        'track': track,
        'duration_ms': duration,
        'explicit': explicit,
        'name':name,
        'track_number': track_number})

    return df2


def get_track_info(df):
    danceability = []
    energy = []
    key = []
    loudness = []
    speechiness = []
    acousticness = []
    instrumentalness = []
    liveness = []
    valence = []
    tempo = []
    for i in df['uri']:
        for x in sp.audio_features(tracks=[i]):
            danceability.append(x['danceability'])
            energy.append(x['energy'])
            key.append(x['key'])
            loudness.append(x['loudness'])
            speechiness.append(x['speechiness'])
            acousticness.append(x['acousticness'])
            instrumentalness.append(x['instrumentalness'])
            liveness.append(x['liveness'])
            valence.append(x['valence'])
            tempo.append(x['tempo'])

    df2 = pd.DataFrame({
        'danceability': danceability,
        'energy': energy,
        'key': key,
        'loudness': loudness,
        'speechiness': speechiness,
        'acousticness': acousticness,
        'instrumentalness': instrumentalness,
        'liveness': liveness,
        'valence': valence,
        'tempo': tempo})

    return df2

def merge_frames(df1, df2):
    df3 = df1.merge(df2, left_index= True, right_index= True)
    return df3

def scrape_song_lyrics(url):
    page = requests.get(url)
    html = BeautifulSoup(page.text, "html.parser")
    lyrics = html.select_one(
        'div[class^="lyrics"], div[class^="SongPage__Section"]'
    ).get_text(separator="\n")
    # remove identifiers like chorus, verse, etc
    lyrics = re.sub(r"[\(\[].*?[\)\]]", "", lyrics)
    # remove empty lines
    lyrics = os.linesep.join([s for s in lyrics.splitlines() if s])
    return lyrics

lyrics = scrape_song_lyrics('https://genius.com/Maroon-5-girls-like-you-remix-lyrics')
lyrics = lyrics[94:-460]

def scrape_lyrics(artistname, songname):
    artistname2 = str(artistname.replace(' ','-')) if ' ' in artistname else str(artistname)
    songname2 = str(songname.replace(' ','-')) if ' ' in songname else str(songname)
    page = requests.get('https://genius.com/'+ artistname2 + '-' + songname2 + '-' + 'lyrics')
    html = BeautifulSoup(page.text, 'html.parser')
    lyrics1 = html.find("div", class_="lyrics")
    lyrics2 = html.find("div", class_="Lyrics__Container-sc-1ynbvzw-2 jgQsqn")
    if lyrics1:
        lyrics = lyrics1.get_text()
    elif lyrics2:
        lyrics = lyrics2.get_text()
    elif lyrics1 == lyrics2 == None:
        lyrics = None
    return lyrics

#function to attach lyrics onto data frame
#artist_name should be inserted as a string
def lyrics_onto_frame(df1, artist_name):
    for i,x in enumerate(df1['track']):
        test = scrape_lyrics(artist_name, x)
        df1.loc[i, 'lyrics'] = test
    return df1

blonde_df1_tracks = get_album_tracks('https://open.spotify.com/album/3T4tUhGYeRNVUGevb0wThu?si=w7wvsAMnS1GWrBbEkxk0Pg')
blonde_df2_metadata = get_track_info(blonde_df1_tracks)
df1 = merge_frames(blonde_df1_tracks, blonde_df2_metadata)
lyrics_onto_frame(df1, 'Ed-sheeran')

from lyricsgenius import Genius

cid_G = 'hIO0J7nECFUJIrv9Yrb3VdDWaHSYMzZs92J-oORdj854sDc1DSHYwOqHE95FTpei'
secret_G = '8xiwiU8NcuYOoguFVM99_XVqSsKGNnoA9xLW6UsOKFcKPT0mjLQ7diUlPyeoBnqanS0R8rrprHnSRQJD3cBApw'
token = 'i5xspGcgWHUnwUXB88GxRkJD6EejJYXBpoNeMmJZ6ZUwDC7ljbaYkN-WFvi0P8jG'
uri = 'http://www.google.com'


genius = Genius(token)
lyric = genius.search_artist('Ed Sheeran',max_songs=5)
lyrics = lyric.save_lyrics()


api = genius.Genius(geniusCreds)
artist = api.search_artist(artist_name, max_songs=5)

