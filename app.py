import pandas as pd
import streamlit as st
import requests
from datetime import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
import sys


if st.secrets.ENVIRONMENT == 'production':
    base_url = 'https://articles-sentiment-app-back.onrender.com'
else:
    base_url = 'http://localhost:8000'


# ニュースデータの呼び出し
def get_article_data():
    url = base_url + '/articles/'
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        data = pd.DataFrame(data)
        return data


# 感情分析結果の呼び出し
def get_sentiment_data():
    url = base_url + '/sentiments/'
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        data = pd.DataFrame(data)
        return data


def get_data():
    article = get_article_data()
    sentiment = get_sentiment_data()
    data = pd.merge(article, sentiment, left_on='id', right_on='article_id')
    df = data.filter([
        'title', 'description', 'url', 'published_at', 'fetched_at', 'ranking', 'sentiment', 'score'
    ])
    return df


def get_year_from_data():
    df = get_article_data()
    str_date = df['fetched_at'][0]
    date = dt.strptime(str_date, '%Y-%m-%dT%H:%M:%S.%f')
    return date.date()


def parse_date(str_date):
    date = dt.strptime(str_date, '%Y-%m-%dT%H:%M:%S.%f')
    return date.date()


def sentiment_graph():
    pass


# streamlitメインアプリ
def main():

    st.set_page_config(layout="wide")

    st.title('ニュース記事感情分析アプリ')
    st.write('')
    st.write(
        """
        このアプリは、NewsAPIから日本のトレンドニュースを取得してその内容を感情分析し、
        ポジティブ・ネガティブに分類し、可視化・表示するアプリです。
        """
    )
    st.write('')
    st.header('出力条件設定')
    # 冒頭データ（サンプル）表示
    # st.write(get_article_data())

    min_date = get_year_from_data()
    header_left, header_mid, header_right = st.columns([2.5, 2.5, 5])
    date_from = header_left.date_input('いつから', min_date)
    date_to = header_mid.date_input('いつまで')
    st.write('')

    st.write('条件設定が完了したら下の表示ボタンを押してください')
    st.write('')

    if st.button('分析開始'):
        # データの取得
        df = get_data().filter(['title', 'description', 'published_at', 'fetched_at', 'sentiment', 'score', 'ranking'])

        df['published_at'] = pd.to_datetime(df['published_at']).dt.date
        df['fetched_at'] = pd.to_datetime(df['fetched_at']).dt.date
        df.set_index('published_at', inplace=True)
        output = df.query('@date_from <= fetched_at <= @date_to')

        st.write(output)

        fig, ax = plt.subplots()
        sns.violinplot(data=df, x='fetched_at', y='score', hue='sentiment')
        st.pyplot(fig)


if __name__ == "__main__":
    main()
