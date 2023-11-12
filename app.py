import pandas as pd
import streamlit as st
import requests
from datetime import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
from wordcloud import WordCloud
from collections import Counter

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


def morphological(text):
    url = base_url + '/analyze-text/'
    response = requests.post(url, json={"text": text})

    if response.status_code == 200:
        return response.json()


# WordCloud
def generate_wordcloud(text):
    wordcloud = WordCloud(
        background_color='white',
        font_path='static/Yusei_Magic/YuseiMagic-Regular.ttf',
        width=800, height=400
    ).fit_words(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)


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

        # WordCloudの前処理
        text = list(output['title'])
        text = ','.join(text)
        # st.write(text)
        nouns = morphological(text)

        # 除外するキーワード指定
        exclude_words = ['年', '月', '日', '|', 'com', 'Powered', 'by', 'ｋｍ', '東スポ', '読売', '.co.jp', 'NEWS', 'さん', 'JBpress', '%', '％', 'さま','位', '新聞', '毎日']
        output_text = []
        for noun in nouns['nouns']:
            if noun not in exclude_words:
                output_text.append(noun)

        words_count = Counter(output_text)
        result = words_count.most_common()[:30]
        st.write(pd.DataFrame(result))
        generate_wordcloud(dict(result))

        # fig, ax = plt.subplots()
        # sns.violinplot(data=df, x='fetched_at', y='score', hue='sentiment')
        # st.pyplot(fig)


if __name__ == "__main__":
    main()
