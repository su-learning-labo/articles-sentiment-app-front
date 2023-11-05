import streamlit as st
import requests
from datetime import datetime

BASE_URL = 'https://articles-sentiment-app-back.onrender.com'

# ニュースデータの呼び出し
def get_article_data():
    url = BASE_URL + '/articles/'
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        data = pd.DataFrame(data)
        return data


# 感情分析結果の呼び出し
def get_sentiment_data():
    url = 'http://127.0.0.1:8000/sentiments/'
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        data = pd.DataFrame(data)
        return data


def get_data():
    article = get_article_data()
    sentiment = get_sentiment_data()
    data = pd.merge(article, sentiment, left_on='id', right_on='news_id')
    df = data.filter([
        'title', 'description', 'url', 'published_at', 'fetched_at', 'ranking', 'sentiment', 'sentiment_score'
    ])
    return df


def get_year_from_data():
    df = get_article_data()
    str_date = df['fetched_at'][0]
    date = datetime.datetime.strptime(str_date, '%Y-%m-%dT%H:%M:%S.%f')
    return date.date()


def parse_date(str_date):
    date = datetime.datetime.strptime(str_date, '%Y-%m-%dT%H:%M:%S.%f')
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
    st.sidebar.header('出力条件設定')
    # 冒頭データ（サンプル）表示
    st.write(get_article_data().head())
    min_date = get_year_from_data()
    date_from = st.sidebar.date_input('いつから', min_date)
    date_to = st.sidebar.date_input('いつまで')
    st.sidebar.write('')
    # sentiment_filter = st.sidebar.radio('感情選択', options=['Positice', 'Negative'])

    st.write('条件設定が完了したら下の表示ボタンを押してください')
    st.write('')

    if st.button('分析開始'):
        df = get_data()
        st.write(df)

        df['published_at'] = pd.to_datetime(df['published_at'])
        df['fetched_at'] = pd.to_datetime(df['fetched_at'])
        # df.set_index('fetched_at', inplace=True)
        # df = df.groupby('fetched_at').count()
        # st.dataframe(df)
        # st.bar_chart(df, x='fetched_at', y='sentiment_score')

        st.write(df.dtypes)


if __name__ == "__main__":
    main()