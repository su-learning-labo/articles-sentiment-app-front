import pandas as pd
import streamlit as st
import requests
from datetime import datetime as dt
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
from PIL import Image
import numpy as np


# 定数
BASE_URL_PRODUCTION = 'https://articles-sentiment-app-back.onrender.com'
BASE_URL_DEVELOPMENT = 'http://localhost:8000'
API_ENDPOINTS = {
    'articles': '/articles/',
    'sentiments': '/sentiments',
    'analyze_text': '/analyze-text/',
    'extract_nouns': '/extract-nouns/',
}
WORDCLOUD_MASK = {
    '正方形': 'lens',
    'いいね': 'good',
    '日本地図': 'japanesemap',
    'ゲッコー': 'tokage',
    'メンフクロウ': 'menfukurou',
}


# セッション状態管理クラスの定義
class SessionState:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


# セッション状態の取得or初期化
def get_session_state():
    if 'session_state' not in st.session_state:
        st.session_state['session_state'] = SessionState(selected_option_btn1=False, selected_option_btn2=False)
    return st.session_state['session_state']


def get_base_url():
    return BASE_URL_PRODUCTION if st.secrets.ENVIRONMENT == 'production' else BASE_URL_DEVELOPMENT


def api_request(endpoint, method='get', data=None):
    url = get_base_url() + API_ENDPOINTS[endpoint]
    response = getattr(requests, method)(url, json=data)
    if response.status_code == 200:
        return response.json()


def get_merged_data():
    article_data = pd.DataFrame(api_request('articles'))
    sentiment_data = pd.DataFrame(api_request('sentiments'))
    df = pd.merge(article_data, sentiment_data, left_on='id', right_on='article_id')
    df = df.filter([
        'title', 'description', 'published_at', 'fetched_at', 'sentiment', 'score'
    ])
    return df


def parse_date(str_date):
    return dt.strptime(str_date, '%Y-%m-%dT%H:%M:%S.%f').date()


# 記事の最初の段落のみを抽出する処理
def extract_first_paragraph(text):
    paragraphs = text.split('。')
    return paragraphs[0] if paragraphs else ''


# WordCloud
def get_wordcloud(text, mask='レンズ'):
    mask_path = f'static/img/{WORDCLOUD_MASK[mask]}.png'
    mask_array = np.array(Image.open(mask_path))
    wordcloud = WordCloud(
        background_color='white',
        font_path='static/Sawarabi_Gothic/SawarabiGothic-Regular.ttf',
        mask=mask_array,
        height=200
    ).generate_from_frequencies(text)
    fig, ax = plt.subplots(figsize=(5, 3))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    return fig


# streamlitメインアプリ
def main():
    st.set_page_config(layout="wide")

    st.title('ニュース記事感情分析アプリ')
    st.write('')

    # 説明文
    st.write(
        """
        NewsAPIから日本のトレンドニュースを取得。
        ニュース記事に基づく感情分析とともに、テキスト抽出や推移変化を用いて可視化・表示するアプリです。
        """
    )
    st.write('')

    # 出力条件設定
    st.sidebar.header('出力条件設定')
    min_date = parse_date(pd.DataFrame(api_request('articles'))['fetched_at'][0])

    date_range = st.sidebar.date_input(
        "対象期間を選択してください",
        value=(min_date, dt.today()),
        min_value=min_date,
        max_value=dt.today(),
    )
    date_from = date_range[0]
    try:
        date_to = date_range[1]
    except IndexError:
        date_to = date_range[0]

    # 表示条件用のキャッシュ
    session_state = get_session_state()

    # セッション状態の確認
    if st.sidebar.button('データを表示', use_container_width=True):
        session_state.selected_option_btn1 = True

    if session_state.selected_option_btn1:
        # データの取得
        st.write('')
        st.write('-- ニュース記事  from NewsAPI --')
        df = get_merged_data()\
            .filter(['title', 'description', 'published_at', 'fetched_at', 'sentiment', 'score'])

        # データの前処理とデータ表示
        df['published_at'] = pd.to_datetime(df['published_at']).dt.date
        df['fetched_at'] = pd.to_datetime(df['fetched_at']).dt.date
        df.set_index('published_at', inplace=True)
        df = df.query('@date_from <= published_at <= @date_to')
        df['title'] = df['title'].str.split('-').str[0].str.split('|').str[0]
        st.dataframe(df)

        # 情報掲載 - サイドバー
        with st.sidebar:
            st.write('---')
            st.subheader('< info >')
            st.write(f'記事数: {df.shape[0]} 件')
            st.write(f"- ポジティブ: {df[df['sentiment']=='positive'].shape[0]} 件")
            st.write(f"- ネガティブ: {df[df['sentiment']=='negative'].shape[0]} 件")
            st.write(f"- ニュートラル: {df[df['sentiment']=='neutral'].shape[0]} 件")

        st.write('')
        st.sidebar.write('---')
        st.sidebar.write('データを確認のうえ、実行してください')
        # セッション状態の確認
        if st.sidebar.button('集計処理を実行', use_container_width=True):
            session_state.selected_option_btn2 = True

        if session_state.selected_option_btn2:

            # テキスト抽出
            first_paragraphs = df['description'].apply(extract_first_paragraph)

            # 抽出した段落を結合
            combined_text = ' '.join(first_paragraphs)

            # グラフ（割合）

            st.write('## 可視化フィールド')
            tab1, tab2, tab3 = st.tabs(['グラフ', 'テキスト抽出', 'ワードクラウド'])
            with tab1:
                # ヒストグラム（密度関数）
                st.write('-- ヒストグラム --')
                subset = df.filter(['published_at', 'sentiment', 'score'])

                fig = plt.figure(figsize=(10, 4))
                nega = subset['sentiment'] == 'negative'
                posi = subset['sentiment'] == 'positive'
                sns.distplot(subset[nega]['score'], label='negative')
                sns.distplot(subset[posi]['score'], label='positive')
                plt.legend()
                st.pyplot(fig)

            with tab2:
                body_left, body_right = st.columns([1, 1])
                body_left.caption('<原文>')
                body_left.write(combined_text)

                # テキスト抽出と分析
                body_right.caption('<テキスト抽出と分類>')
                analyze_text = api_request('analyze_text', method='post', data={'text': combined_text})
                df_tokens = pd.DataFrame(analyze_text['tokens'])
                df_tokens.set_index('token_no', inplace=True)
                body_right.dataframe(df_tokens)

            with tab3:
                wc_select = st.selectbox(
                    'ワードクラウドの出力イメージ選択',
                    options=WORDCLOUD_MASK,
                )

                # ワードクラウド
                nouns = api_request('extract_nouns', method='post', data={'text': combined_text})['nouns']

                # 除外するキーワード指定
                exclude_words = [
                    '年', '月', '日', '時', '大', '週間', 'ｋｍ', 'さん', '%', '％', 'さま', '位', 'nolink', '米'
                ]
                filtered_nouns = [noun for noun in nouns if noun not in exclude_words]
                words_counts = Counter(filtered_nouns)

                st.caption('<ワードクラウドによる可視化>')
                col_left, col_center, col_right = st.columns([.1, .8, .1])
                col_center.pyplot(get_wordcloud(words_counts, wc_select))


if __name__ == "__main__":
    main()
