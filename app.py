import calendar
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
        st.session_state['session_state'] = SessionState(views_selector=False, selected_option_btn=False)
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
        'title', 'description', 'url', 'published_at', 'source_name', 'sentiment', 'score'
    ])
    return df


def parse_date(str_date):
    return dt.strptime(str_date, '%Y-%m-%dT%H:%M:%S.%f').date()


# 記事の最初の段落のみを抽出する処理
def extract_first_paragraph(text):
    try:
        paragraphs = text.split('。')
        return paragraphs[0] if paragraphs else ''
    except AttributeError:
        return text[:100]
    except TypeError:
        return text[:100]


# WordCloud
def get_wordcloud(text, mask='レンズ'):
    mask_path = f'static/img/{WORDCLOUD_MASK[mask]}.png'
    mask_array = np.array(Image.open(mask_path))
    wordcloud = WordCloud(
        background_color='white',
        font_path='static/Sawarabi_Gothic/SawarabiGothic-Regular.ttf',
        mask=mask_array,
        max_words=100,
        height=200
    ).generate_from_frequencies(text)
    fig, axes = plt.subplots(figsize=(5, 3))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    return fig


# streamlitメインアプリ
def main():
    st.set_page_config(layout="wide")

    # 表示条件用のキャッシュ
    session_state = get_session_state()

    st.title('ニュース記事感情分析アプリ')
    st.write('')

    # 説明文
    st.write(
        """
        NewsAPIから日本のトレンドニュースを取得。
        ニュース記事に基づく感情分析とともに、テキスト抽出や推移変化を用いて可視化・表示するアプリです。
        """
    )

    # データの取得と前処理
    df = get_merged_data()
    df['published_at'] = pd.to_datetime(df['published_at']).dt.date
    df['year'] = df['published_at'].apply(lambda x: x.year).astype('str')
    df['month'] = df['published_at'].apply(lambda x: x.month).astype('str')
    df['day_of_week'] = df['published_at'].apply(lambda x: calendar.weekday(x.year, x.month, x.day))

    # 出力条件設定 (サイドバー）
    st.sidebar.header('出力条件設定')
    views_selector = st.sidebar.radio(
        '表示切り替え',
        options=('トレンド', '個別記事'),
        horizontal=True
    )

    # Session_Stateの管理
    if views_selector == 'トレンド':
        session_state.views_selector = False
    elif views_selector == '個別記事':
        session_state.views_selector = True

    min_date = parse_date(pd.DataFrame(api_request('articles'))['fetched_at'][0])

    if not session_state.views_selector:
        # 対象期間フィルター
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

        # sentiment
        sent_filter = st.sidebar.multiselect(
            label='分類で絞る',
            options=['positive', 'negative', 'neutral'],
            default=['positive', 'negative', 'neutral']
        )

        df = df.query('@date_from <= published_at <= @date_to').query('sentiment in @sent_filter')
        # 計算用パラメータの取得 >> positive と negativeの件数取得と比率計算
        total_news = df['sentiment'].count()
        positive_count = df[df['sentiment'] == 'positive']['sentiment'].count()
        negative_count = df[df['sentiment'] == 'negative']['sentiment'].count()
        neutral_count = df[df['sentiment'] == 'neutral']['sentiment'].count()
        positeve_rate = positive_count / total_news
        neutral_rate = neutral_count / total_news
        negative_rate = negative_count / total_news

        # 記事一覧の表示
        st.write('---')
        head_con = st.container()
        head_con.subheader('データサマリー')
        col1, col2, col3, col4, col5 = head_con.columns([1, 1, 1, 1, 3], gap='large')

        col1.metric(label='## Total Articles', value=f'{total_news} 件')
        col2.metric(label='Positive Rate', value='{:.1%}'.format(positeve_rate))
        col3.metric(label='Neutral Rate', value='{:.1%}'.format(neutral_rate))
        col4.metric(label='Negative Rate', value='{:.1%}'.format(negative_rate))

        # sentimentの割合帯グラフ用データ
        cross = pd.crosstab(index=df['sentiment'], columns='rate', normalize=True)
        fig = plt.figure(figsize=(10, 1))
        axes = plt.axes()
        # Stacking the bars horizontally
        axes.barh('Sentiments', positeve_rate, color='lightseagreen')
        axes.barh('Sentiments', neutral_rate, left=positeve_rate, color='darkslategray')
        axes.barh('Sentiments', negative_rate, left=positeve_rate + neutral_rate, color='lightcoral')

        # Adding data labels
        plt.text(positeve_rate / 2, 0, f'Positive: {positeve_rate:.1%}', va='center', ha='center', color='white')
        plt.text(positeve_rate + neutral_rate / 2, 0, f'Neutral: {neutral_rate:.1%}', va='center', ha='center', color='white')
        plt.text(positeve_rate + neutral_rate + negative_rate / 2, 0, f'Negative: {negative_rate:.1%}', va='center', ha='center', color='white')

        # Hiding the axes
        plt.axis('off')

        # Display the horizontal stacked bar chart
        st.pyplot(fig)

        # ヒストグラム
        st.write('---')
        st.subheader('ヒストグラムによる可視化')
        subset = df.filter(['published_at', 'sentiment', 'score'])
        negative_col = subset['sentiment'] == 'negative'
        positive_col = subset['sentiment'] == 'positive'

        fig = plt.figure(figsize=(10, 5))
        axes = plt.axes()
        ax = sns.distplot(a=subset[positive_col]['score'], kde=True, label='Positive')
        sns.distplot(a=subset[negative_col]['score'], ax=ax, kde=True, label='Negative')
        plt.legend()
        st.pyplot(fig)

        # 時系列変化の表示グラフ
        st.write('---')
        st.subheader('時系列での変化')
        fig = plt.figure(figsize=(10, 5))
        axes = plt.axes()
        sns.lineplot(x='published_at', y='score', hue='sentiment', data=df, palette='bright')
        axes.legend(loc="best")
        axes.set_xlabel(None)
        fig.autofmt_xdate(rotation=30)
        plt.tight_layout()
        st.pyplot(fig)

        st.write('---')

        # 曜日ごとの違いを表示（箱ひげ）
        st.subheader('曜日ごとの分布')
        subset = df[['day_of_week', 'sentiment', 'score']]

        fig, axes = plt.subplots()
        sns.boxplot(data=subset, x='day_of_week', y='score', hue='sentiment', palette='pastel', linewidth=0.7)
        axes.legend(loc="best")
        axes.set_xticklabels(['月', '火', '水', '木', '金', '土', '日'])
        axes.set_xlabel(None)
        plt.tight_layout()
        st.pyplot(fig)



        # 一覧データフレーム
        st.subheader('記事一覧')
        st.dataframe(df.set_index('published_at'))

    # ここから個別の記事分析・可視化
    if session_state.views_selector:

        # セッション状態の確認
        if st.sidebar.button('データを表示', use_container_width=True):
            session_state.selected_option_btn = True

        if session_state.selected_option_btn:
            # データの取得
            st.write('')

            df['title'] = df['title'].str.split('-').str[0].str.split('|').str[0]

            negative_df = df.query('sentiment == "negative"')
            positive_df = df.query('sentiment == "positive"')
            negative_cnt = negative_df.groupby(['published_at']).count()['sentiment']
            positive_cnt = positive_df.groupby(['published_at']).count()['sentiment']

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
                session_state.selected_option_btn = True

            if session_state.selected_option_btn:

                # テキスト抽出
                first_paragraphs = df['description'].apply(extract_first_paragraph)

                # 抽出した段落を結合
                combined_text = ' '.join(first_paragraphs)[:1000]

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

                    # 推移
                    st.write('')
                    st.write('-- 時系列変化 --')
                    fig_plot = plt.figure(figsize=(10, 4))
                    plt.plot(negative_cnt, label='negative_articles')
                    plt.plot(positive_cnt, label='positive_articles')
                    plt.legend()
                    st.pyplot(fig_plot)

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
