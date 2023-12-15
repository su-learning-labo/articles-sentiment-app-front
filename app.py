import io

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib

import requests
import spacy
from spacy import displacy
import ginza

import calendar
from datetime import datetime as dt
import datetime
import nlplot

# 定数
BASE_URL_PRODUCTION = 'https://articles-sentiment-app-back.onrender.com'
BASE_URL_DEVELOPMENT = 'http://localhost:8000'
API_ENDPOINTS = {
    'articles': '/articles/',
    'sentiments': '/sentiments',
    'analyze_text': '/analyze-text/',
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
    df['published_at'] = pd.to_datetime(df['published_at']).dt.date
    df['year'] = df['published_at'].apply(lambda x: x.year).astype('str')
    df['month'] = df['published_at'].apply(lambda x: x.month).astype('str')
    df['day_of_week'] = df['published_at'].apply(lambda x: calendar.weekday(x.year, x.month, x.day))
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


nlp = spacy.load('ja_ginza')


def analyze_text(text):
    doc = nlp(text)
    ginza.set_split_mode(nlp, 'C')
    return ' '.join([token.lemma_ for token in doc if token.pos_ != 'ADP'])


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

    # 出力条件設定 (サイドバー）
    st.sidebar.header('出力条件設定')
    views_selector = st.sidebar.radio(
        '表示切り替え',
        options=('トレンド', '個別分析'),
        horizontal=True
    )

    # Session_Stateの管理
    if views_selector == 'トレンド':
        session_state.views_selector = False
    elif views_selector == '個別分析':
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

        st.sidebar.write('')
        show_trends_button = st.sidebar.button(
            'トレンドを表示',
            type='secondary',
        )

        if show_trends_button:

            # 記事一覧の表示
            st.write('---')
            head_con = st.container()
            head_con.subheader('データサマリー')
            col1, col2, col3, col4, col5 = head_con.columns([1, 1, 1, 1, 3], gap='large')

            col1.metric(label='Total Articles', value=f'{total_news} 件')
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

            st.write('---')
            # 一覧データフレーム
            st.subheader('記事一覧')
            st.dataframe(df.set_index('published_at'))

    # ここから個別の記事分析・可視化
    if session_state.views_selector:

        date_list = df['published_at'].apply(lambda x: x.strftime('%Y/%m/%d')).unique().tolist()

        # セッション状態の確認
        if session_state.views_selector:

            st.sidebar.write('')
            select_analysis = st.sidebar.selectbox(
                '可視化方法を選択してください',
                options=[
                    '可視化方法を選択してください,'
                    '共起ネットワーク',
                    'サンバースト・チャート',
                    'ベクトル化',
                    'ワードクラウド',
                    'ツリーマップ',
                    '係り受け分析',
                ]
            )

            if select_analysis in ['共起ネットワーク', 'サンバースト・チャート']:

                # データの取得
                df = get_merged_data()
                target_df = df[['title', 'description', 'url', 'source_name', 'sentiment', 'score', 'published_at']]

                with open('static/stopwords.txt') as f:
                    stopwords_list = f.read().splitlines()

                # 対象の日付の絞りこみ
                st.sidebar.write('')
                date_selector = st.sidebar.date_input(
                    '対象の日付を選択してください',
                    datetime.date(2023, 11, 10)
                )

                target_df = target_df.query('published_at == @date_selector')

                min_index_value = target_df.index[0]
                max_index_value = target_df.index[-1]
                id_selector = st.sidebar.number_input(
                    '記事IDを選択してください',
                    min_value=min_index_value,
                    max_value=max_index_value,
                )

                # フィルタリング
                target_df = target_df[target_df.index == id_selector]
                st.write('### 対象記事')
                st.markdown(target_df["description"].values[0])

                # 分析対象の記事idを選択させる
                target_df['analyzed_text'] = target_df['description'].apply(analyze_text)

                # 個別分析の可視化
                npt = nlplot.NLPlot(target_df, target_col='analyzed_text')

                if select_analysis == '共起ネットワーク':
                    # Co-occurrence networks
                    npt.build_graph(stopwords=stopwords_list, min_edge_frequency=1)

                    fig_co_network = npt.co_network(
                        title='Co-occurrence network',
                        sizing=100,
                        node_size='adjacency_frequency',
                        color_palette='hls',
                        width=1100,
                        height=700,
                        save=False
                    )
                    st.plotly_chart(fig_co_network, use_container_width=True, sharing='streamlit')

                elif select_analysis == 'サンバースト・チャート':

                    # sunburst chart
                    fig_sunburst = npt.sunburst(
                        title='sunburst chart',
                        colorscale=True,
                        color_continuous_scale='Oryel',
                        width=1000,
                        height=800,
                        save=False
                    )
                    st.plotly_chart(fig_sunburst, use_container_width=True, sharing='streamlit')


            elif select_analysis in ['ベクトル化',  'ワードクラウド', 'ツリーマップ']:

                # 対象期間の表示
                st.sidebar.write('')
                date_selector = st.sidebar.date_input(
                    '対象とする日付を選択してください',
                    value=datetime.date(2023, 11, 10),
                    min_value=min_date,
                    max_value=dt.today()
                )

                # データの取得
                st.write('')
                df = get_merged_data()
                target_df = df[['title', 'description', 'url', 'source_name', 'sentiment', 'score', 'published_at']]
                target_df = target_df.query('published_at == @date_selector')
                target_df['analyzed_text'] = target_df['description'].apply(analyze_text)

                st.sidebar.write('')

                # 分析対象の記事idを選択させる
                id_selector = st.sidebar.slider(
                    '記事IDを選択してください',
                    target_df.index.values.tolist()[0],
                    target_df.index.values.tolist()[-1],
                    (target_df.index.values.tolist()[0], target_df.index.values.tolist()[2])
                )
                mask = [i for i in range(id_selector[0], id_selector[1]+1)]
                target_df = target_df[target_df.index.isin(mask)]

                if target_df is not None:
                    st.write(f'## 記事一覧 ({target_df.shape[0]} 件)')
                    st.caption(f'{date_selector.strftime("%Y年%m月%d日")} から抽出した {target_df.shape[0]} 件')
                    st.dataframe(target_df)
                    target_df['title'] = target_df['title'].str.split('-').str[0].str.split('|').str[0]
                else:
                    st.error("Sorry, failed to retrieve article. Please select another date.")

                negative_df = target_df.query('sentiment == "negative"')
                positive_df = target_df.query('sentiment == "positive"')
                negative_cnt = negative_df.groupby(['published_at']).count()['sentiment']
                positive_cnt = positive_df.groupby(['published_at']).count()['sentiment']

                selected_option_btn = st.sidebar.button(
                    '解析実行',
                    type='primary',
                    use_container_width=True
                )

                if selected_option_btn:
                    session_state.selected_option_btn = True

                if session_state.selected_option_btn:

                    with open('static/stopwords.txt') as f:
                        stopwords_list = f.read().splitlines()

                    st.write(stopwords_list)
                    # 個別分析の可視化
                    npt = nlplot.NLPlot(target_df, target_col='analyzed_text')


                    fig_unigram = npt.bar_ngram(
                        title='N-gram bar chart',
                        xaxis_label='word_count',
                        yaxis_label='word',
                        ngram=1,
                        top_n=50,
                        width=400,
                        height=800,
                        color=None,
                        horizon=True,
                        stopwords=stopwords_list,
                        verbose=False,
                        save=False
                    )

                    st.plotly_chart(fig_unigram, use_container_width=True, sharing='streamlit')

                    # N-gram tree Map
                    fig_treemap = npt.treemap(
                        title='Tree map',
                        ngram=1,
                        top_n=50,
                        width=1300,
                        height=600,
                        stopwords=stopwords_list,
                        verbose=False,
                        save=False
                    )

                    st.plotly_chart(fig_treemap, use_container_width=True, sharing='streamlit')

                    # 単語数の分布
                    fig_histgram = npt.word_distribution(
                        title='word distribution',
                        xaxis_label='count',
                        yaxis_label='',
                        width=1000,
                        height=500,
                        color=None,
                        template='plotly',
                        bins=None,
                        save=False
                    )

                    st.plotly_chart(fig_histgram, use_container_width=True, sharing='streamlit')

                    # wordcloud
                    fig_wc = npt.wordcloud(
                        # width=1000,
                        # height=600,
                        max_words=300,
                        max_font_size=300,
                        colormap='tab20_r',
                        stopwords=stopwords_list,
                        mask_file='static/img/japanesemap.png',
                    )

                    plt.figure(figsize=(10, 7))
                    plt.imshow(fig_wc)
                    plt.axis('off')

                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    st.image(buf, use_column_width=True)


if __name__ == "__main__":
    main()
