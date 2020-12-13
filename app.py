import streamlit as st 
import plotly.express as px 
import pandas as pd
import base64 as b64

def get_table_download_link(df,text="Download CSV file"):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False,encoding="utf-8",sep='\t')
    b64_enc = b64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64_enc}" download="poems.tsv" center="true">{text}</a>'
    return href

@st.cache()
def load_data():
    verses =  pd.read_csv('verses.tsv',sep='\t').dropna()
    verses['poet_with_century'] = verses['poet'] + ' از قرن ' +verses['century'].astype('str')#verses.apply(lambda row: f"{row['poet']} (قرن {row['century']})",axis=1)
    return verses

st.set_page_config(page_title="بررسی استفاده از عبارات مختلف در اشعار فارسی در طول زمان از قرن سوم هجری تا دوران معاصر",layout='wide')
st.title("بررسی استفاده از عبارات مختلف در اشعار فارسی در طول زمان از قرن سوم هجری تا دوران معاصر")
st.write('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
st.markdown('همه‌ی اشعار به همت مجموعه‌ی [گنجور](https://ganjoor.net/) جمع‌آوری و از  [مخزن این پروژه](https://github.com/ganjoor/ganjoor-db) برداشت شده است')
verses = load_data()


poets = sorted(list(set(verses['poet_with_century'])))


words = st.text_input(label="لطفاً کلمه یا کلمات مورد نظرتان را وارد کنید. کلمه‌‌های مختلف را با با ویرگول (،) از هم جدا کنید",value='تلخ، شیرین')
poets_list = st.multiselect(label='لطفاً شاعران مورد نظرتان را انتخاب کنید',options=poets,default=poets)
verses = verses [verses['poet_with_century'].isin(poets_list)]

st.markdown(f' در مجموع {len(verses)} «مصرع» شعر از این شعرا داریم')


groupby_var = st.radio(label='دسته‌بندی بر اساس شاعر یا قرن؟',options=['century','poet_with_century'],index=0,format_func=lambda v: 'قرن' if v=='century' else 'شاعر')

ngram_all = verses.groupby(by=[groupby_var])['verse'].count()#.plot.bar()

cols = st.beta_columns([1,1])
with cols[0]: only_whole_word = st.checkbox(label='کلمه فقط به شکل کامل',value=True)
with cols[1]: compute_proportion = st.checkbox(label=f'نسبت به تعداد کل‌ شعرهای هر {"قرن" if groupby_var=="century" else "شاعر" }',value=True)

df = pd.DataFrame()
words_list=[]
if len(words)>0:
    words_list = words.split('،')
    words_list = [w.strip() for w in words_list]
    for w in words_list:
        ngram = verses[ (verses['verse'].str.contains(' '+w+' ')) | (verses['verse'].str.startswith(w+' ')) | (verses['verse'].str.endswith(' '+w)) ].groupby(by=[groupby_var])['verse'].count() \
            if only_whole_word else verses[ (verses['verse'].str.contains(w))].groupby(by=[groupby_var])['verse'].count()
        ngram = (100 if compute_proportion else ngram_all) * (ngram / ngram_all) # Hack to keep empty groups in the df as 0 values
        

        df[w] = ngram
else:
    ngram = ngram_all

df = df.fillna(value=0)
df[groupby_var] = ngram_all.index


fig = px.bar(df,y=groupby_var,x=words_list,orientation='h',barmode='group',height=(1+len(words_list))*(200 if groupby_var=='century' else 600))


fig.update_layout(
    title=f"نسبت استفاده از کلمات مختلف در شعر فارسی در گذر زمان و بین شعرای مختلف",
    yaxis_title="‌قرن هجری " if groupby_var=='century' else 'شاعر' ,
    xaxis_title=f" {'درصد' if compute_proportion else 'تعداد'} مصراع‌های دارای کلمه‌ی مورد نظر ",
    legend_title="کلمه",
    font=dict(
        family="Courier New, monospace",
        size=12,
        color="RebeccaPurple"
    )
)

fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))


st.plotly_chart(fig,use_container_width=True)

filter_fn = lambda s,w : True if  ( (' '+w+' ' in s or s.startswith(w+' ') or s.endswith(' '+w))  if only_whole_word else w in s ) else False

for w in words_list:
    verses[w] = verses['verse'].apply(lambda s: filter_fn (s,w))

dl_file=pd.DataFrame(columns=verses.columns)
for w in words_list:
    dl_file = pd.concat([dl_file,verses [ verses[w]]])

dl_file = dl_file.drop_duplicates()
dl_file = dl_file.drop('poet_with_century',axis=1)

with st.empty():
    st.markdown('لطفاً تا آماده‌ شدن فایل حاوی اشعار صبر کنید ⌛')
    st.markdown(get_table_download_link(dl_file,text='فایل حاوی شعرهایی که هر یک از کلمات بالا در انها به کار رفته را از اینجا دانلود کنید'),unsafe_allow_html=True)
