
import math
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Saudi Trading Agent Mobile",
    page_icon="🇸🇦",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 4rem; max-width: 760px;}
div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.25); padding: 10px; border-radius: 12px;}
.stButton>button {width:100%; height:48px; font-size:18px;}
</style>
""", unsafe_allow_html=True)

def ema(s,n): return s.ewm(span=n, adjust=False).mean()

def rsi(s,n=14):
    d=s.diff()
    up=d.clip(lower=0)
    dn=-d.clip(upper=0)
    rs=up.ewm(alpha=1/n, adjust=False).mean()/dn.ewm(alpha=1/n, adjust=False).mean()
    return 100-(100/(1+rs))

def atr(df,n=14):
    pc=df["Close"].shift(1)
    tr=pd.concat([
        (df["High"]-df["Low"]).abs(),
        (df["High"]-pc).abs(),
        (df["Low"]-pc).abs()
    ],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def indicators(df):
    d=df.copy()
    for n in [9,20,50,200]:
        d[f"EMA{n}"]=ema(d["Close"],n)
    d["RSI14"]=rsi(d["Close"])
    d["ATR14"]=atr(d)
    d["VOL20"]=d["Volume"].rolling(20).mean()
    d["VOL_RATIO"]=d["Volume"]/d["VOL20"]
    d["HH20"]=d["High"].rolling(20).max().shift(1)
    return d.dropna()

@st.cache_data(ttl=300)
def fetch(ticker, mode):
    if mode=="مضاربة يومية":
        d=yf.download(ticker, period="60d", interval="15m", progress=False, auto_adjust=False)
    else:
        d=yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=False)
    if d is None or d.empty: return pd.DataFrame()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns=d.columns.get_level_values(0)
    return d[["Open","High","Low","Close","Volume"]].dropna()

def score_day(d):
    x=d.iloc[-1]; score=0; reasons=[]
    if x.Close>x.EMA20>x.EMA50: score+=25; reasons.append("الاتجاه اللحظي صاعد")
    if x.EMA9>x.EMA20: score+=15; reasons.append("EMA9 أعلى من EMA20")
    if 52<=x.RSI14<=72: score+=20; reasons.append("الزخم مناسب")
    if x.VOL_RATIO>=1.30: score+=20; reasons.append("السيولة أعلى من المتوسط")
    if x.Close>x.HH20: score+=20; reasons.append("اختراق قمة 20 شمعة")
    action="BUY" if score>=70 else "WATCH" if score>=50 else "WAIT"
    stop=x.Close-1.35*x.ATR14
    risk=max(x.Close-stop,.01)
    return action,score,reasons,float(stop),float(x.Close+1.5*risk),float(x.Close+2.5*risk)

def score_swing(d):
    x=d.iloc[-1]; score=0; reasons=[]
    if x.Close>x.EMA50: score+=20; reasons.append("السعر فوق EMA50")
    if x.EMA50>x.EMA200: score+=25; reasons.append("الاتجاه المتوسط صاعد")
    if 48<=x.RSI14<=68: score+=15; reasons.append("RSI صحي")
    if x.VOL_RATIO>=1.15: score+=15; reasons.append("السيولة فوق المتوسط")
    if x.Close>x.HH20: score+=25; reasons.append("اختراق قمة 20 يوم")
    action="BUY" if score>=70 else "WATCH" if score>=50 else "WAIT"
    stop=min(x.EMA20, x.Close-1.8*x.ATR14)
    risk=max(x.Close-stop,.01)
    return action,score,reasons,float(stop),float(x.Close+2*risk),float(x.Close+3*risk)

st.title("🇸🇦 أداة التداول السعودية")
st.caption("نسخة جوال — تحليل ومراقبة فقط")

mode = st.segmented_control("نوع التداول", ["مضاربة يومية","سوينق"], default="مضاربة يومية")
symbol = st.text_input("رمز السهم", value="2222", placeholder="مثال: 2222")
capital = st.number_input("رأس المال (ر.س)", min_value=1000.0, value=100000.0, step=5000.0)
risk_pct = st.slider("المخاطرة لكل صفقة %", 0.25, 2.0, 1.0, 0.25)

if st.button("حلّل السهم"):
    ticker = symbol if symbol.endswith(".SR") else symbol + ".SR"
    df = fetch(ticker, mode)
    if df.empty:
        st.error("تعذر جلب البيانات. تأكد من الرمز أو جرّب لاحقًا.")
    else:
        d=indicators(df)
        if len(d)<5:
            st.error("البيانات غير كافية.")
        else:
            if mode=="مضاربة يومية":
                action,score,reasons,stop,t1,t2=score_day(d)
            else:
                action,score,reasons,stop,t1,t2=score_swing(d)

            entry=float(d.iloc[-1].Close)
            risk_cash=capital*(risk_pct/100)
            qty=max(math.floor(risk_cash/max(entry-stop,.01)),0)

            st.subheader(f"{ticker} — {mode}")
            c1,c2=st.columns(2)
            c1.metric("الإشارة", action)
            c2.metric("Score", f"{score}/100")
            c3,c4=st.columns(2)
            c3.metric("السعر", f"{entry:.2f}")
            c4.metric("عدد الأسهم", f"{qty:,}")

            c5,c6=st.columns(2)
            c5.metric("وقف الخسارة", f"{stop:.2f}")
            c6.metric("الهدف 1", f"{t1:.2f}")
            st.metric("الهدف 2", f"{t2:.2f}")

            st.write("**الأسباب:**")
            for r in reasons:
                st.write("•",r)

            st.line_chart(d[["Close","EMA20","EMA50"]].tail(120))

st.info("هذه الأداة تعليمية وتجريبية وليست توصية شراء أو بيع. لا تستخدم تنفيذًا حقيقيًا قبل الاختبار الكافي.")
