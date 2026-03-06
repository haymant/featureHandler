# Alpha158/Alpha360 Expressions

The formulae for all Alpha158 and Alpha360 features are defined in the
`get_feature_config` methods of `Alpha158DL` and `Alpha360DL` – in this
workspace they have been copied into the `scripts/Alpha/featureHandler/loader.py`
module (original qlib implementations were identical).


The first table below lists every indicator exposed by those loaders.  A
`category` column assigns a rough IBKR‑style classification to aid
navigation; the following table explains the categories.

## 1. Expression glossary with categories

| indicator name | expression | short description | category |
|---------------|------------|-------------------|----------|
| **[Alpha360]** | | | comprehensive |
| CLOSE59 … CLOSE1 | `Ref($close, n)/$close` | close n days ago, scaled to latest close | comprehensive |
| CLOSE0 | `$close/$close` | current close normalised to 1 | comprehensive |
| OPEN59 … OPEN0 | `Ref($open, n)/$close` or `$open/$close` | open price history | comprehensive |
| HIGH59 … HIGH0 | `Ref($high, n)/$close` or `$high/$close` | high price history | comprehensive |
| LOW59 … LOW0 | `Ref($low, n)/$close` or `$low/$close` | low price history | comprehensive |
| VWAP59 … VWAP0 | `Ref($vwap, n)/$close` or `$vwap/$close` | volume‑weighted avg price | comprehensive |
| VOLUME59 … VOLUME0 | `Ref($volume, n)/($volume+1e-12)` or `$volume/($volume+1e-12)` | raw volume normalised | comprehensive |

| **[Alpha158‑kbar]** | – | price–open relationships | momentum |
| KMID | `($close-$open)/$open` | closing move relative to open | momentum |
| KLEN | `($high-$low)/$open` | range size | volatility |
| KMID2 | `($close-$open)/($high-$low+1e-12)` | normalised body | momentum |
| KUP | `($high-Greater($open,$close))/$open` | upside wick above open/close | momentum |
| KUP2 | `…/($high-$low+1e-12)` | normalized wick | momentum |
| KLOW | `(Less($open,$close)-$low)/$open` | downside wick | momentum |
| KLOW2 | `…/($high-$low+1e-12)` | normalized downside wick | momentum |
| KSFT | `(2*$close-$high-$low)/$open` | candle shift | trend |
| KSFT2 | `…/($high-$low+1e-12)` | normalized shift | trend |

| **[Alpha158‑price]** | | raw price lags | comprehensive |
| OPEN0…OPEN4 | `$open/$close`, `Ref($open,n)/$close` | open price history | comprehensive |
| HIGH0…HIGH4 | `$high/$close`, `Ref($high,n)/$close` | high price history | comprehensive |
| LOW0…LOW4 | `$low/$close`, `Ref($low,n)/$close` | low price history | comprehensive |
| VWAP0…VWAP4 | `$vwap/$close`, `Ref($vwap,n)/$close` | VWAP history | comprehensive |

| **[Alpha158‑volume]** | | raw volume ratios | volume |
| VOLUME0…VOLUME4 | `$volume/(volume+\u03B5)` | and lags | volume |

| **[Alpha158‑rolling]** | | technical operators | (see below) |
| ROC* | `Ref($close,d)/$close` | rate‑of‑change | momentum |
| MA* | `Mean($close,d)/$close` | simple moving average | moving averages |
| STD* | `Std($close,d)/$close` | volatility | volatility |
| BETA* | `Slope($close,d)/$close` | regression slope | regression |
| RSQR* | `Rsquare($close,d)` | R² of fit | regression |
| RESI* | `Resi($close,d)/$close` | linearity residual | regression |
| MAX* | `Max($high,d)/$close` | d‑day high | trend |
| MIN* | `Min($low,d)/$close` | d‑day low | trend |
| QTLU* | `Quantile($close,d,0.8)/$close` | upper quantile | trend |
| QTLD* | `Quantile($close,d,0.2)/$close` | lower quantile | trend |
| RANK* | `Rank($close,d)` | percentile of close | momentum |
| RSV* | `($close-Min($low,d))/(Max($high,d)-Min($low,d)+\u03B5)` | position in band | pivot |
| IMAX* | `IdxMax($high,d)/d` | days since high | trend |
| IMIN* | `IdxMin($low,d)/d` | days since low | trend |
| IMXD* | `(IdxMax($high,d)-IdxMin($low,d))/d` | gap high‑low days | trend |
| CORR* | `Corr($close,Log($volume+1),d)` | price‑volume corr | volume |
| CORD* | `Corr($close/Ref($close,1),Log($volume/Ref($volume,1)+1),d)` | return‑volume corr | volume |
| CNTP* | `Mean($close>Ref($close,1),d)` | pct days up | momentum |
| CNTN* | `Mean($close<Ref($close,1),d)` | pct days down | momentum |
| CNTD* | `CNTP*-CNTN*` | up‑down diff | momentum |
| SUMP* | `Sum(Greater($close-Ref($close,1),0),d)/…` | total gain ratio | momentum |
| SUMN* | total loss ratio | momentum |
| SUMD* | gain‑loss diff | momentum |
| VMA* | `Mean($volume,d)/($volume+\u03B5)` | vol moving average | volume |
| VSTD* | `Std($volume,d)/($volume+\u03B5)` | vol‑std | volatility |
| WVMA* | volume‑weighted volatility | volatility |
| VSUMP*, VSUMN*, VSUMD* | RSI‑style volume measures | volume |

> `*` replaced by window (5,10,20,30,60) or lag; `\u03B5` is a tiny constant.

## 2. Rationale & usage

| feature group | rationale & trading use |
|---------------|-------------------------|
| k‑bar features | Capture candle shape (body, wicks, asymmetry). A large `KUP` may
| | indicate selling pressure; systems often short after a “shooting star”
| | formation. |
| price/volume lags | Raw historical levels useful for mean‑reversion/momentum
| | models. Example: comparing `OPEN2` and `OPEN0` for pullback detection. |
| ROC, MA, STD | Classic indicators measuring momentum, trend and volatility. A
| | momentum strategy might require `ROC5>0` and `MA20>1.01`. |
| RSV/RANK/IMAX/IMIN | Position‑in‑band metrics feed oscillators (e.g. stochastic,
| | Aroon). Traders may fade when `RSV20>0.9` (overbought). |
| volume ratios | Volume confirms price moves; high `VSUMP10` with rising price
| | (`CORD10`) signals accumulation and potential breakout. |
| Alpha360 series | Provides 60‑day historical window to ML models (CNN/RNN/Transformer),
| | capturing patterns such as double bottoms without hand‑crafting rules. |

## 3. Classification definitions

The categories follow a loose IBKR scheme used to tag technical indicators:

| category | definition |
|----------|------------|
| comprehensive | raw price/volume or otherwise all‑purpose features; not tied to
| | a single concept. |
| momentum | measures of change or directionality (ROC, K‑bar body, CNTP,
| | RANK). |
| moving averages | any simple or weighted average (MA*, VMA*). |
| pivot | position within a recent high/low band (RSV, IMAX/IMIN relative to
| | window). |
| regression | statistics derived from linear fits (BETA, RSQR, RESI). |
| trend | extremes and persistence (MAX, MIN, KSFT, IMXD). |
| volatility | measures of dispersion (STD, VSTD, WVMA, KLEN). |
| volume | indicators derived solely from volume or price–volume relations.

These classifications are not used by the loader but may help you segment
features when building portfolios, pruning models, or performing EDA.

