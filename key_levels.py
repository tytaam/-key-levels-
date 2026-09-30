"""
KEY LEVELS -> DeepCharts
Makes a DeepCharts annotation file (same format as the GEX levels file) with:
  pVAH / pPOC / pVAL   prior-day volume profile (RTH by default)
  MonH / MonL etc.     prior day high / low, named by weekday (RTH by default)
  TueH / TueL etc.     today's RTH high / low (only once RTH has opened)
  ONH / ONL            overnight high / low (18:00 -> 09:30 ET)
  WH / WL              this week's high / low
  PWH / PWL            last week's high / low

Runs automatically on GitHub (see .github/workflows/update.yml).
Local run:  pip install yfinance pandas numpy  ->  python key_levels.py
Output:     docs/LEVELS_keylevels_<SYMBOL>.xml  +  docs/levels.json
"""
import json, sys, time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# ---------------- settings ----------------
PROFILE_SESSION = "RTH"   # "RTH" (09:30-16:00 ET) or "ETH" (full 18:00-17:00 Globex day)
DAY_HL_SESSION  = "RTH"   # which session PDH/PDL use: "RTH" or "ETH"
VALUE_AREA_PCT  = 0.70
TZ = "America/New_York"

SYMBOLS = {  # root: (DeepCharts symbol, Yahoo symbol, tick size)
    "MNQ": ("MNQ-CME", "NQ=F", 0.25),  "NQ": ("NQ-CME", "NQ=F", 0.25),
    "MES": ("MES-CME", "ES=F", 0.25),  "ES": ("ES-CME", "ES=F", 0.25),
    "MYM": ("MYM-CBOT", "YM=F", 1.0),  "YM": ("YM-CBOT", "YM=F", 1.0),
    "M2K": ("M2K-CME", "RTY=F", 0.1),  "RTY": ("RTY-CME", "RTY=F", 0.1),
}

# label: (line color, line width, line style 0=solid 1=dash 2=dot)
STYLES = {
    "pVAH": ("#FFFF9F0A", 2, 1), "pPOC": ("#FFFF9F0A", 3, 0), "pVAL": ("#FFFF9F0A", 2, 1),
    "PDH":  ("#FF64D2FF", 2, 0), "PDL":  ("#FF64D2FF", 2, 0),
    "HOD":  ("#FF30D158", 2, 0), "LOD":  ("#FFFF3B30", 2, 0),
    "ONH":  ("#FFBF5AF2", 2, 1), "ONL":  ("#FFBF5AF2", 2, 1),
    "WH":   ("#FFFFFFFF", 3, 0), "WL":   ("#FFFFFFFF", 3, 0),
    "PWH":  ("#FF98989D", 2, 1), "PWL":  ("#FF98989D", 2, 1),
}
# ------------------------------------------


def fetch(yf_symbol, interval, period):
    import yfinance as yf
    df = pd.DataFrame()
    for attempt in range(4):
        try:
            df = yf.download(yf_symbol, interval=interval, period=period,
                             prepost=True, progress=False, auto_adjust=False)
        except Exception as e:
            print(f"Yahoo error ({e}), retrying...")
        if not df.empty:
            break
        time.sleep(20 * (attempt + 1))
    if df.empty:
        raise SystemExit(f"No {interval} data returned for {yf_symbol}.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    df.index = (df.index.tz_localize("UTC") if df.index.tz is None else df.index).tz_convert(TZ)
    return df[["High", "Low", "Volume"]].dropna()


def add_session_cols(df):
    # Globex day starts 18:00 ET, so shift +6h to get the trade date
    df = df.copy()
    df["tdate"] = (df.index + pd.Timedelta(hours=6)).date
    mins = df.index.hour * 60 + df.index.minute
    df["rth"] = (mins >= 570) & (mins < 960)           # 09:30-16:00
    df["on"] = (mins >= 1080) | (mins < 570)           # 18:00-09:30
    return df


def rnd(p, tick):
    return round(round(p / tick) * tick, 6)


def volume_profile(bars, tick, va_pct):
    lo = rnd(bars["Low"].min(), tick)
    hi = rnd(bars["High"].max(), tick)
    n = int(round((hi - lo) / tick)) + 1
    vol = np.zeros(n)
    for h, l, v in zip(bars["High"].values, bars["Low"].values, bars["Volume"].values):
        a = int(round((rnd(l, tick) - lo) / tick))
        b = int(round((rnd(h, tick) - lo) / tick))
        vol[a:b + 1] += v / (b - a + 1)          # spread bar volume evenly across its range
    poc = int(np.argmax(vol))
    target, total = vol.sum() * va_pct, vol[poc]
    up = dn = poc
    while total < target and (up < n - 1 or dn > 0):
        above = vol[up + 1] if up < n - 1 else -1
        below = vol[dn - 1] if dn > 0 else -1
        if above >= below:
            up += 1; total += above
        else:
            dn -= 1; total += below
    price = lambda i: rnd(lo + i * tick, tick)
    return price(up), price(poc), price(dn)


_CACHE = {}


def get_data(yf_symbol):
    if yf_symbol not in _CACHE:
        _CACHE[yf_symbol] = (add_session_cols(fetch(yf_symbol, "1m", "7d")),
                             add_session_cols(fetch(yf_symbol, "1h", "60d")))
    return _CACHE[yf_symbol]


def compute(root):
    ds_symbol, yf_symbol, tick = SYMBOLS[root]
    m1, h1 = get_data(yf_symbol)

    dates = sorted(m1["tdate"].unique())
    today = dates[-1]
    prev = next((d for d in reversed(dates[:-1]) if m1[(m1.tdate == d) & m1.rth].shape[0]), None)
    if prev is None:
        raise SystemExit("Couldn't find a prior session in the data.")

    lv = {}
    pday = m1[m1.tdate == prev]
    prof = pday[pday.rth] if PROFILE_SESSION == "RTH" else pday
    lv["pVAH"], lv["pPOC"], lv["pVAL"] = volume_profile(prof, tick, VALUE_AREA_PCT)
    hl = pday[pday.rth] if DAY_HL_SESSION == "RTH" else pday
    lv["PDH"], lv["PDL"] = hl["High"].max(), hl["Low"].min()

    tday = m1[m1.tdate == today]
    on = tday[tday.on]
    if len(on) >= 360:
        lv["ONH"], lv["ONL"] = on["High"].max(), on["Low"].min()
    rth = tday[tday.rth]
    if len(rth):
        lv["HOD"], lv["LOD"] = rth["High"].max(), rth["Low"].min()

    # weeks: Sunday-evening bars carry Monday's trade date, so ISO week works
    wk = pd.to_datetime(h1["tdate"]).dt.isocalendar()
    h1 = h1.assign(week=(wk.year * 100 + wk.week).values)
    this_wk = h1["week"].iloc[-1]
    cur = h1[h1.week == this_wk]
    lv["WH"], lv["WL"] = cur["High"].max(), cur["Low"].min()
    older = h1[h1.week < this_wk]
    if len(older):
        last = older[older.week == older["week"].max()]
        lv["PWH"], lv["PWL"] = last["High"].max(), last["Low"].min()

    # day-named labels: yesterday Mon -> MonH/MonL, today Tue -> TueH/TueL
    names = {k: k for k in lv}
    pd_ = prev.strftime("%a")
    td_ = today.strftime("%a")
    names.update({"PDH": f"{pd_}H", "PDL": f"{pd_}L", "HOD": f"{td_}H", "LOD": f"{td_}L"})
    return ds_symbol, tick, {k: rnd(float(v), tick) for k, v in lv.items()}, names


def build_annotations(ds_symbol, levels, names=None):
    names = names or {}
    # merge levels sitting on the same price into one line ("PDH + WH")
    by_price = {}
    for name, p in levels.items():
        by_price.setdefault(p, []).append(name)
    now = int(time.time())
    out = []
    for i, (p, keys) in enumerate(sorted(by_price.items()), 1):
        color, width, style = max((STYLES[n] for n in keys), key=lambda s: s[1])
        out.append({
            "AnnType": 19, "alert": None, "SymbName": ds_symbol,
            "Name": f"UserAnnKL{i}", "FontSize": 11, "CAIndex": 0,
            "X1SizeUnit": 0, "X1SizeValue": now, "DTX1": "0001-01-01T00:00:00",
            "X2SizeUnit": 0, "X2SizeValue": now + 26400, "DTX2": "0001-01-01T00:00:00",
            "X3SizeUnit": 0, "X3SizeValue": now, "DTX3": "0001-01-01T00:00:00",
            "Y1": p, "Y2": p, "Y3": p, "FreehandPoints": None,
            "SlopeY2": 0, "SlopeY3": 0, "SourceTfInMs": 300000,
            "IsExtended": True, "IsLocked": False,
            "Ann1": {"Hidden": False,
                     "Style": {"Color": color, "LineWidth": width, "LStyle": style},
                     "Background": {"Color": "#FF58550B", "Opacity": 20}},
            "Ann2": None, "TextMsg": " + ".join(names.get(n, n) for n in keys), "TextPadding": 0,
            "TextColor": "#FFFFFFFF", "ShowDifference": False, "ShowPercent": False,
            "ShowPrice": True, "ShowLblBackground": False, "IsExchDT": True,
            "LabelAlign": 0, "AnnParam": None, "Money": None,
        })
    return out


RUN_SYMBOLS = ["MNQ", "NQ", "MES", "ES"]


def main():
    roots = [s.upper() for s in sys.argv[1:]] or RUN_SYMBOLS
    out_dir = Path(__file__).resolve().parent / "docs"
    out_dir.mkdir(exist_ok=True)
    now_pt = datetime.now(ZoneInfo("America/Los_Angeles"))
    summary = {"updated": now_pt.strftime("%a %b %d, %Y %I:%M %p PT"),
               "date": now_pt.strftime("%Y-%m-%d"), "symbols": {}}
    for root in roots:
        ds_symbol, tick, levels, names = compute(root)
        anns = build_annotations(ds_symbol, levels, names)
        (out_dir / f"LEVELS_keylevels_{root}.xml").write_text(json.dumps(anns, indent=2))
        summary["symbols"][root] = [{"label": a["TextMsg"], "price": a["Y1"]} for a in reversed(anns)]
        print(f"--- {root} ---")
        for a in reversed(anns):
            print(f"{a['Y1']:>12}  {a['TextMsg']}")
    (out_dir / "levels.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
