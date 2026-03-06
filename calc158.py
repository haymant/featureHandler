#!/usr/bin/env python
"""
Example script that (re)calculates the Alpha158 feature set and
demonstrates where the resulting .bin files land.

Usage:

    python calc_alpha158.py \
        --start 2008-01-01 --end 2020-12-31 \
        --inst AAPL

After running you can inspect the printed paths or manually list
`$QLIB_DATA/features/...` to see the new bin files.
"""
import argparse

from featureHandler import C, REG_US, Alpha158, Alpha360, D, init

def main(start, end, instruments):
    # initialise qlib (change provider_uri if you use a different region)
    init(provider_uri='/home/zhaoli/.qlib/qlib_data/us_data', region=REG_US)

    print("qlib data root:", C.dpm.get_data_uri("day"), flush=True)

    # instantiate the handler – its ctor/`setup_data` triggers a load
    dh = Alpha158(
        instruments=instruments,
        start_time=start,
        end_time=end,
        freq="day",
    )
    # access the underlying dataframe to force evaluation
    df = dh._data
    print("loaded dataframe shape", df.shape, flush=True)
    # show the first few rows and column names to confirm features
    print("dataframe head:\n", df.head(), flush=True)
    print("columns (showing first 20):", df.columns.tolist()[:20], flush=True)

    dh1 = Alpha360(
        instruments=instruments,
        start_time=start,
        end_time=end,
        freq="day",
        fit_start_time=start,
        fit_end_time=end,        # required by the default ZScoreNorm
    )
    df1 = dh1._data
    print("alpha360 dataframe shape", df1.shape, flush=True)
    print("alpha360 dataframe head:\n", df1.head(), flush=True)
    print("alpha360 columns (first 20):", df1.columns.tolist()[:20], flush=True)
    
    # alternatively you can call D.features directly; the cache is the same
    # fields, names = Alpha158.get_feature_config()  # same config as above
    # _ = D.features(instruments, fields, start, end, freq="day")

    # inspect the feature directory
    # root = Path(C.dpm.get_data_uri("day")) / "features"
    # print(f"looking for .bin files under {root}")
    # n_bins = 0
    # for inst_dir in root.iterdir():
    #     if not inst_dir.is_dir():
    #         continue
    #     for fn in inst_dir.glob("*.bin"):
    #         print("  ", fn.relative_to(root))
    #         n_bins += 1
    # print(f"found {n_bins} feature bin files")

    # # check a concrete example – pick one instrument/field
    # # (modify the instrument name to one that exists in your universe)
    # example_bin = root / "aapl" / "close.day.bin"
    # print("example bin exists?", example_bin.exists())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2008-01-01")
    parser.add_argument("--end", default="2020-12-31")
    parser.add_argument("--inst", nargs="+", default=["AAPL"])
    args = parser.parse_args()
    main(args.start, args.end, args.inst)