import pandas as pd

data = pd.read_csv("bond_data.csv")
selected_isins = [
    "CA135087K528",  # CAN 1.25 Mar 25: moderate coupon.
    "CA135087N340",  # CAN 1.5 Apr 25: slightly longer term.
    "CA135087D507",  # CAN 2.25 Jun 25: more data points for 2025.
    "CA135087K940",  # CAN 0.5 Sep 25: ultra-low coupon.
    "CA135087L518",  # CAN 0.25 Mar 26: very low coupon.
    "CA135087L930",  # CAN 1.0 Sep 26: mid-range maturity.
    "CA135087F825",  # CAN 1.0 Jun 27: representative mid-term.
    "CA135087Q491",  # CAN 3.25 Sep 28: extends the curve.
    "CA135087J397",  # CAN 2.25 Jun 29: approaching 5-year.
    "CA135087N670"   # CAN 2.25 Dec 29: completes the range.
]

selected_bonds = data[data["ISIN"].isin(selected_isins)].copy()
selected_bonds.to_csv("selected_bonds.csv", index=False)
print("Selected bonds have been saved to selected_bonds.csv.")
