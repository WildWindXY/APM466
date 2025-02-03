import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq

# ------------------- Data Loading & Preprocessing -------------------
# Read bond data and clean columns/formats.
selected_bonds = pd.read_csv("selected_bonds.csv")
selected_bonds.columns = [col.strip() for col in selected_bonds.columns]
selected_bonds["Maturity Date"] = pd.to_datetime(selected_bonds["Maturity Date"])
selected_bonds["Issue Date"] = pd.to_datetime(selected_bonds["Issue Date"])
selected_bonds["Coupon"] = selected_bonds["Coupon"].str.replace('%', '').astype(float)
price_columns = [col for col in selected_bonds.columns if col not in ["Coupon", "ISIN", "Issue Date", "Maturity Date"]]
observation_dates = pd.bdate_range(start="2025-01-06", end="2025-01-17")


# ------------------- Helper Functions -------------------
def get_next_coupon_date(obs_date, mat_date):
    try:
        candidate = pd.Timestamp(year=obs_date.year, month=mat_date.month, day=mat_date.day)
    except ValueError:
        candidate = pd.Timestamp(year=obs_date.year, month=mat_date.month, day=1) + pd.offsets.MonthEnd(1)
    if candidate <= obs_date:
        candidate += pd.DateOffset(months=6)
    return candidate if candidate <= mat_date else mat_date


def get_previous_coupon_date(next_coupon):
    return next_coupon - pd.DateOffset(months=6)


def compute_coupon_periods(next_coupon, mat_date):
    n = 0
    current = next_coupon
    while current <= mat_date:
        n += 1
        current += pd.DateOffset(months=6)
    return n


def bond_price(ytm, coupon, f, n):
    cp = coupon / 2
    price = cp / (1 + ytm / 2) ** f
    for i in range(1, n):
        price += cp / (1 + ytm / 2) ** (f + i)
    price += (cp + 100) / (1 + ytm / 2) ** (f + n)
    return price


def solve_ytm(price, coupon, f, n):
    if n == 1:
        return 2 * (coupon / price + 1) ** (1 / (2 * f)) - 2
    func = lambda y: bond_price(y, coupon, f, n) - price
    try:
        return brentq(func, -0.05, 0.20)
    except ValueError:
        return np.nan


def compute_effective_T(obs_date, mat_date):
    next_coupon = get_next_coupon_date(obs_date, mat_date)
    prev_coupon = get_previous_coupon_date(next_coupon)
    period_length = (next_coupon - prev_coupon).days
    f = (next_coupon - obs_date).days / period_length
    n = compute_coupon_periods(next_coupon, mat_date)
    return f + n, f, n


def get_df(t, boot_list):
    if not boot_list:
        return None
    T_min, DF_min = boot_list[0]
    if t <= T_min:
        return DF_min ** (t / T_min)
    T_max, DF_max = boot_list[-1]
    if t >= T_max:
        return DF_max ** (t / T_max)
    for i in range(len(boot_list) - 1):
        T1, DF1 = boot_list[i]
        T2, DF2 = boot_list[i + 1]
        if T1 <= t <= T2:
            weight = (t - T1) / (T2 - T1)
            return DF1 + weight * (DF2 - DF1)
    return None


# ------------------- Part 1: Yield Calculation & Yield Curve Plot -------------------
# Compute yields for each bond on each observation date.
yield_df = pd.DataFrame(index=selected_bonds["ISIN"],
                        columns=[str(d.date()) for d in observation_dates])
for idx, row in selected_bonds.iterrows():
    isin, coupon, mat_date = row["ISIN"], row["Coupon"], row["Maturity Date"]
    for i, obs_date in enumerate(observation_dates):
        if i >= len(price_columns):
            continue
        price = row[price_columns[i]]
        next_coupon = get_next_coupon_date(obs_date, mat_date)
        f = (next_coupon - obs_date).days / 182.5
        n = compute_coupon_periods(next_coupon, mat_date)
        ytm = solve_ytm(price, coupon, f, n)
        yield_df.at[isin, str(obs_date.date())] = ytm

# Plot 5-year yield curves (YTMs) for each observation date (using linear interpolation).
plt.figure(figsize=(10, 6))
for obs_date in observation_dates:
    ttm_list, ytm_list = [], []
    for idx, row in selected_bonds.iterrows():
        ttm = (row["Maturity Date"] - obs_date).days / 365.0
        ttm_list.append(ttm)
        ytm_list.append(yield_df.at[row["ISIN"], str(obs_date.date())])
    ttm_array = np.array(ttm_list)
    ytm_array = np.array(ytm_list, dtype=float)
    sort_idx = np.argsort(ttm_array)
    ttm_sorted = ttm_array[sort_idx]
    ytm_sorted = ytm_array[sort_idx]

    # Linear interpolation on a denser grid for a smoother line
    if len(ttm_sorted) > 1:
        x_new = np.linspace(ttm_sorted.min(), ttm_sorted.max(), 100)
        y_new = np.interp(x_new, ttm_sorted, ytm_sorted)
        plt.plot(x_new, y_new * 100, label=str(obs_date.date()))
        # Plot original points
        plt.scatter(ttm_sorted, ytm_sorted * 100, marker='o')
    else:
        # If there's only one point, just plot it
        plt.scatter(ttm_sorted, ytm_sorted * 100, marker='o', label=str(obs_date.date()))

plt.xlabel("Remaining Maturity (Years)")
plt.ylabel("YTM (%)")
plt.title("5-Year Yield Curves")
plt.legend(title="Observation Date")
plt.grid(True)
plt.show()

# ------------------- Part 2: Spot Curve Bootstrapping & Plot -------------------
spot_curves = {}
for obs_idx, obs_date in enumerate(observation_dates):
    obs_str = str(obs_date.date())
    # Compute effective maturity T, f, and n for each bond.
    temp = selected_bonds["Maturity Date"].apply(lambda d: compute_effective_T(obs_date, d))
    selected_bonds["T"] = temp.apply(lambda x: x[0])
    selected_bonds["f"] = temp.apply(lambda x: x[1])
    selected_bonds["n"] = temp.apply(lambda x: x[2])

    price_col = price_columns[obs_idx]
    boot_df = selected_bonds[['ISIN', 'Coupon', 'T', 'f', 'n', 'Maturity Date']].copy()
    boot_df["Price"] = selected_bonds[price_col]
    boot_df = boot_df.sort_values("T")

    boot_list = []  # List of tuples (T, DF)
    bootstrapped = {}
    unique_T = sorted(boot_df["T"].unique())
    for T_val in unique_T:
        tol = 1e-6
        group = boot_df[np.abs(boot_df["T"] - T_val) < tol]
        if group.empty:
            group = boot_df[boot_df["T"] == T_val]
        bond = group.sort_values("Coupon").iloc[0]
        coupon, price = bond["Coupon"], bond["Price"]
        f, n = bond["f"], int(bond["n"])

        if n == 1:
            y = 2 * (coupon / price + 1) ** (1 / (2 * f)) - 2
            t_final = f + n
            DF_t_final = 1 / (1 + y / 2) ** t_final
            s_n = y
        else:
            discounted_sum = 0.0
            for i in range(n):
                t_i = f + i
                DF_t = get_df(t_i, boot_list)
                if DF_t is None:
                    DF_t = 1.0
                discounted_sum += (coupon / 2) * DF_t
            t_final = f + n
            DF_t_final = (price - discounted_sum) / (coupon / 2 + 100)
            if DF_t_final <= 0:
                continue
            s_n = 2 * ((1 / DF_t_final) ** (1 / t_final) - 1)

        boot_list.append((t_final, DF_t_final))
        boot_list.sort(key=lambda x: x[0])
        bootstrapped[T_val] = (DF_t_final, s_n)

    maturities = np.array([t * 0.5 for t in sorted(bootstrapped.keys())])
    spot_rates = np.array([bootstrapped[t][1] * 100 for t in sorted(bootstrapped.keys())])
    spot_curves[obs_str] = (maturities, spot_rates)

# Plot 5-year spot curves (using linear interpolation).
plt.figure(figsize=(10, 6))
for obs_date in observation_dates:
    obs_str = str(obs_date.date())
    if obs_str in spot_curves:
        maturities, spot_rates = spot_curves[obs_str]
        sort_idx = np.argsort(maturities)
        mat_sorted = maturities[sort_idx]
        sr_sorted = spot_rates[sort_idx]

        if len(mat_sorted) > 1:
            x_new = np.linspace(mat_sorted.min(), mat_sorted.max(), 100)
            y_new = np.interp(x_new, mat_sorted, sr_sorted)
            plt.plot(x_new, y_new, label=obs_str)
            plt.scatter(mat_sorted, sr_sorted, marker='o')
        else:
            plt.scatter(mat_sorted, sr_sorted, marker='o', label=obs_str)

plt.xlabel("Maturity (Years)")
plt.ylabel("Spot Rate (%)")
plt.title("5-Year Spot Curves")
plt.legend(title="Observation Date")
plt.grid(True)
plt.show()

# ------------------- Part 3: Forward Curve Calculation & Plot -------------------
desired_maturities = np.array([1, 2, 3, 4, 5])
forward_curves = {}
for obs_str, (maturities, spot_rates) in spot_curves.items():
    spot_nominal = np.array(spot_rates) / 100.0
    maturities = np.array(maturities)
    sort_idx = np.argsort(maturities)
    maturities_sorted = maturities[sort_idx]
    spot_sorted = spot_nominal[sort_idx]
    interp_nominal = np.interp(desired_maturities, maturities_sorted, spot_sorted)
    effective_spot = (1 + interp_nominal / 2) ** 2 - 1  # effective annual rate
    s1 = effective_spot[0]
    fwd_rates = [
        ((1 + effective_spot[i]) ** desired_maturities[i] / (1 + s1)) ** (1 / (desired_maturities[i] - 1)) - 1
        for i in range(1, len(desired_maturities))
    ]
    fwd_rates = np.array(fwd_rates) * 100
    forward_periods = desired_maturities[1:] - 1
    forward_curves[obs_str] = (forward_periods, fwd_rates)

plt.figure(figsize=(10, 6))
for obs_str, (forward_periods, fwd_rates) in forward_curves.items():
    plt.plot(forward_periods, fwd_rates, marker='o', label=obs_str)
plt.xlabel("Forward Period Length (Years) [Starting 1 Year from Now]")
plt.ylabel("Forward Rate (%)")
plt.title("1-Year Forward Curves (1yr-1yr to 1yr-4yr)")
plt.legend(title="Observation Date")
plt.grid(True)
plt.show()

# ------------------- Part 4: Covariance & Eigen Decomposition -------------------
# (a) Yield Covariance
desired_maturities_yield = np.array([1, 2, 3, 4, 5])
yield_time_series = {}
for obs_date in observation_dates:
    obs_str = str(obs_date.date())
    ttm = (selected_bonds["Maturity Date"] - obs_date).dt.days / 365.0
    yields = np.array([yield_df.at[isin, obs_str] for isin in selected_bonds["ISIN"]], dtype=float)
    sorted_idx = np.argsort(ttm)
    ttm_sorted = ttm.iloc[sorted_idx].values
    yields_sorted = yields[sorted_idx]
    interp_yields = np.interp(desired_maturities_yield, ttm_sorted, yields_sorted)
    yield_time_series[obs_str] = interp_yields
yield_dates = []
yield_matrix = []
for obs_date in observation_dates:
    obs_str = str(obs_date.date())
    yield_dates.append(obs_str)
    yield_matrix.append(yield_time_series[obs_str])
yield_matrix = np.array(yield_matrix)
log_returns_yields = np.log(yield_matrix[1:] / yield_matrix[:-1])
cov_yields = np.cov(log_returns_yields, rowvar=False)

# (b) Forward Rates Covariance
desired_forward_maturities = np.array([1, 2, 3, 4])
forward_dates = sorted(forward_curves.keys())
forward_matrix = []
for obs_str in forward_dates:
    fr = np.array(forward_curves[obs_str][1], dtype=float) / 100.0
    forward_matrix.append(fr)
forward_matrix = np.array(forward_matrix)
log_returns_forward = np.log(forward_matrix[1:] / forward_matrix[:-1])
cov_forward = np.cov(log_returns_forward, rowvar=False)

# Eigen decomposition
eig_vals_y, eig_vecs_y = np.linalg.eig(cov_yields)
idx_y = np.argsort(eig_vals_y)[::-1]
eig_vals_y = eig_vals_y[idx_y]
eig_vecs_y = eig_vecs_y[:, idx_y]

eig_vals_f, eig_vecs_f = np.linalg.eig(cov_forward)
idx_f = np.argsort(eig_vals_f)[::-1]
eig_vals_f = eig_vals_f[idx_f]
eig_vecs_f = eig_vecs_f[:, idx_f]

print("Yield Covariance Matrix:\n", cov_yields)
print("\nForward Rates Covariance Matrix:\n", cov_forward)
print("\nYield Eigenvalues:\n", eig_vals_y)
print("\nYield Eigenvectors (columns correspond to eigenvalues):\n", eig_vecs_y)
print("\nForward Rates Eigenvalues:\n", eig_vals_f)
print("\nForward Rates Eigenvectors (columns correspond to eigenvalues):\n", eig_vecs_f)
