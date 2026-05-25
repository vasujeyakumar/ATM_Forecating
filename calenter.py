import pandas as pd
import numpy as np
import holidays

ind_holidays = holidays.India(years=[2024,2025, 2026])

state_holiday_map = {
    "Tamil Nadu": holidays.country_holidays("IN", subdiv="TN"),
    "Kerala": holidays.country_holidays("IN", subdiv="KL"),
    "Gujarat": holidays.country_holidays("IN", subdiv="GJ"),
    "Karnataka": holidays.country_holidays("IN", subdiv="KA"),
    "Delhi": holidays.country_holidays("IN", subdiv="DL"),
    "Andhra Pradesh": holidays.country_holidays("IN", subdiv="AP"),
    "Telangana": holidays.country_holidays("IN", subdiv="TG"),
    "Maharashtra": holidays.country_holidays("IN", subdiv="MH"),
}

national_holidays = holidays.country_holidays("IN")

def get_event(row):
    date = row["date"]
    state = row["State_Name"]

    # national holiday
    if date in national_holidays:
        return 1

    # state holiday
    if state in state_holiday_map and  date in state_holiday_map[state]:
        return 2
    return 0

def is_holiday(df):

    df["Local_festival"] = df.apply(get_event, axis=1)

    df["is_holiday"] = df["date"].apply(lambda x: 1 if x in ind_holidays else 0 )

    return df

def basic_preprocessing(df):

    df = df.copy()

    # convert date
    df["date"] = pd.to_datetime(df["date"])

    # sort
    df = df.sort_values(["ATM ID", "date"])

    # remove duplicates
    df = df.drop_duplicates()

    df = df.reset_index(drop=True)

    return df
def create_calendar_features(df):

    df = df.copy()

    # --------------------------
    # basic calendar features
    # --------------------------
    df["day"] = df["date"].dt.day

    df["month"] = df["date"].dt.month

    df["year"] = df["date"].dt.year

    df["day_of_week"] = df["date"].dt.dayofweek

    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    df["is_weekend"] = (
        df["day_of_week"].isin([5, 6]).astype(int)
    )


    # month cyclic
    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    # weekday cyclic
    df["dow_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["dow_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    return df

def create_lag_features(df):

    df = df.copy()

    df["lag_1"] = (
        df.groupby("ATM ID")["Total Amount Dispensed"]
        .shift(1)
    )

    df["lag_7"] = (
        df.groupby("ATM ID")["Total Amount Dispensed"]
        .shift(7)
    )

    df["lag_30"] = (
        df.groupby("ATM ID")["Total Amount Dispensed"]
        .shift(30)
    )

    df["rolling_mean_7"] = (
        df.groupby("ATM ID")["Total Amount Dispensed"]
        .transform(
            lambda x: x.shift(1).rolling(7).mean()
        )
    )

    df["rolling_mean_30"] = (
        df.groupby("ATM ID")["Total Amount Dispensed"]
        .transform(
            lambda x: x.shift(1).rolling(30).mean()
        )
    )

    df["rolling_std_7"] = (
        df.groupby("ATM ID")["Total Amount Dispensed"]
        .transform(
            lambda x: x.shift(1).rolling(7).std()
        )
    )

    return df


def create_future_calendar_features(df, forecast_days=7):

    df = df.copy()

    for i in range(1, forecast_days + 1):

        future_date = df["date"] + pd.to_timedelta(i, unit="D")

        # future date parts
        df[f"future_day_{i}"] = future_date.dt.day

        df[f"future_month_{i}"] = future_date.dt.month

        df[f"future_year_{i}"] = future_date.dt.year

        df[f"future_day_of_week_{i}"] = future_date.dt.dayofweek

        df[f"future_week_of_year_{i}"] = (
            future_date.dt.isocalendar().week.astype(int)
        )

        # weekend
        df[f"future_is_weekend_{i}"] = (
            future_date.dt.dayofweek.isin([5, 6]).astype(int)
        )

        # holiday
        df[f"future_is_holiday_{i}"] = (
            future_date.apply(
                lambda x: 1 if x in ind_holidays else 0
            )
        )

        # local festival
        temp_df = pd.DataFrame({
            "date": future_date,
            "State_Name": df["State_Name"]
        })

        df[f"future_local_festival_{i}"] = (
            temp_df.apply(get_event, axis=1)
        )

        # cyclic month
        future_month = future_date.dt.month

        df[f"future_month_sin_{i}"] = np.sin(
            2 * np.pi * future_month / 12
        )

        df[f"future_month_cos_{i}"] = np.cos(
            2 * np.pi * future_month / 12
        )

        # cyclic weekday
        future_dow = future_date.dt.dayofweek

        df[f"future_dow_sin_{i}"] = np.sin(
            2 * np.pi * future_dow / 7
        )

        df[f"future_dow_cos_{i}"] = np.cos(
            2 * np.pi * future_dow / 7
        )

    return df

def create_multistep_targets(df, forecast_days=7):

    df = df.copy()

    for i in range(1, forecast_days + 1):

        df[f"target_day_{i}"] = (
            df.groupby("ATM ID")["Total Amount Dispensed"]
            .shift(-i)
        )

    return df

def full_preprocessing_pipeline(df):

    df = basic_preprocessing(df)

    df = is_holiday(df)

    df = create_calendar_features(df)

    df = create_lag_features(df)

    df = create_future_calendar_features(df)

    df = create_multistep_targets(df)

    df = df.dropna()

    df = df.reset_index(drop=True)

    return df

# =========================================================
# CREATE MODEL INPUT
# =========================================================

def create_latest_prediction_input(
    history_df,
    national_holidays,
    state_holiday_map,
    encoders
):

    history_df = history_df.copy()

    # date convert
    history_df["date"] = pd.to_datetime(
        history_df["date"]
    )

    # sort
    history_df = history_df.sort_values(
        ["ATM ID", "date"]
    )

    final_rows = []

    # =====================================================
    # ATM WISE LOOP
    # =====================================================
    for atm_id, group in history_df.groupby("ATM ID"):

        group = group.sort_values("date")

        # latest row
        last_row = group.iloc[-1]

        # latest date
        current_date = last_row["date"]

        # target prediction start date
        next_date = current_date + pd.Timedelta(days=1)

        # history values
        amount_values = (
            group["Total Amount Dispensed"]
            .values
        )

        withdrawal_values = (
            group["No of Withdrawals"]
            .values
        )

        # =================================================
        # CREATE SINGLE INPUT ROW
        # =================================================
        row = {}

        # =================================================
        # STATIC COLUMNS
        # =================================================
        row["ATM ID"] = last_row["ATM ID"]
        row["State_Name"] = last_row["State_Name"]
        row["ATM_City"] = last_row["ATM_City"]
        row["Terminal_Location"] = last_row["Terminal_Location"]

        # =================================================
        # CURRENT DAY FEATURES
        # =================================================
        row["No of Withdrawals"] = last_row["No of Withdrawals"]

        row["Total Amount Dispensed"] = (
            last_row["Total Amount Dispensed"]
        )

        row["Local_festival"] = (
                get_holiday(
                    current_date,
                    last_row["State_Name"]
                )
            )
        
        row["is_holiday"] = int(
                current_date in national_holidays
            )

        row["day"] = current_date.day
        row["month"] = current_date.month
        row["year"] = current_date.year

        row["day_of_week"] = (
            current_date.dayofweek
        )

        row["week_of_year"] = (
            current_date.isocalendar().week
        )

        row["is_weekend"] = int(
            current_date.dayofweek in [5, 6]
        )

        # cyclic
        row["month_sin"] = np.sin(
            2 * np.pi * current_date.month / 12
        )

        row["month_cos"] = np.cos(
            2 * np.pi * current_date.month / 12
        )

        row["dow_sin"] = np.sin(
            2 * np.pi * current_date.dayofweek / 7
        )

        row["dow_cos"] = np.cos(
            2 * np.pi * current_date.dayofweek / 7
        )

        # =================================================
        # LAG FEATURES
        # =================================================

        # lag 1
        row["lag_1"] = amount_values[-1]

        # lag 7
        row["lag_7"] = (
            amount_values[-7]
            if len(amount_values) >= 7
            else np.nan
        )

        # lag 30
        row["lag_30"] = (
            amount_values[-30]
            if len(amount_values) >= 30
            else np.nan
        )

        # rolling mean 7
        row["rolling_mean_7"] = (
            np.mean(amount_values[-7:])
            if len(amount_values) >= 7
            else np.nan
        )

        # rolling mean 30
        row["rolling_mean_30"] = (
            np.mean(amount_values[-30:])
            if len(amount_values) >= 30
            else np.nan
        )

        # rolling std 7
        row["rolling_std_7"] = (
            np.std(amount_values[-7:])
            if len(amount_values) >= 7
            else np.nan
        )

        # =================================================
        # FUTURE 7 DAYS CALENDAR FEATURES
        # =================================================

        for i in range(1, 8):

            future_date = (
                current_date
                + pd.Timedelta(days=i)
            )

            # basic calendar
            row[f"future_day_{i}"] = (
                future_date.day
            )

            row[f"future_month_{i}"] = (
                future_date.month
            )

            row[f"future_year_{i}"] = (
                future_date.year
            )

            row[f"future_day_of_week_{i}"] = (
                future_date.dayofweek
            )

            row[f"future_week_of_year_{i}"] = (
                future_date.isocalendar().week
            )

            row[f"future_is_weekend_{i}"] = int(
                future_date.dayofweek in [5, 6]
            )

            # holiday
            row[f"future_is_holiday_{i}"] = int(
                future_date in national_holidays
            )

            # festival
            row[f"future_local_festival_{i}"] = (
                get_holiday(
                    future_date,
                    last_row["State_Name"]
                )
            )

            # cyclic features
            row[f"future_month_sin_{i}"] = np.sin(
                2 * np.pi * future_date.month / 12
            )

            row[f"future_month_cos_{i}"] = np.cos(
                2 * np.pi * future_date.month / 12
            )

            row[f"future_dow_sin_{i}"] = np.sin(
                2 * np.pi * future_date.dayofweek / 7
            )

            row[f"future_dow_cos_{i}"] = np.cos(
                2 * np.pi * future_date.dayofweek / 7
            )

        # append
        final_rows.append(row)

    # =====================================================
    # FINAL DATAFRAME
    # =====================================================
    prediction_input_df = pd.DataFrame(final_rows)
    
    columns_to_encode = ["ATM ID", "State_Name", "ATM_City", "Terminal_Location"]

    for col in columns_to_encode:
        le = encoders[col]
        prediction_input_df[col] = le.transform(prediction_input_df[col])
        
    return prediction_input_df


def get_holiday(date, state):

    # national holiday
    if date in national_holidays:
        return 1

    # state holiday
    if (
        state in state_holiday_map
        and date in state_holiday_map[state]
    ):
        return 2

    return 0