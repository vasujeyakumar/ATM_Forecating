import pandas as pd
import joblib
import holidays


from pymongo import MongoClient
from datetime import timedelta

from calenter import create_latest_prediction_input


def predict_next_7_days():

    # =====================================================
    # LOAD ENCODERS + MODEL
    # =====================================================

    encoders = joblib.load(
        "artifacts/label_encoders.pkl"
    )

    model = joblib.load(
        "XgbVersion_2_model.pkl"
    )

    # =====================================================
    # MONGODB
    # =====================================================

    client = MongoClient(
        "mongodb://localhost:27017/"
    )

    db = client["ATM_DETAILS"]

    location_collection = db["Atm_locations"]

    transaction_collection = db["AtmRawData"]

    # =====================================================
    # LOCATION DATA
    # =====================================================

    cursor = list(
        location_collection.find(
            {},
            {
                "_id": 0,
                "Address": 0,
                "Country_Code": 0,
                "State_Code": 0
            }
        )
    )

    location_df = pd.DataFrame(cursor)

    # =====================================================
    # LAST 40 DAYS DATA
    # =====================================================

    pipeline = [
        {
            "$group": {
                "_id": "$ATM ID",
                "docs": {
                    "$bottomN": {
                        "n": 40,
                        "sortBy": {"date": 1},
                        "output": "$$ROOT"
                    }
                }
            }
        },
        {
            "$project": {
                "_id": 0,
                "docs._id": 0
            }
        }
    ]

    cursor1 = transaction_collection.aggregate(
        pipeline
    )

    records = []

    for row in cursor1:
        records.extend(row["docs"])

    transaction_df = pd.DataFrame(records)

    # =====================================================
    # MERGE
    # =====================================================

    merged_df = pd.merge(
        location_df,
        transaction_df,
        on="ATM ID",
        how="inner"
    )

    # =====================================================
    # HOLIDAYS
    # =====================================================

    national_holidays = holidays.country_holidays("IN")

    state_holiday_map = {
        "Tamil Nadu": holidays.country_holidays("IN", subdiv="TN"),
        "Kerala": holidays.country_holidays("IN", subdiv="KL"),
        "Gujarat": holidays.country_holidays("IN", subdiv="GJ"),
        "Karnataka": holidays.country_holidays("IN", subdiv="KA"),
        "Delhi": holidays.country_holidays("IN", subdiv="DL"),
        "Andhra Pradesh": holidays.country_holidays("IN", subdiv="AP"),
        "Telangana": holidays.country_holidays("IN", subdiv="TS"),
        "Maharashtra": holidays.country_holidays("IN", subdiv="MH"),
    }

    # =====================================================
    # CREATE MODEL INPUTs
    # =====================================================

    prediction_input_df = create_latest_prediction_input(
        history_df=merged_df,
        national_holidays=national_holidays,
        state_holiday_map=state_holiday_map,
        encoders=encoders
    )

    # =====================================================
    # PREDICT
    # =====================================================

    pred = model.predict(
        prediction_input_df
    )

    # =====================================================
    # FUTURE DATE COLUMN NAMES
    # =====================================================

    today = pd.Timestamp.today().normalize()

    future_dates = [
        (
            today + timedelta(days=i)
        ).strftime("%Y-%m-%d")
        for i in range(1, 8)
    ]

    predicted_date = pd.DataFrame(
        pred,
        columns=future_dates
    )

    # =====================================================
    # FINAL OUTPUT
    # =====================================================

    final_pred_df = pd.concat(
        [
            prediction_input_df[
                [
                    "ATM ID",
                    "State_Name",
                    "ATM_City",
                    "Terminal_Location"
                ]
            ].reset_index(drop=True),

            predicted_date.reset_index(drop=True)
        ],
        axis=1
    )

    # =====================================================
    # DECODE
    # =====================================================

    for col in [
        "ATM ID",
        "State_Name",
        "ATM_City",
        "Terminal_Location"
    ]:

        le = encoders[col]

        final_pred_df[col] = (
            le.inverse_transform(
                final_pred_df[col]
            )
        )

    # =====================================================
    # STATE SUMMARY
    # =====================================================



    return final_pred_df