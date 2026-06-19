# ==================================================
# IMPORTS
# ==================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import holidays
import tensorflow as tf

from sklearn.base import clone
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from xgboost import XGBRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
    Conv1D,
    Flatten,
    BatchNormalization,
    TimeDistributed
)
from tensorflow.keras.regularizers import l2


# ==================================================
# 1) LOAD AND SORT DATA
# ==================================================
df = P1_master_df.copy()

df['timestamp'] = pd.to_datetime(df['timestamp'])

df = (
    df.sort_values('timestamp')
      .reset_index(drop=True)
)


# ==================================================
# 2) FEATURE ENGINEERING
# ==================================================

# Electricity usage lags
df['lag_1'] = df['USAGE_kWh'].shift(1)
df['lag_3'] = df['USAGE_kWh'].shift(3)
df['lag_6'] = df['USAGE_kWh'].shift(6)
df['lag_12'] = df['USAGE_kWh'].shift(12)
df['lag_24'] = df['USAGE_kWh'].shift(24)
df['lag_72'] = df['USAGE_kWh'].shift(72)
df['lag_168'] = df['USAGE_kWh'].shift(168)
df['lag_720'] = df['USAGE_kWh'].shift(720)

# Weather lags
df['temp_lag_1'] = df['temp'].shift(1)
df['temp_lag_24'] = df['temp'].shift(24)
df['temp_lag_168'] = df['temp'].shift(168)

# Rolling electricity features
df['rolling_mean_24h'] = (
    df['USAGE_kWh']
    .shift(1)
    .rolling(24)
    .mean()
)

df['rolling_std_24h'] = (
    df['USAGE_kWh']
    .shift(1)
    .rolling(24)
    .std()
)

df['rolling_mean_7d'] = (
    df['USAGE_kWh']
    .shift(1)
    .rolling(24 * 7)
    .mean()
)

df['rolling_std_7d'] = (
    df['USAGE_kWh']
    .shift(1)
    .rolling(24 * 7)
    .std()
)

df['rolling_mean_30d'] = (
    df['USAGE_kWh']
    .shift(1)
    .rolling(24 * 30)
    .mean()
)

df['rolling_std_30d'] = (
    df['USAGE_kWh']
    .shift(1)
    .rolling(24 * 30)
    .std()
)

# Fourier features for seasonality
day = 24
week = 24 * 7
month = 24 * 30

df['sin_day'] = np.sin(
    2 * np.pi * df.index / day
)

df['cos_day'] = np.cos(
    2 * np.pi * df.index / day
)

df['sin_week'] = np.sin(
    2 * np.pi * df.index / week
)

df['cos_week'] = np.cos(
    2 * np.pi * df.index / week
)

df['sin_month'] = np.sin(
    2 * np.pi * df.index / month
)

df['cos_month'] = np.cos(
    2 * np.pi * df.index / month
)

# Time features
df['weekday'] = df['timestamp'].dt.dayofweek
df['hour_of_day'] = df['timestamp'].dt.hour
df['month_number'] = df['timestamp'].dt.month
df['day_of_month'] = df['timestamp'].dt.day

df['is_weekend'] = (
    df['weekday']
    .isin([5, 6])
    .astype(int)
)

df['is_morning'] = (
    df['hour_of_day']
    .between(5, 11)
    .astype(int)
)

df['is_evening'] = (
    df['hour_of_day']
    .between(17, 22)
    .astype(int)
)

# U.S. holidays
us_holidays = holidays.CountryHoliday('US')

df['is_holiday'] = df['timestamp'].apply(
    lambda timestamp:
    int(timestamp.date() in us_holidays)
)

# Weather interaction
df['temp_rhum_interaction'] = (
    df['temp'] * df['rhum']
)

# Optional formatting column
df['year_month'] = (
    df['timestamp']
    .dt.to_period('M')
)


# ==================================================
# 3) PARAMETERS
# ==================================================
HORIZON = 720
STEP = 1

EPOCHS_LSTM = 10
BATCH_SIZE = 32


# ==================================================
# 4) TARGET: 720-HOUR / MONTH-AHEAD VALUE
# ==================================================

# Because every row represents one hour,
# shift(-720) means the value after 720 hours.
df['y720'] = (
    df['USAGE_kWh']
    .shift(-HORIZON)
)


# ==================================================
# 5) FEATURE SET
# ==================================================
FEATURES = [

    # Behavioral
    # 'CaloricBurn',
    # 'Steps',
    # 'total_time_asleep',

    # Environmental
    # 'temp',
    # 'dwpt',
    # 'rhum',
    # 'wdir',
    # 'wspd',
    # 'pres',
    # 'temp_rhum_interaction',
    # 'temp_lag_1',
    # 'temp_lag_24',
    # 'temp_lag_168',

    # Load profile / calendar
    'is_holiday',
    'weekday',
    'is_weekend',
    'is_morning',
    'is_evening',
    'month_number',
    'day_of_month',
    'hour_of_day',

    # Lagged load
    'lag_1',
    'lag_3',
    'lag_6',
    'lag_12',
    'lag_24',
    'lag_72',
    'lag_168',
    'lag_720',

    # Rolling statistics
    'rolling_mean_24h',
    'rolling_std_24h',
    'rolling_mean_7d',
    'rolling_std_7d',
    'rolling_mean_30d',
    'rolling_std_30d',

    # Seasonality encodings
    'sin_day',
    'cos_day',
    'sin_week',
    'cos_week',
    'sin_month',
    'cos_month'
]


# ==================================================
# 6) REMOVE INVALID ROWS
# ==================================================

# Remove rows missing model inputs or the 720-hour target.
df = (
    df.dropna(
        subset=FEATURES + ['y720']
    )
    .reset_index(drop=True)
)


# ==================================================
# 7) CUSTOM METRICS
# ==================================================
def mape_corr(y_true, y_pred):

    y_true = np.asarray(
        y_true,
        dtype=float
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float
    )

    return (
        np.mean(
            np.abs(
                1 - (
                    (y_pred + 1) /
                    (y_true + 1)
                )
            )
        )
        * 100
    )


def r2_goodness(y_true, y_pred):

    y_true = np.asarray(
        y_true,
        dtype=float
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float
    )

    y_bar = np.mean(y_true)

    ss_reg = np.sum(
        (y_pred - y_bar) ** 2
    )

    ss_tot = np.sum(
        (y_true - y_bar) ** 2
    )

    return ss_reg / ss_tot


# ==================================================
# 8) MODEL DEFINITIONS
# ==================================================
sklearn_models = {

    'XGBoost': XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    ),

    'Ridge': Ridge(
        alpha=1.0
    )
}


def build_lstm(input_dim):

    model = Sequential([

        Input(
            shape=(1, input_dim)
        ),

        LSTM(
            100,
            return_sequences=True
        ),

        Dropout(0.3),

        LSTM(50),

        Dropout(0.3),

        Dense(1)
    ])

    model.compile(
        optimizer='adam',
        loss='mse'
    )

    return model


def build_cnn_lstm(timesteps, input_dim):

    model = Sequential([

        Input(
            shape=(
                timesteps,
                input_dim,
                1
            )
        ),

        TimeDistributed(
            Conv1D(
                filters=64,
                kernel_size=3,
                activation='relu',
                padding='same',
                kernel_regularizer=l2(0.01)
            )
        ),

        TimeDistributed(
            BatchNormalization()
        ),

        TimeDistributed(
            Flatten()
        ),

        Dropout(0.3),

        LSTM(
            64,
            return_sequences=False
        ),

        Dropout(0.3),

        Dense(1)
    ])

    model.compile(
        optimizer='adam',
        loss='mse'
    )

    return model


# ==================================================
# 9) EXPANDING-WINDOW WALK-FORWARD
# ==================================================
def walk_forward_light(
    model_name,
    step=STEP
):

    results = []

    # User-requested first origin:
    # For HORIZON = 720:
    # max(720 * 2, 1000) = 1440
    first_origin = max(
        HORIZON * 2,
        1000
    )

    for t in range(
        first_origin,
        len(df),
        step
    ):

        # ------------------------------------------
        # Current forecast origin
        # ------------------------------------------
        test = df.iloc[[t]].copy()

        origin_time = (
            test['timestamp'].iloc[0]
        )

        y_true = (
            test['y720'].iloc[0]
        )

        # ------------------------------------------
        # Leakage-safe training boundary
        # ------------------------------------------
        #
        # A training row j has target at j + 720.
        #
        # At forecast origin t, require:
        #
        # j + 720 <= t
        #
        # Therefore:
        #
        # j <= t - 720
        #
        # Since iloc excludes the end index,
        # we use +1.
        train_end = (
            t
            - HORIZON
            + 1
        )

        train = (
            df.iloc[:train_end]
            .copy()
        )

        # ------------------------------------------
        # Train and test arrays
        # ------------------------------------------
        X_train = (
            train[FEATURES]
            .values
        )

        y_train = (
            train['y720']
            .values
        )

        X_test = (
            test[FEATURES]
            .values
        )

        # ------------------------------------------
        # Fit scaler only on the current training set
        # ------------------------------------------
        scaler = RobustScaler()

        X_train_scaled = (
            scaler.fit_transform(X_train)
        )

        X_test_scaled = (
            scaler.transform(X_test)
        )

        # ------------------------------------------
        # Retrain the model at every origin
        # ------------------------------------------
        if model_name in sklearn_models:

            model = clone(
                sklearn_models[model_name]
            )

            model.fit(
                X_train_scaled,
                y_train
            )

            y_pred = model.predict(
                X_test_scaled
            )[0]

        elif model_name == 'LSTM':

            tf.keras.backend.clear_session()

            model = build_lstm(
                input_dim=X_train_scaled.shape[1]
            )

            X_train_lstm = (
                X_train_scaled.reshape(
                    -1,
                    1,
                    X_train_scaled.shape[1]
                )
            )

            X_test_lstm = (
                X_test_scaled.reshape(
                    1,
                    1,
                    X_test_scaled.shape[1]
                )
            )

            model.fit(
                X_train_lstm,
                y_train,
                epochs=EPOCHS_LSTM,
                batch_size=BATCH_SIZE,
                verbose=0,
                shuffle=False
            )

            y_pred = model.predict(
                X_test_lstm,
                verbose=0
            )[0, 0]

            tf.keras.backend.clear_session()

        elif model_name == 'CNN-LSTM':

            tf.keras.backend.clear_session()

            model = build_cnn_lstm(
                timesteps=1,
                input_dim=X_train_scaled.shape[1]
            )

            X_train_cnn = (
                X_train_scaled.reshape(
                    -1,
                    1,
                    X_train_scaled.shape[1],
                    1
                )
            )

            X_test_cnn = (
                X_test_scaled.reshape(
                    1,
                    1,
                    X_test_scaled.shape[1],
                    1
                )
            )

            model.fit(
                X_train_cnn,
                y_train,
                epochs=EPOCHS_LSTM,
                batch_size=BATCH_SIZE,
                verbose=0,
                shuffle=False
            )

            y_pred = model.predict(
                X_test_cnn,
                verbose=0
            )[0, 0]

            tf.keras.backend.clear_session()

        else:

            raise ValueError(
                f'Unknown model: {model_name}'
            )

        # ------------------------------------------
        # Store one 720-hour-ahead forecast
        # ------------------------------------------
        results.append({

            # Time when the forecast is made
            'origin_time': origin_time,

            # Time being predicted
            'time': (
                origin_time
                + pd.Timedelta(
                    hours=HORIZON
                )
            ),

            'actual': y_true,

            'pred': float(y_pred),

            'training_samples': len(train)
        })

    return pd.DataFrame(results)


# ==================================================
# 10) RUN ALL MODELS
# ==================================================
all_models = [
    'XGBoost',
    'Ridge',
    'LSTM',
    'CNN-LSTM'
]

dfs = []
summaries_P1 = []

for model_name in all_models:

    print(
        f'\nRunning {model_name}...'
    )

    df_res = walk_forward_light(
        model_name=model_name,
        step=STEP
    )

    mape = mape_corr(
        df_res['actual'].values,
        df_res['pred'].values
    )

    print(
        f"{model_name:10}  "
        f"MAPE={mape:.2f}%"
    )

    df_res = df_res.rename(
        columns={
            'pred': model_name
        }
    )

    dfs.append(
        df_res[
            [
                'time',
                model_name,
                'actual'
            ]
        ]
    )

    summaries_P1.append({

        'model': model_name,

        'mape': mape
    })


# ==================================================
# 11) SUMMARY TABLE
# ==================================================
summary_P1_df = (
    pd.DataFrame(summaries_P1)
    .set_index('model')
    .round(3)
)

print(
    "\nSummary P1 df month-ahead:\n",
    summary_P1_df
)


# ==================================================
# 12) MERGE MODEL FORECASTS
# ==================================================
df_merged_P1 = dfs[0].copy()

for df_i in dfs[1:]:

    df_merged_P1 = (
        df_merged_P1.merge(
            df_i,
            on=['time', 'actual'],
            how='inner'
        )
    )

df_merged_P1 = (
    df_merged_P1
    .set_index('time')
    .sort_index()
)


# ==================================================
# 13) SELECT LAST 30 DAYS
# ==================================================
last_date = (
    df_merged_P1.index.max()
)

start_date = (
    last_date
    - pd.Timedelta(days=30)
)

df_last_month = (
    df_merged_P1.loc[
        start_date:last_date
    ]
)


# ==================================================
# 14) PLOT LAST 30 DAYS
# ==================================================
plt.figure(
    figsize=(12, 6)
)

df_last_month['actual'].plot(
    label='Actual',
    alpha=0.8
)

for model_name in all_models:

    df_last_month[model_name].plot(
        label=model_name,
        alpha=0.7
    )

plt.xlabel('Time')
plt.ylabel('USAGE_kWh')

plt.title(
    '720 h-Ahead Forecast: '
    'Actual vs. All Models'
)

plt.legend()
plt.tight_layout()
plt.show()