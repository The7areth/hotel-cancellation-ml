"""Explicit predictor allowlist. Outcome/status dates are split metadata only."""
import pandas as pd
NUMERIC = ['lead_time','arrival_month','stays_in_weekend_nights','stays_in_week_nights','adults','children','babies','is_repeated_guest','total_nights','total_guests']
CATEGORICAL = ['hotel','meal','market_segment','distribution_channel','reserved_room_type','customer_type']
FEATURES = NUMERIC + CATEGORICAL
FORBIDDEN = {'is_canceled','reservation_status','reservation_status_date','assigned_room_type','booking_changes','deposit_type','adr','days_in_waiting_list','required_car_parking_spaces','total_of_special_requests','country','agent','company','previous_cancellations','previous_bookings_not_canceled'}

def features(frame):
    x=frame.copy()
    if 'arrival_month' not in x:
        x['arrival_month']=pd.to_datetime(x['arrival_date_month'],format='%B').dt.month
    x['total_nights']=x.stays_in_weekend_nights+x.stays_in_week_nights
    x['total_guests']=x.adults+x.children.fillna(0)+x.babies
    return x[FEATURES].copy()

def load_data(path):
    raw=pd.read_csv(path)
    dates=pd.to_datetime(dict(year=raw.arrival_date_year,month=pd.to_datetime(raw.arrival_date_month,format='%B').dt.month,day=raw.arrival_date_day_of_month))
    guests=raw.adults+raw.children.fillna(0)+raw.babies
    valid=(guests>0)&(raw.stays_in_weekend_nights+raw.stays_in_week_nights>0)
    meta={'raw_rows':len(raw),'excluded_zero_guests_or_zero_nights':int((~valid).sum()),'exact_duplicate_rows':int(raw.duplicated().sum()),'duplicates_policy':'Retained: no booking ID proves that identical rows are errors; dependence is a limitation.'}
    return raw.loc[valid].copy(), dates.loc[valid], meta

def split_masks(raw, dates):
    # Use status date only to remove outcomes not available by the next cohort.
    status=pd.to_datetime(raw.reservation_status_date)
    return {
      'train':(dates<'2016-10-01')&(status<'2016-10-01'),
      'validation':(dates>='2016-10-01')&(dates<'2017-01-01')&(status<'2017-01-01'),
      'calibration':(dates>='2017-01-01')&(dates<'2017-03-01')&(status<'2017-03-01'),
      'test':dates>='2017-03-01'}
