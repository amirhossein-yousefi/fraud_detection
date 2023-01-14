from pydantic import BaseModel


# 2. Class which describes a single fraud measurements
class FraudFeatures(BaseModel):
    '''
    gender: F:Female
            M: Male
    trans_date_trans_time :'2019-01-01 00:00:18'
    city_pop: city population
    merchant:'fraud_Rippin, Kub and Mann'
    category: 'grocery_pos'
    city: 'Malad City'
    state: 'KS'
    job:'Pathologist'
    '''
    trans_date_trans_time: str
    merchant: str
    amt: float
    gender: str
    city: str
    state: str
    city_pop: int
    job: str
    category: str


Labels = ('Not_Fraud', 'Fraud')
