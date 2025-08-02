# EconomicCalendarIntegrator.py
import pandas as pd
import requests

class EconomicCalendarIntegrator:
    def __init__(self, api_url):
        self.api_url = api_url

    def fetch_events(self):
        response = requests.get(self.api_url)
        events = response.json()
        return pd.DataFrame(events)

    def parse_events(self, events_df):
        events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])
        return events_df

if __name__ == "__main__":
    integrator = EconomicCalendarIntegrator("https://api.example.com/economic_calendar")
    events = integrator.fetch_events()
    parsed_events = integrator.parse_events(events)
    print(parsed_events.head())
