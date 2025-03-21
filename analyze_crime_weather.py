import pandas as pd
import sklearn.linear_model as lm


YEAR = "ALL"


def main(year):
    crime_weather_df = pd.read_csv("C:/Users/walkadmin/Downloads/merged_file_weather+daily_counts.csv", parse_dates=['Date'])

    if year != "ALL":
        if type(year) == str:
            year = int(year)
        crime_weather_df = crime_weather_df[crime_weather_df.Date.dt.year == year]

    crime_weather_df = crime_weather_df.dropna()
    
    number_of_crimes_df = crime_weather_df[['NumberofCrimes']]
    temp_df = crime_weather_df[['Temp']]

    model = lm.LinearRegression()
    model.fit(temp_df, number_of_crimes_df)
    r_squared = model.score(temp_df, number_of_crimes_df)
    print(r_squared)


if __name__ == "__main__":
    main(YEAR)
