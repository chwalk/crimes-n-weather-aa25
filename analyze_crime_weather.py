import pandas as pd
import sklearn.linear_model as lm


def main():
    crime_weather_df = pd.read_csv("C:/Users/chwalker/Downloads/merged_file_weather+daily_counts.csv")
    # print("Displaying Raw Crime Weather")
    # print(crime_weather_df)
    # print("Done Displaying Raw Crime Weather")

    crime_weather_df = crime_weather_df.dropna()
    # print("Displaying Cleaned")
    # print(cleaned_crime_weather_df)
    # print("Done displaying cleaned.")
    
    # print(crime_weather_df.columns)
    number_of_crimes_df = crime_weather_df[['NumberofCrimes']]
    temp_df = crime_weather_df[['Temp']]
    
    # cleaned_temp_df = temp_df.dropna()
    # print(cleaned_temp_df)
    # print(temp_df)

    model = lm.LinearRegression()
    model.fit(temp_df, number_of_crimes_df)
    r_squared = model.score(temp_df, number_of_crimes_df)
    print(r_squared)


if __name__ == "__main__":
    main()
