import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("seattle-weather.csv")
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# CORRECT FIX: Fit the encoder on the training data BEFORE saving
encoder = LabelEncoder()
df['weather_encoded'] = encoder.fit_transform(df['weather'])

# Choose features and target
x = df[['temp_min', 'wind', 'weather_encoded']]
y = df['temp_max']

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state=42)

# Train model
model = LinearRegression()
model.fit(xtrain, ytrain)

# Save the TRAINED model and the FITTED encoder
joblib.dump(model, "model.sav")
joblib.dump(encoder, "encoder.sav")

print("Training complete. model.sav and encoder.sav have been created.")
print("Weather categories learned:", encoder.classes_)