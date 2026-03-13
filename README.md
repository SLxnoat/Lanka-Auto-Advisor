# Lanka Auto Advisor

**A machine learning system for used vehicle price prediction in Sri Lanka, built with economic context awareness.**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-189A18?style=flat-square)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![R2](https://img.shields.io/badge/R2%20Score-0.9422-brightgreen?style=flat-square)](/)
[![MAE](https://img.shields.io/badge/MAE-Rs.349%2C447-blue?style=flat-square)](/)

---

## Why I Built This

Anyone who has tried buying a used car in Sri Lanka knows the problem. You find a listing on ikman, the seller quotes Rs. 9.2M, and you have no way to know if that is reasonable or not. Most people either trust the seller or ask a friend who may know no more than they do.

What makes Sri Lanka's vehicle market harder than most is that prices are not just driven by the car itself. The 2022 economic crisis doubled some car prices almost overnight. The USD/LKR rate, inflation, and import policies all affect what a car is worth at any given time. A valuation tool that ignores this context will consistently give wrong answers.

This project is an attempt to build a pricing model that accounts for those economic factors alongside the standard vehicle specifications.

---

## How It Works

The system takes basic vehicle details from the user and returns a predicted fair market price along with a deal assessment. Behind the scenes, the model was trained on data that includes the economic context of each listing, specifically the USD/LKR rate and the inflation rate at the time the vehicle was listed.

The model itself is an XGBoost regression trained on approximately 4,000 records calibrated against real Sri Lankan market listings from 2019 to 2026.

---

## Model Performance

Training covered the full economic cycle, from the pre-crisis stability of 2019 through the 2022 hyperinflation peak and into the 2024 to 2026 recovery period.

```
Algorithm       XGBoost Regressor
R2 Score        0.9422
MAE             Rs. 349,447
Training set    3,200 records  (80 percent)
Test set          800 records  (20 percent)
Total features  15
```

An R2 of 0.9422 means the model accounts for roughly 94 percent of the price variance in the test data. The MAE of Rs. 349,447 represents the average prediction error, which is acceptable for a market where listing prices themselves can vary by 5 to 10 percent depending on the seller.

---

## What the Analysis Showed

**Year is the strongest predictor**, with a correlation of 0.64 to price. That is expected given depreciation.

**Mileage has a strong inverse relationship**, at negative 0.40. Every 10,000 km above 80,000 km pushes the price down in a measurable way.

**The USD rate shows a mild linear correlation of 0.11**, but this understates its real effect. The scatter plot below shows three visually distinct clusters corresponding to different economic regimes: pre-crisis stability at USD 175 to 210, the crisis peak at USD 320 to 370, and the stabilisation period at USD 290 to 325. XGBoost handles this non-linearity much better than a linear model would.

**Auction grade has an outsized effect**. For vehicles registered between 2014 and 2016, the average price difference between Grade 5.0 and Grade 4.0 is approximately 56 percent. Most buyers comparing listings do not account for this.

### Correlation Between Vehicle Price and Economic Indicators

![Correlation between Vehicle Price and Economic Indicators](img/Correlation%20between%20Vehicle%20Price%20and%20Economic%20Indicators.png)

The heatmap confirms that year (0.64) and mileage (-0.40) are the dominant numerical predictors. The USD/LKR rate and inflation rate show weaker linear correlations, but their true influence is non-linear and concentrated around crisis periods, which is why tree-based models capture them better than regression.

### How USD Rate Impacts Vehicle Prices

![Price vs USD Rate](img/Price%20vs%20USD%20Rate.png)

The regression line suggests a modest linear relationship, but the scatter tells a more important story. The data clusters into three distinct bands that correspond directly to Sri Lanka's economic phases. During the 2022 crisis, the same vehicle could command almost double its pre-crisis price, an effect no linear model can accurately represent.

### Average Vehicle Price by Condition

![Average Price by Condition](img/Average%20Price%20by%20Condition.png)

Condition has a significant impact on price. Vehicles in excellent condition command a premium of roughly Rs. 1.0M to 1.5M over fair condition units of the same model and year. During the 2022 inflation peak, this gap widened further as buyers treated well-maintained vehicles as stores of value.

---

## Project Structure

```
Lanka-Auto-Advisor/
|
+-- data/
|   +-- vehicle_data.csv              # 4,000 market-calibrated records
|
+-- img/
|   +-- Correlation between Vehicle Price and Economic Indicators.png
|   +-- Price vs USD Rate.png
|   +-- Average Price by Condition.png
|
+-- notebooks/
|   +-- vehicle_market_analysis.ipynb # EDA, correlation analysis, visualisations
|
+-- src/
|   +-- __init__.py
|   +-- data_loader.py                # CSV loading, exchange rate fetch
|   +-- model.py                      # XGBoost training and inference
|   +-- advisor.py                    # Deal scoring logic
|   +-- preprocessing/
|   |   +-- preprocessor.py           # Label encoding, date feature extraction
|   +-- models/
|   |   +-- vehicle_price_model.pkl   # Serialised trained model
|   +-- utils/
|       +-- encoders.pkl              # Saved label encoders
|
+-- app.py                            # Streamlit web interface
+-- test_pipeline.py                  # End-to-end pipeline test
+-- setup_project.py                  # Creates folder structure
+-- requirements.txt
+-- README.md
```

---

## Getting Started

**Requirements:** Python 3.9 or newer.

**Installation:**

```bash
git clone https://github.com/SLxnoat/Lanka-Auto-Advisor.git
cd Lanka-Auto-Advisor
pip install -r requirements.txt
```

**Create the project folders** (if running from scratch):

```bash
python setup_project.py
```

**Train the model:**

```bash
python test_pipeline.py
```

Expected output:

```
Loading vehicle data from data/vehicle_data.csv...
Successfully loaded 4000 records.
Starting preprocessing...
Training data shape: (3200, 15), Testing data shape: (800, 15)
Model training completed.
MAE: Rs. 349,447.00
R-Squared Score: 0.9422
Model saved to src/models/vehicle_price_model.pkl.
```

**Run the application:**

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Dependencies

```
pandas
numpy
scikit-learn
xgboost
streamlit
matplotlib
seaborn
requests
```

---

## Known Limitations

These are honest observations from building and testing the system. Acknowledging them is more useful than pretending they do not exist.

**The Streamlit app uses placeholder encoding.** The current `app.py` passes hardcoded integer values for most categorical features rather than running them through the saved label encoders. This means the app's predictions may not fully reflect the model's learned relationships for brand, model name, fuel type, and similar fields. Wiring the saved encoders from `src/utils/encoders.pkl` into the app is the most important pending improvement.

**The dataset is synthetic.** The 4,000 training records were generated by calibrating against real listing prices from riyasewana, ikman, and pricelanka rather than by scraping those sites directly. The economic data (USD/LKR and inflation figures) is drawn from verified historical sources. The vehicle prices and feature distributions reflect the real market but are not raw scraped listings. A live scraping pipeline would improve this.

**The exchange rate is hardcoded.** The `DataLoader` class currently returns a static value of 312.50 rather than fetching from a live API. For a demo or portfolio project this is acceptable, but a production system would need a reliable exchange rate feed.

**The dataset covers a specific period.** Training data runs from 2019 to early 2026. Significant shifts in import policy, fuel prices, or economic conditions after that period will reduce accuracy until the model is retrained on newer data.

**Model coverage is uneven across vehicle types.** Rare models with few training examples will produce less reliable predictions than common models like the Toyota Aqua or Suzuki Wagon R that appear frequently in the data. The Corolla 121 example highlighted in testing showed this clearly: the model had no specific training examples for that variant and underestimated its price significantly.

**No confidence intervals in the current version.** The app reports a single predicted price. A more informative output would include a prediction range reflecting model uncertainty, particularly for less common vehicles.

---

## Future Development

- [ ] Connect the Streamlit app to the saved label encoders for proper categorical encoding
- [ ] Add prediction confidence intervals using XGBoost quantile regression
- [ ] Integrate a live exchange rate API
- [ ] Build a scraping pipeline for ikman and riyasewana to enable continuous retraining
- [ ] Expand vehicle coverage, particularly for older Japanese models and local assembled brands
- [ ] Add a "best time to buy" feature based on seasonal price trends

---

## Data Sources

Vehicle price benchmarks cross-referenced from **riyasewana.com**, **ikman.lk**, **pricelanka.lk**, and **careka.lk**.

Economic indicators sourced from **Central Bank of Sri Lanka (CBSL)**, **macrotrends.net**, and **exchange-rates.org**.

---

Built by **[Mayura Bandara](https://www.linkedin.com/in/mayura-bandara)**

*This started as a frustration with not being able to tell a fair price from an inflated one.*
*Hopefully it helps someone else make a more informed decision.*
