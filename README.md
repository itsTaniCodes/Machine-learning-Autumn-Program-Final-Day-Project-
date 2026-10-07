# Machine-learning-Autumn-Program-Final-Day-Project-
As part of the programme, I applied the concepts learned throughout the sessions to develop a Machine Learning project, following the complete workflow from preparing the data to training, evaluating, and presenting the final model.
# 🚗 ML Auction Bid Recommendation System

## Machine Learning Project — MIPT International Autumn School 2026

This project develops a **Machine Learning–based auction bidding system** for an automotive buyer named **Viktor**, who purchases approximately 40 cars per month at auction and resells them.

The goal is not simply to predict a car's price. The system must convert a predicted market value into a **recommended maximum bid**, while explicitly accounting for the financial cost of overpaying versus losing an auction.

---

## 🎯 Client Problem

Viktor needs a reliable answer to:

> **"What is the maximum amount I should bid for this car?"**

The system therefore needs to:

- Clean and validate incoming vehicle data
- Predict the expected vehicle price
- Quantify uncertainty in the prediction
- Determine an economically rational bidding percentile
- Calculate a recommended maximum bid in **euros**
- Detect listings that are outside the model's training experience
- Identify situations where the model is systematically inaccurate

The final system is designed to provide a practical **bid/no-bid decision**, rather than only a machine-learning prediction.

---

## 📊 Dataset

The project uses a dataset containing **4,255 used-car listings** and 15 columns.

### Main features

| Feature | Description |
|---|---|
| `listing_id` | Unique listing identifier |
| `brand` | Vehicle brand |
| `model_year` | Vehicle manufacturing year |
| `mileage_km` | Vehicle mileage |
| `engine_litres` | Engine size |
| `fuel_type` | Fuel type |
| `transmission` | Transmission type |
| `body_type` | Vehicle body type |
| `owners` | Number of previous owners |
| `service_history` | Service history information |
| `accident_reported` | Whether an accident was reported |
| `city` | Listing location |
| `days_on_market` | Number of days listed |
| `price_eur` | Vehicle price in euros |
| `sold_fast` | Whether the vehicle sold quickly |

### Important leakage rule

`days_on_market` is **not used as a model feature** because it is only known after the vehicle has been on the market.

Using it to predict the auction decision would introduce **data leakage**.

---

## 🧹 Data Cleaning

The raw dataset contains several issues that need to be handled before modelling.

Examples include:

- Missing mileage, engine size, owners and service-history values
- Negative mileage values
- Extremely large mileage values
- Unrealistic model years
- Inconsistent capitalization and whitespace in categorical variables
- Duplicate records
- Missing categorical values

The cleaning pipeline:

1. Corrects recoverable mileage errors
2. Converts impossible values to missing values
3. Standardizes categorical values
4. Creates a new `age_years` feature
5. Leaves missing values for the modelling pipeline to impute

The current cleaning process reduces the dataset from **4,255 raw rows to 4,200 cleaned rows**.

---

## 🧠 Feature Engineering

A new feature is created:

```text
age_years = 2026 - model_year
```

Vehicle age is more useful for the pricing model than the raw manufacturing year.

The model uses:

### Numerical features

```text
age_years
mileage_km
engine_litres
owners
accident_reported
```

### Categorical features

```text
brand
fuel_type
transmission
body_type
city
service_history
```

---

## 🏗️ Machine Learning Pipeline

The project uses a Scikit-Learn pipeline consisting of:

```text
Raw Data
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Train/Test Split
   ↓
Missing-Value Imputation
   ↓
Feature Scaling
   ↓
One-Hot Encoding
   ↓
Log-Price Regression Model
   ↓
Price Prediction
   ↓
Residual / Percentile Analysis
   ↓
Recommended Maximum Bid
```

Numerical features are:

- Imputed using the median
- Standardized using `StandardScaler`

Categorical features are:

- Filled with `"unknown"`
- Encoded using `OneHotEncoder`

---

## 📈 Target Variable

The model predicts:

```python
log(price_eur)
```

instead of directly predicting:

```python
price_eur
```

The predicted value is converted back to euros using:

```python
np.exp(prediction)
```

This allows the project to model price multiplicatively and then perform all client-facing financial calculations in **euros**.

---

## 🥇 Baseline Model

Before evaluating the ML model, a simple baseline is created.

The baseline predicts the:

> **Median log-price of vehicles with the same brand and body type**

If a brand/body-type combination is not present in the training data, the global median is used.

The baseline test MAE from the starter project is approximately:

```text
€4,836
```

The final ML system must demonstrate that it improves upon this baseline.

---

## 🤖 Regression Model

The primary model is:

```text
Linear Regression
```

trained on:

```text
log(price_eur)
```

The project can also compare alternative models using cross-validation, but model selection must be performed using the **training data**, not by repeatedly checking the test set.

---

## 🔬 Honest Model Evaluation

The dataset is split into:

```text
75% Training
25% Testing
```

Resulting in approximately:

```text
Training: 3,150 listings
Testing:  1,050 listings
```

The test set is intended to be used **only once at the end** for the final evaluation.

Cross-validation is used on the training data when comparing different modelling approaches.

This prevents the test set from becoming part of the model-selection process.

---

# 💰 Auction Bidding Strategy

This is the central part of the project.

Viktor has two different costs:

```text
Cost of overpaying = €900
Cost of losing an auction = €150
```

The system derives the economically optimal percentile from these two costs.

### Theoretical optimal percentile

The derivation is:

```text
p* = Cost of losing
     -------------------------------
     Cost of overpaying + Cost of losing
```

Therefore:

```text
p* = 150 / (900 + 150)
   ≈ 0.1429
   ≈ 14.3%
```

This provides the theoretically justified starting point for the bidding strategy.

---

## 📉 Percentile Sweep

The project does not blindly accept the theoretical value.

Several candidate percentiles are tested.

For each percentile, the system calculates:

- Number of cars overpaid for
- Number of auctions lost
- Mean euros overpaid
- Mean euros of margin lost
- Total financial loss

The total cost combines:

```text
€900 × number of overpaid cars
+
€150 × number of lost auctions
```

The percentile with the lowest total cost becomes the **empirically preferred bidding percentile**.

The theoretical result and the measured sweep are then compared.

---

## 💶 Recommended Maximum Bid

The final bidding system:

1. Predicts the log-price
2. Estimates the residual distribution using held-out training data
3. Selects the required residual percentile
4. Applies the percentile offset to the predicted log-price
5. Converts the result back to euros

Conceptually:

```text
Recommended Bid
=
exp(predicted log-price + percentile residual offset)
```

The output is therefore a **maximum bid in euros**, rather than simply a predicted vehicle price.

---

# 🛡️ Trust Check

A prediction should not automatically be trusted when a new vehicle looks very different from the training data.

The system checks for:

### Numeric outliers

Examples:

- Mileage outside the training range
- Engine size outside the training range
- Vehicle age outside the training range
- Number of owners outside the training range

### Unseen categories

Examples:

- Previously unseen brand
- Previously unseen fuel type
- Previously unseen city

### Excessive missing information

Listings with too many blank input fields are flagged.

Each flagged listing receives a reason explaining why it should not be trusted.

---

## 🚨 Policy for Untrusted Listings

The system uses a conservative policy:

> **Do not automatically place a bid on an untrusted listing.**

Instead, the listing is flagged for **human review**.

This prevents the system from producing a confident-looking bid for a vehicle that falls outside the experience represented in the training data.

---

# 🔍 Known Failure Analysis

The project also investigates where the model performs poorly.

Potential segments include:

- Small-engine vehicles
- Large-engine vehicles
- Different vehicle-age groups
- Specific brand/body-type combinations

For each important failure segment, the analysis should report:

1. The affected segment
2. Direction of the error
3. Size of the error
4. Why the error may occur
5. What the error means for Viktor's bidding decision

This is important because overall model accuracy alone does not tell the client **where the model can be trusted**.

---

# 🧪 Project Validation

The project follows several important ML evaluation principles:

- Split data before model comparison
- Keep the test set untouched during model selection
- Use cross-validation on training data
- Compare against a simple baseline
- Never use `days_on_market` as an input feature
- Convert predictions back to euros before financial calculations
- Evaluate the bidding strategy using actual financial costs
- Test whether trust-check flags correspond to genuinely worse predictions
- Identify specific model failure segments

---

# 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn
- Jupyter Notebook

### Scikit-Learn components

- `Pipeline`
- `ColumnTransformer`
- `SimpleImputer`
- `StandardScaler`
- `OneHotEncoder`
- `LinearRegression`
- `train_test_split`
- `cross_val_score`
- `KFold`
- Regression evaluation metrics

---

# 📁 Project Structure

```text
ML-Auction-Bid-Recommendation/
│
├── STARTER_MLAP_PROJECT.ipynb
├── car_listings_for_project.csv
├── Client1.pdf
└── README.md
```

---

# 🎯 Final Objective

The final system is intended to answer:

> **"Given this vehicle listing, what is the maximum amount Viktor should bid?"**

while also answering:

- How conservative is the recommendation?
- Why was this percentile selected?
- How much does the strategy cost when it is wrong?
- When should Viktor avoid trusting the prediction?
- Where does the model systematically fail?

The project therefore focuses on turning a machine-learning price prediction into a **real-world decision system**.

---

## 📌 Project Status

This repository is based on the provided MLAP starter project and client brief.

The starter notebook provides the data-cleaning framework, preprocessing pipeline, baseline, regression setup, bidding-strategy framework, trust-check framework and failure-analysis framework.

The remaining work is to complete and validate the client-specific analysis, including the final percentile sweep, recommended bid policy, trust-check validation, sensitivity analysis and documented failure segments.

---

## 👤 Client

**Viktor — Auction Car Buyer**

Business objective:

> Buy vehicles at auction while minimizing the combined cost of overpaying and missing profitable purchases.

---

## 🎓 Context

**Machine Learning Program — MIPT International Autumn School 2026**

This project demonstrates the application of a complete ML workflow:

```text
Data Understanding
       ↓
Data Cleaning
       ↓
Feature Engineering
       ↓
Train/Test Split
       ↓
Baseline
       ↓
Model Training
       ↓
Cross-Validation
       ↓
Model Evaluation
       ↓
Uncertainty Analysis
       ↓
Business Decision
       ↓
Trust & Failure Analysis
```


/n 
# 💰 Auction Bidding Calculation

The main objective is to determine **how conservative Viktor's maximum bid should be**.

Viktor has two possible mistakes:

1. **Overpaying** for a car — this costs him **€900**.
2. **Losing the auction** because his bid was too low — this costs him **€150** in missed margin.

Therefore, the model must balance these two costs.

---

## 1. Cost of the Two Possible Mistakes

```text
Overpaying cost = €900
Losing-auction cost = €150
```

Total cost of either type of mistake:

```text
€900 + €150 = €1,050
```

---

## 2. Deriving the Optimal Percentile

The rational target percentile is calculated from the two costs:

```text
Target percentile
=
Cost of losing the auction
--------------------------------
Cost of overpaying + Cost of losing
```

Substituting Viktor's costs:

```text
Target percentile
=
€150
----------------
€900 + €150
```

```text
=
150
----
1050
```

```text
= 0.142857
```

Converting to percentage:

```text
0.142857 × 100
≈ 14.3%
```

### Theoretical result

> **The theoretically optimal target percentile is approximately 14.3%.**

This means the bidding strategy should initially target approximately the **14th percentile of the predicted price distribution**.

---

## 3. Why Does a Lower Percentile Make Sense?

Viktor's cost of overpaying is much higher than his cost of losing an auction:

```text
Overpaying:       €900
Losing auction:   €150
```

Since:

```text
€900 > €150
```

the strategy should be **conservative**.

In other words:

> It is better for Viktor to occasionally miss an auction than to regularly overpay by a large amount.

That is why the theoretical calculation produces a relatively low percentile of approximately **14.3%**.

---

## 4. Converting the Model Prediction into a Bid

The model predicts:

```text
log(price_eur)
```

The prediction is first converted back into euros.

For example, if:

```text
Predicted log-price = 8.80
```

then:

```text
Predicted price
=
e^8.80
≈ €6,631
```

The model then uses the residual/error distribution to determine how much the prediction should be adjusted.

For a selected percentile:

```text
Adjusted log-price
=
Predicted log-price
+
Residual percentile
```

Then:

```text
Maximum bid
=
exp(Adjusted log-price)
```

Therefore, the final recommendation is always expressed in **euros**.

---

## 5. Example of the Bidding Calculation

Suppose the model predicts:

```text
Predicted price = €10,000
```

and the selected residual percentile gives a downward adjustment equivalent to:

```text
Adjusted bid = €8,500
```

Then:

```text
Viktor's maximum bid = €8,500
```

So even though the model estimates the car's market value at approximately:

```text
€10,000
```

the system recommends:

```text
Maximum bid = €8,500
```

because Viktor needs a conservative bid that protects him against the cost of overpaying.

**Note:** The €10,000 → €8,500 numbers above are only an illustration of the calculation. The actual project recommendation must come from the model's measured residual distribution.

---

## 6. Measuring Different Percentiles

The theoretical 14.3% is not automatically accepted as the final answer.

The project tests several candidate percentiles, for example:

```text
10%
15%
20%
25%
30%
...
```

For each percentile, the strategy is evaluated on the **test set**.

For every candidate percentile we calculate:

### Mean overpayment

How many euros Viktor loses, on average, when the recommended bid causes him to overpay.

```text
Mean overpayment
=
Total euros overpaid
--------------------
Number of overpaid cars
```

### Mean margin lost

How many euros Viktor loses, on average, when the recommended bid is too conservative and he misses a profitable purchase.

```text
Mean margin lost
=
Total euros of missed margin
----------------------------
Number of missed auctions
```

---

## 7. Total Cost of a Percentile

For each candidate percentile, the two failure modes are combined using Viktor's costs.

```text
Total cost
=
(€900 × Number of overpaid cars)
+
(€150 × Number of lost auctions)
```

For example, suppose a particular percentile produces:

```text
10 overpaid cars
20 lost auctions
```

Then:

```text
Overpaying cost
=
10 × €900
=
€9,000
```

and:

```text
Lost-auction cost
=
20 × €150
=
€3,000
```

Therefore:

```text
Total cost
=
€9,000 + €3,000
=
€12,000
```

The same calculation is repeated for every candidate percentile.

---

## 8. Selecting the Final Percentile

The final percentile is the one with the **lowest total financial cost** on the evaluation.

For example:

| Percentile | Overpaid Cars | Lost Auctions | Total Cost |
|---:|---:|---:|---:|
| 10% | 8 | 30 | €11,700 |
| 15% | 10 | 20 | €12,000 |
| 20% | 13 | 15 | €14,250 |
| 25% | 16 | 10 | €15,900 |

In this **illustrative example**, the 10% percentile would be selected because:

```text
€11,700
```

is the lowest total cost.

The actual project must use the **measured results from the test set**, not these example numbers.

---

## 9. Comparing Theory vs. Measurement

There are therefore two answers:

### Theoretical answer

Calculated directly from Viktor's costs:

```text
150 / (900 + 150)
≈ 14.3%
```

### Empirical answer

Found by sweeping several percentiles and selecting the one with the lowest measured total cost.

The project should compare these two results.

If they are close:

> The theoretical derivation is supported by the measured experiment.

If they differ:

> The difference should be investigated rather than simply choosing one number.

This comparison provides stronger evidence for the final bidding strategy.

---

## 10. Sensitivity Analysis

Viktor's €900 overpayment cost is an estimate, so the recommendation should also be tested under different assumptions.

### Scenario A — Overpayment cost = €600

```text
Target percentile
=
150 / (600 + 150)
=
150 / 750
=
20%
```

### Scenario B — Overpayment cost = €900

```text
Target percentile
=
150 / (900 + 150)
≈ 14.3%
```

### Scenario C — Overpayment cost = €1,500

```text
Target percentile
=
150 / (1500 + 150)
≈ 9.1%
```

This shows the business logic clearly:

| Overpayment Cost | Losing Cost | Theoretical Percentile |
|---:|---:|---:|
| €600 | €150 | 20.0% |
| €900 | €150 | 14.3% |
| €1,500 | €150 | 9.1% |

### Interpretation

As the cost of overpaying increases:

```text
Overpayment becomes more expensive
            ↓
Viktor should become more conservative
            ↓
Recommended percentile decreases
            ↓
Maximum bid becomes lower
```

This sensitivity analysis shows how the bidding recommendation changes when Viktor's business assumptions change.