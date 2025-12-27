# Titanic ML Project (Python)

Predict passenger survival on the **Kaggle Titanic** dataset using:
- a **baseline model**
- a **Random Forest** model

✅ This repo contains code + setup files  
❌ Dataset files (`train.csv`, `test.csv`) are **not included** (kept private)

---

## Project Overview
This is my first end-to-end machine learning project:
- Load Titanic data
- Clean / preprocess features
- Train models
- Evaluate performance
- (Optional) Create a Kaggle-style submission file

---

## Files
- `train_baseline.py` — baseline training + evaluation  
- `train_rf.py` — Random Forest training + evaluation  
- `main.py` — (optional) main runner / experiments  
- `requirements.txt` — minimal dependencies  
- `.gitignore` — ignores datasets and local environment files  

---

## Requirements
- Python 3.9+ recommended

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Dataset (Not Included)
Download the Titanic dataset from Kaggle and place the files locally.

**Expected files:**
- `train.csv`
- `test.csv`
- (optional) `gender_submission.csv`

📌 Put them in the same folder as the scripts (project root), unless you changed the paths in code.

---

## How to Run

### 1) Baseline model
```bash
python train_baseline.py
```

### 2) Random Forest model
```bash
python train_rf.py
```

---

## Example Result
Random Forest validation accuracy (from my run):
- **Accuracy:** ~0.84  
(Exact result may change depending on train/test split and random seed.)

---

## Notes
- If you see an error like “file not found train.csv”, it means you need to download the dataset and place it in the correct folder.
- If you want to generate a submission file, you can edit the scripts to save predictions as `submission.csv`.

---

## Future Improvements
- Better feature engineering (Title extraction, FamilySize, Cabin handling)
- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
- Cross-validation for more reliable evaluation
- Add a notebook for explanation + visuals

---

## License
This project is for learning and portfolio purposes.
