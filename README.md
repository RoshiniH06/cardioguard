# CardioGuard - ML part

## Folder layout
```
cardioguard/
  data/cardio_train.csv      <- put the Kaggle file here
  ml/train_model.py          <- trains and saves the model
  ml/test_model.py           <- quick check of the saved model
  backend/main.py            <- FastAPI app
  requirements.txt
```
(`backend/model.joblib` and `backend/meta.json` are created by training.)

## Setup (once)
Open the cardioguard folder in VS Code, then in the terminal:
```
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```
Select this venv as the interpreter in VS Code (Ctrl+Shift+P -> "Python: Select Interpreter").
Use the SAME venv for training and for the API, so the scikit-learn version matches.

## Run
1. Train:      `python ml/train_model.py`   (about 10-20 minutes, the SVC step is the slow one)
2. Test:       `python ml/test_model.py`
3. Start API:  `uvicorn backend.main:app --reload`
4. Open:       http://127.0.0.1:8000/docs   (try POST /predict here)

Charts and result tables are saved in `ml/outputs/`.
