### REQUIREMENT  
```
pip install django  
pip install pickle  
pip install pandas  
```

---

## Load the trained model and encoders  
```python
import pickle

# Load the trained model and encoders  
with open("gb_student_performance_model.pkl", "rb") as model_file:
    gbr = pickle.load(model_file)

with open("gb_label_encoders.pkl", "rb") as encoder_file:
    encoders = pickle.load(encoder_file)

# Load the trained model and encoders  
with open("re_performance_model.pkl", "rb") as rate_model_file:
    model = pickle.load(rate_model_file)

with open("re_encoders.pkl", "rb") as rate_encoder_file:
    rate_encoders = pickle.load(rate_encoder_file)


