from django.urls import path
from .views import *  # Import the view function

urlpatterns = [
    path("predict_cgpa/", predict_cgpa, name="predict_cgpa"),
    path("result/", result, name="result"),
    path("predict/", predict_performance, name="predict"),
    path("student_aid/", student_aid, name="student_aid")
]
