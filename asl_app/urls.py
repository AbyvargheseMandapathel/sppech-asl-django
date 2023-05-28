from django.urls import path
from asl_app import views

urlpatterns = [
    path('', views.home, name='home'),
    path('record/', views.record_data, name='record'),
    path('save/', views.save_recorded_video, name='save_recorded_video'),
    path('train/', views.train_data, name='train'),
]
