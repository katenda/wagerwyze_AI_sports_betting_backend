from django.urls import path
from .views import *
from django.contrib.auth import views as auth_views


urlpatterns = [
    path('get_teams_data', get_teams_data, name = 'get_teams_data'),
    path('make_prediction', make_prediction, name = 'make_prediction'),
    path('get_current_statistics', get_current_statistics, name = 'get_current_statistics'),
    path('user_signup', user_signup, name = 'user_signup'),
    path('user_login', user_login, name = 'user_login'),
    path('get_predictions', get_predictions, name = 'get_predictions'),

]

