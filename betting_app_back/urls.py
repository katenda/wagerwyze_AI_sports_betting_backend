

from django.conf import settings
from django.contrib import admin
from django.urls import path,include,re_path
from django.views.static import serve

urlpatterns = [
    path('admin/', admin.site.urls),
    path('app-dataaa-api/', include('bets.urls')),

]
