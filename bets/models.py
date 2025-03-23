from django.db import models


# Create your models here.
class Client(models.Model):
    username=models.CharField(max_length=200,blank=True,null=True)
    password=models.CharField(max_length=200,blank=True,null=True)    
    def __str__(self):
        return self.username

class Predictions(models.Model):
    client=models.ForeignKey(Client,on_delete=models.CASCADE,blank=True,null=True)
    home_team=models.CharField(max_length=200,blank=True,null=True)    
    away_team=models.CharField(max_length=200,blank=True,null=True)    
    winner=models.CharField(max_length=200,blank=True,null=True)    
    loser=models.CharField(max_length=200,blank=True,null=True)    
    date=models.CharField(max_length=200,blank=True,null=True)    
    def __str__(self):
        return self.home_team