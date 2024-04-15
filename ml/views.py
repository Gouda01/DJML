from django.shortcuts import render

from .models import Iris
from .forms import IrisForm



# Create your views here.

def predict(request):

    form = IrisForm()
    return render(request,'predict.html',{'form':form})
