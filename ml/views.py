from django.shortcuts import render
import pandas as pd

from .models import Iris
from .forms import IrisForm



# Create your views here.

def predict(request):

    if request.method == 'POST' :
        form = IrisForm(request.POST)

        if form.is_valid() :

            myform = form.save(commit=False)
            sepal_length = form.cleaned_data['sepal_length']
            sepal_width = form.cleaned_data['sepal_width']
            petal_length = form.cleaned_data['petal_length']
            petal_width = form.cleaned_data['petal_width']

            #
            # Prediction :
            model = pd.read_pickle("model.pickle")
            resault = model.predict([[sepal_length,sepal_width,petal_length,petal_width]])

            classification = resault[0]

            # Save :
            myform.classification = classification
            myform.save()

            return render(request,'predict.html',{'form':form,'resault':classification})
            
    else :
        form = IrisForm()

    return render(request,'predict.html',{'form':form})
