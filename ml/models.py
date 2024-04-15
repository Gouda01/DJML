from django.db import models



# Create your models here.
class Iris(models.Model):
    sepal_length = models.FloatField()
    sepal_width = models.FloatField()
    petal_length = models.FloatField()
    petal_width = models.FloatField()

    classification = models.CharField(max_length=20, null=True, blank=True)

    def __str__(self):
        return self.classification
    