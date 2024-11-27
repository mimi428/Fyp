from django.db import models
import datetime


#Category Model
class Category(models.Model):
    name = models.CharField(max_length=200)
    def __str__(self):
		    return self.name
    
class Meta:
    verbose_name_plural='categories'


#Customer Model
class Customer(models.Model):
    first_name=models.CharField(max_length=50)
    last_name=models.CharField(max_length=50)
    phone=models.CharField(max_length=10)
    email=models.EmailField(max_length=100)
    password=models.CharField(max_length=100)

    def __str__(self):
        return f'{self.first_name}{self.last_name}'


#Product Model
class Product(models.Model):
    name=models.CharField(max_length=100)
    price=models.DecimalField(default=0,decimal_places=2,max_digits=6)
    category=models.ForeignKey(Category,on_delete=models.CASCADE,default=1)
    description=models.CharField(max_length=250,default='',blank=True,null=True)
    image=models.ImageField(upload_to='uploads/product')

    #add sale stuff
    is_sale=models.BooleanField(default=False)
    sale_price=models.DecimalField(default=0,decimal_places=2,max_digits=6)

    def __str__(self):
		    return self.name

#Order Model
class Order(models.Model):
    product=models.ForeignKey(Product,on_delete=models.CASCADE)
    customer=models.ForeignKey(Customer,on_delete=models.CASCADE)
    quantity=models.IntegerField(default=1)
    address=models.CharField(max_length=100,default='',blank=True)
    phone=models.CharField(max_length=10,default='',blank=True)
    date=models.DateTimeField(default=datetime.datetime.today)
    status=models.BooleanField(default=False)

    def __str__(self):
        return self.product


#for chat history

from django.contrib.auth.models import User

class ChatHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    history = models.TextField()  # JSON or plain text format
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Chat History of {self.user.username}"
