from django.shortcuts import render,get_object_or_404
from .cart import Cart
from ecommerce.models import Product
from django.http import JsonResponse
from django.contrib import messages



def cart_summary(request):
    #get the cart
    cart=Cart(request)
    cart_products=cart.get_prods
    quantities=cart.get_quants
    totals= cart.cart_total()
    sum=cart.total()
    return render(request,"cart_summary.html",{"cart_products": cart_products,"quantities":quantities,"totals":totals,"sum":sum})

def cart_add(request):
    #get the cart
    cart=Cart(request)

    #test for POST
    if request.POST.get('action') == 'post':
        #get stuff
        product_id=int(request.POST.get('product_id'))
        product_qty=int(request.POST.get('product_qty'))

        #lookup product in DB
        product=get_object_or_404(Product,id=product_id)

        #save into a session
        cart.add(product=product,quantity= product_qty)
        
        #get cart quantity
        cart_quantity=cart.__len__()

        #return response
        # response=JsonResponse({'Product Name:':product.name})
        response=JsonResponse({'Quantity:':cart_quantity})
        messages.success(request,("Product added to your cart..."))
        
        return response
    


def cart_delete(request):
    cart=Cart(request)
    if request.POST.get('action') == 'post':
        #get stuff
        product_id=int(request.POST.get('product_id')) 

        #call delete function
        cart.delete(product=product_id)
        response=JsonResponse({'product':product_id})
        messages.success(request,("Items have been deleted in your cart..."))
        return response


def cart_update(request):
    cart=Cart(request)
    if request.POST.get('action') == 'post':
        #get stuff
        product_id=int(request.POST.get('product_id'))
        product_qty=int(request.POST.get('product_qty'))

        cart.update(product=product_id,quantity=product_qty)
        response=JsonResponse({'qty':product_qty})
        messages.success(request,("Product has been updated in your cart..."))
        return response
    